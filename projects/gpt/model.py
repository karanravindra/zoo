"""A Llama-style decoder: pre-norm RMSNorm, rotary embeddings, SwiGLU, bias-free linears, tied input/output embedding.

Attention is `F.scaled_dot_product_attention`, which runs FlashAttention on CUDA in bf16. With `doc_mask` a token only
sees its own document and attention goes through FlexAttention instead (a BlockMask skips what the mask rules out).

`forward` returns the final normed hidden states, not logits: the training loss is computed from them and
`classifier_weight` by a cut cross-entropy, so the (batch, block, vocab) logits never exist. `logits` projects hidden
states when they are needed (sampling, one position at a time).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

from config import Config


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


def rope_tables(head_dim: int, block_size: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """cos / sin tables of shape (block_size, head_dim) for the rotate-half convention."""
    freqs = 1.0 / theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
    angles = torch.outer(torch.arange(block_size).float(), freqs)
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, head_dim); cos / sin: (T, head_dim)."""
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos.type_as(x) + rotated * sin.type_as(x)


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.dim % cfg.n_heads == 0
        self.n_heads, self.head_dim = cfg.n_heads, cfg.dim // cfg.n_heads
        self.wq = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wo = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, cache: dict | None = None, mask: BlockMask | torch.Tensor | None = None) -> torch.Tensor:
        """Plain causal attention; with a `mask` (the document mask) FlexAttention, or dense SDPA off CUDA; with a
        `cache` (generation only) this layer's rotated keys and values so far are kept in it and the new queries
        attend over them."""
        b, t, _ = x.shape
        q, k, v = (w(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2) for w in (self.wq, self.wk, self.wv))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        primed = cache is not None and "k" in cache
        if cache is not None:
            if primed:
                assert t == 1, "with a primed cache, feed one token at a time"
                k, v = torch.cat([cache["k"], k], dim=2), torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v

        if isinstance(mask, BlockMask):
            y = flex_attention(q, k, v, block_mask=mask)
        elif mask is not None:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=not primed)
        return self.wo(y.transpose(1, 2).reshape(b, t, -1))


class SwiGLU(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.w1 = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)  # gate
        self.w3 = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)  # up
        self.w2 = nn.Linear(cfg.hidden_dim, cfg.dim, bias=False)  # down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, cache=None, mask=None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, cache, mask)
        return x + self.mlp(self.mlp_norm(x))


class Llama(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        cos, sin = rope_tables(cfg.dim // cfg.n_heads, cfg.block_size, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        for name, p in self.named_parameters():  # GPT-2 style: scale the residual-branch outputs by depth
            if name.endswith(("wo.weight", "w2.weight")):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    @property
    def classifier_weight(self) -> torch.Tensor:
        """(vocab, dim) output projection, tied to the input embedding."""
        return self.tok_emb.weight

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return F.linear(h, self.classifier_weight)

    def doc_mask(self, idx: torch.Tensor) -> BlockMask | torch.Tensor:
        """Causal attention that stays inside a document: a FlexAttention BlockMask on CUDA, the same mask as a dense
        boolean (B, 1, T, T) tensor elsewhere (FlexAttention is only fast compiled, and has no CPU backward).

        An end-of-text token closes a document as a target, but as an input it starts the next one (that is how
        generation begins), so it opens the new document here. Tokens before the first end-of-text of a window form
        a document whose beginning is out of view. Built inside `forward`, so under `compile` it is part of the graph."""
        doc = (idx == self.cfg.eot_id).cumsum(1)

        def mask_mod(b, h, q_idx, kv_idx):
            return (q_idx >= kv_idx) & (doc[b, q_idx] == doc[b, kv_idx])

        b, t = idx.shape
        if idx.device.type == "cuda":
            return create_block_mask(mask_mod, b, 1, t, t, device=idx.device)
        rows, pos = torch.arange(b, device=idx.device), torch.arange(t, device=idx.device)
        return mask_mod(rows[:, None, None], 0, pos[None, :, None], pos[None, None, :])[:, None]

    def forward(self, idx: torch.Tensor, cache: list[dict] | None = None) -> torch.Tensor:
        """idx: (B, T) token ids, T <= block_size. Returns the normed hidden states (B, T, dim).

        `cache` is a KV cache for generation, a list with one dict per block (start with empty dicts). The first call
        primes it with the whole context; later calls pass only the new token, whose position is the cache length.
        The cached length plus T must stay <= block_size. Document boundaries apply with or without a cache."""
        t = idx.shape[1]
        pos = cache[0].get("pos", 0) if cache else 0
        mask = None
        if self.cfg.doc_mask:
            if not pos:
                mask = self.doc_mask(idx)
            if cache:
                positions = torch.arange(pos, pos + t, device=idx.device)
                start = torch.where(idx == self.cfg.eot_id, positions, -1).amax(dim=1)
                if pos:
                    start = torch.maximum(start, cache[0]["document_start"])
                    keys = torch.arange(pos + t, device=idx.device)
                    mask = keys[None, None, None, :] >= start[:, None, None, None]
                cache[0]["document_start"] = start
        if cache:
            cache[0]["pos"] = pos + t
        cos, sin = self.rope_cos[pos : pos + t], self.rope_sin[pos : pos + t]
        x = self.tok_emb(idx)
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, cache[i] if cache is not None else None, mask)
        return self.norm(x)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_id: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Append up to `max_new_tokens` sampled tokens to idx (B, T); temperature 0 is greedy.

        Stops once every row has produced `eos_id`; a finished row is padded with it. Each step runs the one new token
        against the KV cache, through the eager `forward` (a compiled module would recompile at every length). Once the
        context reaches `block_size` the cache is dropped and every step recomputes the last `block_size` tokens."""
        was_training = self.training
        self.eval()
        bs = self.cfg.block_size
        cache = [{} for _ in self.blocks]
        done = torch.zeros(len(idx), dtype=torch.bool, device=idx.device)
        for step in range(max_new_tokens):
            if cache is not None and idx.shape[1] > bs:
                cache = None  # context is full: fall back to the sliding window
            if cache is None:
                h = self.forward(idx[:, -bs:])
            else:
                h = self.forward(idx if step == 0 else idx[:, -1:], cache)
            logits = self.logits(h[:, -1]).float()  # (B, vocab), last position only
            if temperature == 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = logits.topk(min(top_k, logits.shape[-1])).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, -float("inf"))
                nxt = torch.multinomial(logits.softmax(-1), 1, generator=generator)
            if eos_id is not None:
                nxt = nxt.masked_fill(done[:, None], eos_id)
                done |= nxt[:, 0] == eos_id
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and done.all():
                break
        self.train(was_training)
        return idx
