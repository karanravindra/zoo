"""A small Llama-style decoder for TinyStories: its config, the model, and the checkpoint format.

Pre-norm RMSNorm blocks, rotary position embeddings, SwiGLU MLPs, bias-free linears, causal attention through
`F.scaled_dot_product_attention`, and an output projection tied to the input embedding.

`forward(idx, return_hidden=True)` returns the final normed hidden states instead of logits, so the training loss can
be computed from `model.classifier_weight` without materializing the (batch, block, vocab) logits.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention
from torch.utils.checkpoint import checkpoint


@dataclass
class Config:
    # model (about 5.9M parameters with the defaults)
    vocab_size: int = 4096  # must match the tokenizer from prepare.py
    dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    block_size: int = 256  # context length
    multiple_of: int = 32  # the SwiGLU hidden size (about 8/3 * dim) is rounded up to a multiple of this
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    # documents: by default a training window is a random slice of the concatenated stories with plain causal attention
    doc_mask: bool = False  # attention stays inside a story: a token cannot see the stories before its own in the window
    attn_window: int = 0  # attention lookback: a token sees only the last N tokens, itself included (e.g. 512 with a longer block_size); 0 = the whole context
    attn_global_every: int = 0  # with attn_window: every Nth layer (the Nth, 2Nth, ..., counting from 1) attends over the whole context instead of the window (e.g. 5 with 6 layers: the fifth layer only); 0 = no global layers
    doc_align: Literal["off", "pack", "pad"] = "off"  # where windows start: off = anywhere; pack = at a story start, continuing through the next stories (no padding); pad = one story per row, padded, padding left out of the loss
    # optimization (steps, not epochs)
    batch_size: int = 64  # sequences per micro-batch
    grad_accum: int = 1  # micro-batches per optimizer step
    auto_batch: bool = True  # if the GPU runs out of memory before the first checkpoint, restart with the micro-batch split (batch_size / 2, grad_accum * 2, ...) so tokens per optimizer step stay the same
    max_steps: int = 5000  # optimizer steps
    flop_budget: float = 0.0  # training FLOPs to spend (e.g. 3e16); > 0 replaces max_steps with the number of steps that fit
    time_budget_s: float = 0.0  # training seconds to spend (eval, sampling and checkpointing excluded); > 0 trains until it is used up, ignoring max_steps, with the LR schedule following elapsed time
    lr: float = 1e-3
    min_lr_ratio: float = 0.1  # the cosine decays to lr * min_lr_ratio
    warmup_steps: int = 200
    weight_decay: float = 0.1  # applied to 2D parameters only
    beta2: float = 0.95
    grad_clip: float = 1.0  # global grad norm; 0 disables
    ema: bool = False  # keep an exponential moving average of the weights; evals, samples and checkpoints (best.pt, last.pt) use the averaged weights
    ema_frac: float = 0.1  # EMA window as a fraction of the optimizer steps taken so far (0.1 = averages roughly the last 10% of training), so it is the same share of training at every scale and batch size
    # evaluation
    eval_every: int = 250  # steps between eval rows (val loss, checkpoints)
    eval_batches: int = 50  # val batches per eval, drawn with a fixed seed so rows are comparable
    val_source_batches: int = 20  # val batches per source at each eval, on the held-out stories of every source in the mix (prepare.py's validation_<source>/ dirs; one val_<source>_loss / _bpb column pair each); 0 skips them
    log_every: int = 20  # steps between metrics.csv rows and progress-bar updates (each forces a device sync)
    stats_every: int = 100  # steps between detailed statistics (weight norms, update ratios, gradient direction, per layer in layers.csv); 0 disables
    sample_every: int = 250  # steps between text samples written to samples.txt, independent of eval_every; 0 disables
    sample_prompt: str = "Once upon a time"
    sample_tokens: int = 200
    sample_temperature: float = 0.8
    sample_top_k: int = 40
    # runtime
    run_name: str = ""  # run directory name; default is built from the model shape and budget, e.g. d256-l6-ctx512-3e16 ("-2", ... is appended if it exists)
    seed: int = 0  # init, batch and sampling seed
    deterministic: bool = False  # deterministic algorithms (slower)
    bf16: bool = True  # bfloat16 autocast (cuda only)
    loss: Literal["auto", "cce", "torch_compile", "torch"] = "auto"  # cut cross-entropy impl; auto = cce on cuda with bf16, else torch_compile; torch = plain F.cross_entropy
    compile: bool = True  # torch.compile the model with mode="reduce-overhead" (CUDA graphs on GPU); the first window includes compilation; FlexAttention (doc_mask, attn_window) is only fast when compiled; --no-compile turns it off
    grad_checkpoint: bool = False  # activation checkpointing: each block's activations are recomputed in the backward pass instead of stored; much less memory (so a larger micro-batch fits), about a third more compute
    dev: bool = False  # dev run: outputs go to a temporary directory deleted on exit, so nothing is logged to runs/

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Config:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def ema_decay_at(self, step: int) -> float:
        """EMA decay for the update after optimizer step `step`: the window 1/(1 - decay) is 1 + ema_frac * step
        steps. It starts at 1 (the average is the weights, not dominated by the initial ones) and grows with the
        run, with no cap, so a long run keeps the same share of training in its window."""
        return 1 - 1 / (1 + self.ema_frac * step)

    @property
    def hidden_dim(self) -> int:
        h = int(2 * (4 * self.dim) / 3)
        return self.multiple_of * ((h + self.multiple_of - 1) // self.multiple_of)

    def layer_window(self, layer: int) -> int:
        """Attention lookback of block `layer` (counted from 0): `attn_window`, or 0 (the whole context) when
        there is no window or the block is one of the global layers of `attn_global_every`."""
        every = self.attn_global_every
        return 0 if every and (layer + 1) % every == 0 else self.attn_window

    @property
    def default_run_name(self) -> str:
        """d<dim>-l<layers>-ctx<block_size>-<budget>, the budget being whichever one ends the run: seconds
        (`600s`), FLOPs (`3e16`) or thousands of steps (`20k`, zero-padded so 02k sorts before 10k); then
        `-win<N>` for an attention window (`-win<N>g<M>` with a global layer every M), `-mask` and / or `-pack` / `-pad` for the document options."""
        if self.time_budget_s > 0:
            budget = f"{self.time_budget_s:g}s"
        elif self.flop_budget > 0:
            budget = f"{self.flop_budget:.3g}".replace("e+", "e")
        else:
            k = self.max_steps / 1000
            budget = f"{int(k):02d}k" if k == int(k) else f"{k:g}k"
        glob = f"g{self.attn_global_every}" if self.attn_global_every else ""
        docs = (f"-win{self.attn_window}{glob}" if self.attn_window else "") + ("-mask" if self.doc_mask else "")
        docs += "" if self.doc_align == "off" else f"-{self.doc_align}"
        return f"d{self.dim}-l{self.n_layers}-ctx{self.block_size}-{budget}{docs}"

    @property
    def forward_flops_per_token(self) -> float:
        """FLOPs of one forward pass per token at the full context length, counting matmuls only, 2 FLOPs per
        multiply-add (the convention behind the "6 * params * tokens" rule). Norms, RoPE, activations and the
        embedding lookup are left out; they are a few percent at most. An `attn_window` shortens the attention
        term to the window, except in the global layers of `attn_global_every`. `doc_mask` does not change the count, although FlexAttention skips fully masked
        blocks, so it is an upper bound there; with `doc_align="pad"` padded positions count too."""
        d = self.dim
        spans = [min(self.block_size, self.layer_window(i) or self.block_size) for i in range(self.n_layers)]
        attn = sum(8 * d * d + 4 * t * d for t in spans)  # q, k, v, o projections; QK^T and the weighted sum over the t positions a layer sees
        mlp = 6 * d * self.hidden_dim  # the three SwiGLU matrices
        return attn + self.n_layers * mlp + 2 * self.vocab_size * d  # plus the output projection

    @property
    def train_flops_per_token(self) -> float:
        """Forward + backward, with the usual estimate that the backward pass costs two forwards."""
        return 3 * self.forward_flops_per_token


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
    def __init__(self, cfg: Config, layer: int):
        super().__init__()
        assert cfg.dim % cfg.n_heads == 0
        self.n_heads, self.head_dim = cfg.n_heads, cfg.dim // cfg.n_heads
        self.window = cfg.layer_window(layer)  # 0 in a global layer: the whole context
        self.wq = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wo = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, cache: dict | None = None, mask: BlockMask | torch.Tensor | None = None
    ) -> torch.Tensor:
        """Three paths. With a `mask` (story mask and / or attention window, from `Llama.attention_mask`):
        FlexAttention for a BlockMask, which skips the blocks the mask rules out, or plain attention with a dense
        boolean mask off CUDA. With a `cache` (generation only): this layer's rotated keys and values for the
        tokens seen so far are kept in it, the new ones are appended, and the new queries attend over them (the
        last `attn_window` of them, if this layer has a window; a global layer keeps them all). Otherwise: plain
        causal attention."""
        b, t, _ = x.shape
        q, k, v = (w(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2) for w in (self.wq, self.wk, self.wv))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if isinstance(mask, BlockMask):
            y = flex_attention(q, k, v, block_mask=mask)
        elif mask is not None:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        elif cache is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif "k" in cache:
            assert t == 1, "with a primed cache, feed one token at a time"
            k, v = torch.cat([cache["k"], k], dim=2), torch.cat([cache["v"], v], dim=2)
            if self.window:
                k, v = k[:, :, -self.window :], v[:, :, -self.window :]  # older positions can never be seen again
            cache["k"], cache["v"] = k, v
            y = F.scaled_dot_product_attention(q, k, v)  # the single new query may see every cached position
        else:  # priming the cache with the whole prompt
            cache["k"], cache["v"] = (k[:, :, -self.window :], v[:, :, -self.window :]) if self.window else (k, v)
            if self.window and t > self.window:
                band = torch.ones(t, t, dtype=torch.bool, device=x.device).tril().triu(1 - self.window)
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=band)
            else:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
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
    def __init__(self, cfg: Config, layer: int):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = Attention(cfg, layer)
        self.mlp_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, cache: dict | None = None, mask: BlockMask | torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, cache, mask)
        return x + self.mlp(self.mlp_norm(x))


class Llama(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
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

    def attention_mask(self, idx: torch.Tensor, eot_id: int | None, window: int) -> BlockMask | torch.Tensor | None:
        """The attention mask for this batch and a layer with lookback `window` (0 = none), or None when attention is plain causal: a FlexAttention BlockMask on
        CUDA, and the same mask as a dense boolean (B or 1, 1, T, T) tensor elsewhere (FlexAttention has no
        backward pass on CPU, and is only fast when compiled).

        Causal, plus: with `window`, only the last `window` positions (the query's own included); with
        `eot_id`, only positions in the same story. An end-of-text token closes one story as a target, but as an
        input it is the start token of the next one (that is how generation begins), so it opens the new story
        here: it sees only itself, and the story's tokens see it and each other. Tokens before the first
        end-of-text of a window form a story whose beginning is out of view.

        It is built here, inside `forward`, so that under `model.compile` it is part of the compiled graph: the
        story ids are a tensor of the batch, not something the mask function has to capture from outside."""
        if eot_id is None and not (window and idx.shape[1] > window):
            return None
        doc = (idx == eot_id).cumsum(1) if eot_id is not None else None

        def mask_mod(b, h, q_idx, kv_idx):
            keep = q_idx >= kv_idx
            if window:
                keep = keep & (q_idx - kv_idx < window)
            if doc is not None:
                keep = keep & (doc[b, q_idx] == doc[b, kv_idx])
            return keep

        b, t = idx.shape
        if idx.device.type == "cuda":
            return create_block_mask(mask_mod, b if doc is not None else 1, 1, t, t, device=idx.device)
        rows, pos = torch.arange(b, device=idx.device), torch.arange(t, device=idx.device)
        return mask_mod(rows[:, None, None], 0, pos[None, :, None], pos[None, None, :])[:, None]

    def forward(
        self, idx: torch.Tensor, return_hidden: bool = False, cache: list[dict] | None = None, eot_id: int | None = None
    ) -> torch.Tensor:
        """idx: (B, T) token ids with T <= block_size. Returns logits (B, T, vocab), or with `return_hidden`
        the final normed hidden states (B, T, dim).

        `cache` is a KV cache for generation: a list of one dict per block (start with empty dicts). The first call
        primes it with the whole context; later calls pass only the new token, whose position is the cache length.
        The cached length plus T must stay <= block_size.

        `eot_id` (not together with `cache`) turns on the story mask of `attention_mask`; pass it when training or
        evaluating with `doc_mask`. An `attn_window` is applied on every path without being asked for, in every
        layer but the global ones (`attn_global_every`), which get the mask without the window."""
        assert cache is None or eot_id is None
        windows = {block.attn.window for block in self.blocks}  # at most two: the window, and 0 for global layers
        masks = {w: self.attention_mask(idx, eot_id, w) if cache is None else None for w in windows}
        t = idx.shape[1]
        pos = cache[0].get("pos", 0) if cache else 0  # tokens fed so far (the cached keys may be cut to attn_window)
        if cache:
            cache[0]["pos"] = pos + t
        cos, sin = self.rope_cos[pos : pos + t], self.rope_sin[pos : pos + t]
        x = self.tok_emb(idx)
        recompute = self.cfg.grad_checkpoint and self.training and cache is None and torch.is_grad_enabled()
        for i, block in enumerate(self.blocks):
            mask = masks[block.attn.window]
            if recompute:  # the closure (not an argument) carries the mask: a BlockMask is not a tensor
                x = checkpoint(lambda h, block=block, mask=mask: block(h, cos, sin, None, mask), x, use_reentrant=False, preserve_rng_state=False)
            else:
                x = block(x, cos, sin, cache[i] if cache is not None else None, mask)
        h = self.norm(x)
        return h if return_hidden else F.linear(h, self.classifier_weight)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_id: int | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        hidden_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        kv_cache: bool = True,
    ) -> torch.Tensor:
        """Append up to `max_new_tokens` sampled tokens to idx (B, T); temperature 0 is greedy.

        `generator` may be a list with one generator per row.

        Stops early once every row has produced `eos_id`; a row that has finished is padded with `eos_id`. Only the
        last position's logits are computed.

        With `kv_cache` each step runs the model on the one new token against cached keys / values, through the
        eager `forward` (so a compiled module is not recompiled for every length). Once the context reaches
        `block_size` the cache is dropped and every step recomputes the last `block_size` tokens, which is the
        same sliding window as without a cache. `hidden_fn` replaces the uncached `self(idx, return_hidden=True)`,
        e.g. with the compiled module; passing it turns the cache off."""
        was_training = self.training
        self.eval()
        bs = self.cfg.block_size
        cache = [{} for _ in self.blocks] if kv_cache and hidden_fn is None else None
        hidden_fn = hidden_fn or (lambda i: self(i, return_hidden=True))
        done = torch.zeros(len(idx), dtype=torch.bool, device=idx.device)
        for step in range(max_new_tokens):
            if cache is not None and idx.shape[1] > bs:
                cache = None  # context is full: fall back to the sliding window
            if cache is None:
                h = hidden_fn(idx[:, -bs:])
            elif step == 0:
                h = self.forward(idx, return_hidden=True, cache=cache)  # prime with the whole prompt
            else:
                h = self.forward(idx[:, -1:], return_hidden=True, cache=cache)
            logits = F.linear(h[:, -1], self.classifier_weight).float()  # (B, vocab), last position only
            if temperature == 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = logits.topk(min(top_k, logits.shape[-1])).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, -float("inf"))
                probs = logits.softmax(-1)
                if isinstance(generator, list):  # one generator per row: a row's sample does not depend on the other rows in the batch
                    nxt = torch.stack([torch.multinomial(p, 1, generator=g) for p, g in zip(probs, generator)])
                else:
                    nxt = torch.multinomial(probs, 1, generator=generator)
            if eos_id is not None:
                nxt = nxt.masked_fill(done[:, None], eos_id)
                done |= nxt[:, 0] == eos_id
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and done.all():
                break
        self.train(was_training)
        return idx


def param_group(name: str) -> str:
    """Group label of a parameter for per-layer statistics: `emb`, `L<i>.attn`, `L<i>.mlp`, or `norms`."""
    parts = name.split(".")
    if parts[0] == "tok_emb":
        return "emb"
    if parts[0] == "blocks" and parts[2] in ("attn", "mlp"):
        return f"L{parts[1]}.{parts[2]}"
    return "norms"


def build_model(cfg: Config) -> Llama:
    return Llama(cfg)


def save_checkpoint(path: Path, model: nn.Module, cfg: Config, step: int, metrics: dict) -> None:
    """Write a self-describing checkpoint: weights + the config that built them."""
    tmp = path.with_suffix(".tmp")
    torch.save({"config": cfg.to_dict(), "step": step, "metrics": metrics, "model": model.state_dict()}, tmp)
    tmp.replace(path)


def load_checkpoint(path: str | Path, map_location="cpu") -> tuple[Llama, Config, dict]:
    """Rebuild the model from a checkpoint's own config. Returns (model, config, raw checkpoint)."""
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    cfg = Config.from_dict(ckpt["config"])
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    return model, cfg, ckpt
