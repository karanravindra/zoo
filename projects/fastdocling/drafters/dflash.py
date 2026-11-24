"""DFlash (arXiv 2602.06036): block-diffusion drafting. One forward pass fills a whole block of draft tokens.

A block is the anchor (the free token x[t+1], clean) followed by k mask tokens. The target's fused low/mid/high hidden states
(a shared projection + RMSNorm) are injected as extra keys and values at every drafter layer, for the positions up to t only
(they bypass the layer's queries and MLP). Inside the block attention is bidirectional, and blocks do not see each other.
The mask token at block position j predicts x[t+1+j] directly (it sits where that token will be). The loss weights position j by
exp(-(j-1) / gamma), because an early miss voids everything after it.

Training packs many randomly sampled anchors per page into one sequence of blocks and one sparse mask (one forward pass), with
anchors resampled every step. Embeddings and LM head are the target's, frozen. Block layers, the projection and the mask
embedding train. `n_layers` is the drafter depth (the paper uses 5 for 4-8B targets).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from drafters.base import HEAD_DIM, DraftOut, Drafter, Features, chunked_ce, register, rope, shifted_targets


class Layer(nn.Module):
    """Pre-norm Llama-style layer whose keys/values also cover the injected target context."""

    def __init__(self, H: int):
        super().__init__()
        self.nh = H // HEAD_DIM
        self.nkv = max(1, self.nh // 3)
        inter = round(8 * H / 3 / 64) * 64
        self.norm1, self.norm2 = nn.RMSNorm(H), nn.RMSNorm(H)
        self.q = nn.Linear(H, self.nh * HEAD_DIM, bias=False)
        self.kv = nn.Linear(H, 2 * self.nkv * HEAD_DIM, bias=False)
        self.o = nn.Linear(self.nh * HEAD_DIM, H, bias=False)
        self.gate_up = nn.Linear(H, 2 * inter, bias=False)
        self.down = nn.Linear(inter, H, bias=False)

    def _kv(self, h):
        B, N, _ = h.shape
        k, v = self.kv(h).view(B, N, 2, self.nkv, HEAD_DIM).permute(2, 0, 3, 1, 4)
        return k, v

    def forward(self, x, ctx, q_pos, ctx_pos, mask):
        """x [B, Nq, H] block tokens, ctx [B, T, H] fused target states, mask [B, 1, Nq, T + Nq] (True = may attend)."""
        kc, vc = self._kv(ctx)
        return self.attend(x, rope(kc, ctx_pos), vc, q_pos, mask)

    def attend(self, x, kc, vc, q_pos, mask=None):
        """The layer given the context keys (already rotated) and values. No mask = everything visible (decoding)."""
        B, Nq, _ = x.shape
        h = self.norm1(x)
        q = self.q(h).view(B, Nq, self.nh, HEAD_DIM).transpose(1, 2)
        kb, vb = self._kv(h)
        rep = self.nh // self.nkv
        k = torch.cat([kc, rope(kb, q_pos)], 2).repeat_interleave(rep, 1)
        v = torch.cat([vc, vb], 2).repeat_interleave(rep, 1)
        a = F.scaled_dot_product_attention(rope(q, q_pos), k, v, attn_mask=mask)
        x = x + self.o(a.transpose(1, 2).reshape(B, Nq, -1))
        g, u = self.gate_up(self.norm2(x)).chunk(2, dim=-1)
        return x + self.down(F.silu(g) * u)


class Backbone(nn.Module):
    """Shared by DFlash and DSpark: target-feature projection, mask embedding, and the block layers."""

    def __init__(self, H: int, n_feature_layers: int, n_layers: int):
        super().__init__()
        self.fc = nn.Linear(n_feature_layers * H, H, bias=False)
        self.ctx_norm = nn.RMSNorm(H)
        self.mask_emb = nn.Parameter(torch.randn(H) * 0.02)
        self.layers = nn.ModuleList(Layer(H) for _ in range(n_layers))
        self.norm_out = nn.RMSNorm(H)

    def context(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.ctx_norm(self.fc(hidden))

    def run(self, ctx, anchor_tok, t, n_in: int, embed) -> torch.Tensor:
        """Blocks of n_in inputs (the anchor x[t+1], then masks) for anchors t [B, A] -> hidden [B, A, n_in, H].
        Block input j sits at position t+1+j and sees the context at positions <= t plus its own block."""
        B, A = t.shape
        T = ctx.shape[1]
        dev = ctx.device
        masks = self.mask_emb.to(ctx.dtype).expand(B, A, n_in - 1, -1)
        x = torch.cat([embed(anchor_tok)[:, :, None].to(ctx.dtype), masks], dim=2).reshape(B, A * n_in, -1)
        q_pos = (t[:, :, None] + 1 + torch.arange(n_in, device=dev)).reshape(B, -1)
        ctx_pos = torch.arange(T, device=dev)
        bid = torch.arange(A, device=dev).repeat_interleave(n_in)
        ctx_mask = ctx_pos[None, None, :] <= t.repeat_interleave(n_in, 1)[:, :, None]  # [B, Nq, T]
        blk_mask = (bid[:, None] == bid[None, :]).expand(B, -1, -1)  # bidirectional inside a block only
        mask = torch.cat([ctx_mask, blk_mask], dim=-1)[:, None]
        for layer in self.layers:
            x = layer(x, ctx, q_pos, ctx_pos, mask)
        return self.norm_out(x).view(B, A, n_in, -1)

    def context_kv(self, hidden: torch.Tensor, start: int):
        """Per-layer (rotated keys, values) of the fused context for positions start.. : what decoding caches."""
        ctx = self.context(hidden)
        pos = torch.arange(hidden.shape[1], device=hidden.device) + start
        out = []
        for layer in self.layers:
            k, v = layer._kv(ctx)
            out.append((rope(k, pos), v))
        return out

    def run_cached(self, kv, anchor_tok, t: int, n_in: int, embed) -> torch.Tensor:
        """One block anchored at position t (all of 0..t is context) from cached context: [B, n_in, H]."""
        B = anchor_tok.shape[0]
        dt = kv[0][1].dtype
        masks = self.mask_emb.to(dt).expand(B, n_in - 1, -1)
        x = torch.cat([embed(anchor_tok)[:, None].to(dt), masks], dim=1)
        q_pos = t + 1 + torch.arange(n_in, device=x.device)
        for layer, (kc, vc) in zip(self.layers, kv):
            x = layer.attend(x, kc, vc, q_pos)
        return self.norm_out(x)


def extend_kv(state, bb: "Backbone", hidden: torch.Tensor):
    """Cache update shared by DFlash and DSpark: add the context keys/values of the new positions. state = (kv, n) or None."""
    start = 0 if state is None else state[1]
    new = bb.context_kv(hidden, start)
    kv = new if state is None else [(torch.cat([a, c], 2), torch.cat([b, d], 2)) for (a, b), (c, d) in zip(state[0], new)]
    return kv, start + hidden.shape[1]


def sample_anchors(valid: torch.Tensor, n: int, seed: int | None) -> torch.Tensor:
    """n anchor positions t per page, drawn from the positions where a block has something to score. valid [B, T] bool.
    A fixed seed (validation) gives the same anchors every time, so epochs are comparable."""
    gen = None
    if seed is not None:
        gen = torch.Generator(device=valid.device).manual_seed(seed)
    w = valid.float()
    w = w + (w.sum(1, keepdim=True) == 0).float()  # a page with nothing valid draws anywhere; its blocks score nothing
    return torch.multinomial(w, n, replacement=bool((w.sum(1) < n).any()), generator=gen)


def position_weights(k: int, gamma: float, device) -> torch.Tensor:
    return torch.exp(-torch.arange(k, device=device, dtype=torch.float32) / gamma)


@register("dflash")
class DFlash(Drafter):
    def __init__(self, cfg, hidden_size: int, n_feature_layers: int):
        super().__init__()
        self.k, self.ce_chunk, self.anchors = cfg.k, cfg.ce_chunk, cfg.anchors
        self.gamma = cfg.loss_gamma or max(2.0, (cfg.k + 1) / 2)
        self.bb = Backbone(hidden_size, n_feature_layers, cfg.n_layers)

    def forward(self, feats: Features) -> DraftOut:
        ids = feats.input_ids
        B, T = ids.shape
        k = self.k
        targets, valid = shifted_targets(ids, feats.mask, k)  # [B, T, k]
        A = self.anchors if self.training else 4 * self.anchors
        t = sample_anchors(valid.any(-1), A, None if self.training else 0)  # [B, A]
        nxt = torch.cat([ids[:, 1:], ids[:, -1:]], dim=1)
        out = self.bb.run(self.bb.context(feats.hidden), nxt.gather(1, t), t, k + 1, feats.embed)  # [B, A, k+1, H]
        h = out[:, :, 1:]  # the mask positions; the anchor's own output is unused
        b = torch.arange(B, device=ids.device)[:, None]
        tg, v = targets[b, t], valid[b, t]  # [B, A, k]
        w = position_weights(k, self.gamma, ids.device).expand(B, A, k)
        loss, m = chunked_ce(h[v], feats.embed.weight, tg[v], self.ce_chunk, weights=w[v])
        match = torch.zeros(B, A, k, dtype=torch.bool, device=ids.device)
        match[v] = m
        match, v = match.view(B * A, k), v.view(B * A, k)
        rows = v.any(-1)
        return DraftOut(loss / w[v.view(B, A, k)].sum().clamp(min=1e-6), match[rows], v[rows])

    @torch.no_grad()
    def propose(self, hidden, last, next_tokens, embed, k):
        k = min(k, self.k)
        t = torch.full((next_tokens.shape[0], 1), next_tokens.shape[1] - 1, device=next_tokens.device)
        h = self.bb.run(self.bb.context(hidden), next_tokens[:, -1:], t, k + 1, embed)[:, 0, 1:]  # [B, k, H]
        return (h @ embed.weight.t().to(h.dtype)).argmax(-1)

    @torch.no_grad()
    def propose_cached(self, state, hidden, last, next_tokens, embed, k):
        kv, n = extend_kv(state, self.bb, hidden)
        k = min(k, self.k)
        h = self.bb.run_cached(kv, next_tokens[:, -1], n - 1, k + 1, embed)[:, 1:]
        return (h @ embed.weight.t().to(h.dtype)).argmax(-1), (kv, n)
