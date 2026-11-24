"""Multi-token-prediction heads: head j predicts x[t+2+j] from the target's final state at t and the embedding of x[t+1].

Medusa/DeepSeek-MTP style, the simplest baseline: heads are independent (no autoregression between draft tokens) and share
the target's frozen LM head, so only the small residual blocks train."""

from __future__ import annotations

import torch
import torch.nn as nn

from drafters.base import DraftOut, Drafter, Features, chunked_ce, register, shifted_targets


class Block(nn.Module):
    def __init__(self, h: int):
        super().__init__()
        self.norm = nn.RMSNorm(h)
        self.up = nn.Linear(h, 2 * h, bias=False)
        self.down = nn.Linear(2 * h, h, bias=False)
        nn.init.zeros_(self.down.weight)  # starts as identity

    def forward(self, x):
        return x + self.down(nn.functional.silu(self.up(self.norm(x))))


@register("mtp")
class MTPHeads(Drafter):
    def __init__(self, cfg, hidden_size: int, n_feature_layers: int):
        super().__init__()
        self.k, self.ce_chunk = cfg.k, cfg.ce_chunk
        self.heads = nn.ModuleList(
            nn.ModuleDict(
                dict(
                    fuse=nn.Linear(2 * hidden_size, hidden_size, bias=False),
                    blocks=nn.Sequential(*[Block(hidden_size) for _ in range(cfg.n_layers)]),
                    norm=nn.RMSNorm(hidden_size),
                )
            )
            for _ in range(cfg.k)
        )

    def _head(self, j: int, last: torch.Tensor, nxt_emb: torch.Tensor) -> torch.Tensor:
        m = self.heads[j]
        return m["norm"](m["blocks"](m["fuse"](torch.cat([last, nxt_emb], dim=-1))))

    def forward(self, feats: Features) -> DraftOut:
        B, T = feats.input_ids.shape
        targets, valid = shifted_targets(feats.input_ids, feats.mask, self.k)
        # x[t+1] embedding at position t (the last position has no next token and is never valid)
        nxt = torch.cat([feats.input_ids[:, 1:], feats.input_ids[:, -1:]], dim=1)
        any_valid = valid.any(-1)  # [B, T]
        last, emb = feats.last[any_valid], feats.embed(nxt)[any_valid]  # [N, H]
        v, tg = valid[any_valid], targets[any_valid]  # [N, k]
        weight = feats.embed.weight
        loss, matches = last.new_zeros((), dtype=torch.float32), []
        for j in range(self.k):
            sel = v[:, j]
            h = self._head(j, last[sel], emb[sel])
            lj, mj = chunked_ce(h, weight, tg[sel, j], self.ce_chunk)
            loss = loss + lj
            full = torch.zeros(len(last), dtype=torch.bool, device=last.device)
            full[sel] = mj
            matches.append(full)
        n = v.sum().clamp(min=1)
        match = torch.stack(matches, dim=1)
        return DraftOut(loss / n, match, v)

    @torch.no_grad()
    def propose(self, hidden, last, next_tokens, embed, k):
        last, emb = last[:, -1], embed(next_tokens[:, -1])
        out = [(self._head(j, last, emb) @ embed.weight.t().to(last.dtype)).argmax(-1) for j in range(min(k, self.k))]
        return torch.stack(out, dim=1)

    @torch.no_grad()
    def propose_cached(self, state, hidden, last, next_tokens, embed, k):
        return self.propose(hidden[:, -1:], last[:, -1:], next_tokens[:, -1:], embed, k), None
