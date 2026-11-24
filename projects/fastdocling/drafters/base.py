"""The drafter interface and registry. A new method is one file in this folder with `@register("name")`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class Features:
    """What the frozen target hands a drafter for one batch. All tensors are over the full sequence [B, T, ...].

    Position t holds the target's state after reading token t. The drafter at t knows the next token x[t+1] (the target's own
    greedy choice, which verification yields for free) and proposes x[t+2 .. t+1+k]."""

    hidden: torch.Tensor  # [B, T, F*H] concatenated hidden states of cfg.feature_layers
    last: torch.Tensor  # [B, T, H] final hidden state (after the target's last norm)
    input_ids: torch.Tensor  # [B, T]
    mask: torch.Tensor  # [B, T] bool, True where the token belongs to the target's response (DocTags)
    embed: nn.Embedding  # the target's (frozen, tied) token embedding; also its LM head
    # Drafters that need positions can derive them from input_ids; pages are left-aligned and right-padded.


@dataclass
class DraftOut:
    loss: torch.Tensor  # scalar
    match: torch.Tensor  # [N, k] bool, draft offset j matched the target's token (False where not valid)
    valid: torch.Tensor  # [N, k] bool, offset j has a real target token to be scored against
    logs: dict | None = None  # extra scalars to log (loss terms, confidence stats), logged as <prefix>/<name>


class Drafter(nn.Module):
    """Interface. `forward` trains, `propose` drafts at inference."""

    k: int

    def forward(self, feats: Features) -> DraftOut:
        raise NotImplementedError

    @torch.no_grad()
    def propose(self, hidden: torch.Tensor, last: torch.Tensor, next_tokens: torch.Tensor, embed: nn.Embedding, k: int) -> torch.Tensor:
        """Greedy chain draft from the target's states over the whole prefix: [B, k] token ids that follow x[T].
        `hidden` [B, T, F*H] and `last` [B, T, H] are the target's states at every prefix position, `next_tokens` [B, T] holds
        x[t+1] for each t (so its last entry is the token just verified, the free one). Drafters that only look at the final
        position (MTP) ignore the rest; ones that attend over the prefix (EAGLE) use it. No cache yet: simple and correct,
        and the MLX port will cache."""
        raise NotImplementedError

    @torch.no_grad()
    def propose_cached(self, state, hidden: torch.Tensor, last: torch.Tensor, next_tokens: torch.Tensor, embed: nn.Embedding, k: int):
        """Incremental `propose` for decoding: `hidden`, `last` and `next_tokens` cover only the positions added since the previous
        call (the whole prompt on the first, with state=None). Returns (drafts [B, k], state). This default keeps the history and
        recomputes; drafters override it with a real cache."""
        if state is not None:
            hidden, last, next_tokens = (torch.cat([s, n], 1) for s, n in zip(state, (hidden, last, next_tokens)))
        return self.propose(hidden, last, next_tokens, embed, k), (hidden, last, next_tokens)


REGISTRY: dict[str, Callable[..., Drafter]] = {}


def register(name: str):
    def deco(cls):
        REGISTRY[name] = cls
        cls.registry_name = name
        return cls

    return deco


def make_drafter(cfg, hidden_size: int, n_feature_layers: int) -> Drafter:
    import importlib
    import pkgutil

    import drafters

    for m in pkgutil.iter_modules(drafters.__path__):  # importing a module registers its drafter
        if m.name != "base":
            importlib.import_module(f"drafters.{m.name}")

    if cfg.drafter not in REGISTRY:
        raise SystemExit(f"unknown drafter {cfg.drafter!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[cfg.drafter](cfg, hidden_size, n_feature_layers)


def shifted_targets(input_ids: torch.Tensor, mask: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """For every position t and offset j: the token x[t+2+j] and whether x[t+1..t+2+j] are all in the response.
    Returns (targets [B, T, k] long, valid [B, T, k] bool)."""
    B, T = input_ids.shape
    tg = input_ids.new_zeros(B, T, k)
    valid = torch.zeros(B, T, k, dtype=torch.bool, device=input_ids.device)
    for j in range(k):
        s = 2 + j
        if s >= T:
            break
        tg[:, : T - s, j] = input_ids[:, s:]
        # x[t+1] (the free token) and x[t+2+j] must be in the response; t itself may be the last prompt position,
        # which is where drafting starts at inference. The response is contiguous, so these two checks cover the span.
        valid[:, : T - s, j] = mask[:, s:] & mask[:, 1 : T - s + 1]
    return tg, valid


def chunked_ce(h: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor, chunk: int,
               weights: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-entropy of h @ weight.T against targets without materializing [N, vocab]: chunks are recomputed in backward.
    h [N, H], weight [V, H], targets [N], optional per-row weights [N]. Returns (summed loss, argmax-matches [N] bool)."""

    def one(hc, tc, wc):
        logits = (hc @ weight.t().to(hc.dtype)).float()
        ce = F.cross_entropy(logits, tc, reduction="none")
        return (ce if wc is None else ce * wc).sum(), logits.argmax(-1) == tc

    total, matches = h.new_zeros((), dtype=torch.float32), []
    for i in range(0, len(h), chunk):
        w = None if weights is None else weights[i : i + chunk]
        loss, m = checkpoint(one, h[i : i + chunk], targets[i : i + chunk], w, use_reentrant=False)
        total = total + loss
        matches.append(m)
    return total, torch.cat(matches) if matches else targets.new_zeros(0, dtype=torch.bool)


HEAD_DIM = 64
ROPE_THETA = 100000.0  # the target's


def rope(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """x [B, heads, N, hd], pos [N] or [B, N] absolute positions: rotate halves, the Llama layout."""
    hd = x.shape[-1]
    inv = 1.0 / (ROPE_THETA ** (torch.arange(0, hd, 2, device=x.device, dtype=torch.float32) / hd))
    pos = pos.float()
    if pos.dim() == 1:
        pos = pos[None]
    ang = pos[:, None, :, None] * inv  # [B or 1, 1, N, hd/2]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., : hd // 2], x[..., hd // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def chain_from_match(match: torch.Tensor) -> torch.Tensor:
    """[N, k] per-offset matches -> True while every earlier offset matched too (cumulative AND)."""
    return match.long().cumprod(dim=1).bool()
