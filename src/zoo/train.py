"""Small training utilities (no loops): EMA update, steps/s timing, gradient and weight statistics."""

from __future__ import annotations

import time
from collections.abc import Callable

import torch


@torch.no_grad()
def update_ema(ema, model, decay: float) -> None:
    """EMA of the parameters (`ema = decay * ema + (1 - decay) * model`); buffers are copied. Fused over all tensors
    (`torch._foreach_*`), a handful of kernel launches per step instead of one per tensor."""
    torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1 - decay)
    if buffers := list(ema.buffers()):  # the foreach ops reject an empty list, and a model need not have buffers
        torch._foreach_copy_(buffers, list(model.buffers()))


class Throughput:
    """Units/s over a rolling window (`tick`) and over the whole epoch (`rate`), timed from construction/`restart`.

    A unit is a step by default; pass `per_step` (e.g. tokens per step) to count something else. `pause`/`resume`
    exclude a stretch (eval, sampling, checkpointing) from every reading. Call `tick` only after a device sync
    (e.g. a `.item()`), so the timing is real."""

    def __init__(self) -> None:
        self.restart()

    def restart(self) -> None:
        self.t0 = self.t_log = time.perf_counter()
        self.n_log = 0
        self._paused_at: float | None = None

    def pause(self) -> None:
        self._paused_at = time.perf_counter()

    def resume(self) -> None:
        if self._paused_at is not None:
            gap = time.perf_counter() - self._paused_at
            self.t0 += gap  # shift both clocks forward so the gap counts for nothing
            self.t_log += gap
            self._paused_at = None

    def tick(self, step: int, per_step: float = 1) -> float:
        """Units/s since the previous `tick` (or `restart`); `step` is the count of steps done so far."""
        now = time.perf_counter()
        rate = (step - self.n_log) * per_step / (now - self.t_log)
        self.t_log, self.n_log = now, step
        return rate

    def rate(self, steps: int, per_step: float = 1) -> float:
        return steps * per_step / (time.perf_counter() - self.t0)

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0


STAT_FIELDS = ["grad_norm", "grad_norm_max", "clip_frac", "grad_cos", "weight_norm", "update_ratio"]


class ParamStats:
    """Gradient and weight statistics for training logs, kept on the device until `read()`.

    Call `grads(step, ...)` after `backward()` (and after clipping, if any) and before `optimizer.step()`, then
    `stepped(step)` after `optimizer.step()`. `read()` returns `(overall, per_group)` for the steps since the
    previous `read()` and starts a new window; it is the only call that syncs the device.

    Every step:
      grad_norm      mean global gradient L2 norm, before clipping
      grad_norm_max  its largest value in the window
      clip_frac      fraction of steps whose norm exceeded `clip` (left out when not clipping)
    Every `every` steps (0 turns these off), overall and for each parameter group, averaged over the window:
      weight_norm    L2 norm of the weights
      update_ratio   ||w_after - w_before|| / ||w_before|| for that optimizer step
      grad_cos       cosine between this step's gradient and the previous step's: near 1 means a steady
                     direction, near 0 noise-dominated, negative oscillating
    and per group only: grad_norm (before clipping). `per_group` keys are `<group>.<stat>`, and it is None when
    no detailed step fell in the window. `group(name)` maps a parameter name to its group label (default: the
    top-level module name). The detailed steps clone the gradients (one step earlier) and the weights, so they
    cost two parameter-sized copies."""

    def __init__(self, model: torch.nn.Module, every: int = 100, group: Callable[[str], str] | None = None) -> None:
        group = group or (lambda name: name.split(".")[0])
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.params = [p for _, p in named]
        self.every = every
        self.groups = list(dict.fromkeys(group(n) for n, _ in named))
        dev = self.params[0].device
        self._gid = torch.tensor([self.groups.index(group(n)) for n, _ in named], device=dev)
        self._zero = lambda *shape: torch.zeros(shape, device=dev)
        self._prev: list[torch.Tensor] | None = None  # the previous step's gradients
        self._prev_step = -1
        self._before: list[torch.Tensor] | None = None  # weights before this optimizer step
        self._g2 = self._dot = self._p2 = None
        self._has_cos = False
        self._new_window()

    def _new_window(self) -> None:
        self._n = 0
        self._norm_sum, self._norm_max, self._clipped = self._zero(), self._zero(), self._zero()
        self._clip = 0.0
        self._detail = self._zero(4, len(self.groups) + 1)  # rows: weight_norm, update_ratio, grad_norm, grad_cos
        self._n_detail, self._n_cos = 0, 0

    def _due(self, step: int) -> bool:
        return self.every > 0 and step % self.every == 0

    def _by_group(self, per_param: torch.Tensor) -> torch.Tensor:
        """Per-parameter sums -> per-group sums, with the overall sum appended."""
        out = self._zero(len(self.groups)).index_add_(0, self._gid, per_param)
        return torch.cat([out, per_param.sum()[None]])

    @torch.no_grad()
    def grads(self, step: int, total_norm: torch.Tensor | None = None, clip: float = 0.0) -> None:
        """`total_norm` is the pre-clip global norm if you already have it (`clip_grad_norm_` returns it);
        `clip` is the max norm the gradients were clipped to (0 = not clipped)."""
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in self.params]
        if total_norm is None:
            total_norm = torch.nn.utils.get_total_norm(grads)
        total_norm = total_norm.detach().float()
        self._n += 1
        self._norm_sum += total_norm
        self._norm_max = torch.maximum(self._norm_max, total_norm)
        if clip > 0:
            self._clip = clip
            self._clipped += total_norm > clip
        if self._due(step):
            # undo the clipping scale, so per-group gradient norms are the ones backward() produced
            scale = (clip / (total_norm + 1e-6)).clamp(max=1.0) if clip > 0 else 1.0
            self._g2 = self._by_group(torch.stack([g.float().pow(2).sum() for g in grads])) / scale**2
            self._has_cos = self._prev is not None and self._prev_step == step - 1
            if self._has_cos:
                self._dot = self._by_group(torch.stack([(g.float() * q).sum() for g, q in zip(grads, self._prev)])) / scale
            self._before = [p.detach().clone() for p in self.params]
        if self._due(step + 1):  # keep this step's gradients for the cosine at the next (detailed) step
            scale = (clip / (total_norm + 1e-6)).clamp(max=1.0) if clip > 0 else 1.0
            self._prev = [g.float() / scale for g in grads]
            self._p2 = self._by_group(torch.stack([q.pow(2).sum() for q in self._prev]))
            self._prev_step = step
        elif not self._due(step):
            self._prev = None

    @torch.no_grad()
    def stepped(self, step: int) -> None:
        if not self._due(step) or self._before is None:
            return
        w2 = self._by_group(torch.stack([b.float().pow(2).sum() for b in self._before]))
        d2 = self._by_group(torch.stack([(p.detach() - b).float().pow(2).sum() for p, b in zip(self.params, self._before)]))
        self._detail[0] += w2.sqrt()
        self._detail[1] += (d2 / w2.clamp_min(1e-30)).sqrt()
        self._detail[2] += self._g2.sqrt()
        self._n_detail += 1
        if self._has_cos:
            self._detail[3] += self._dot / (self._g2 * self._p2).sqrt().clamp_min(1e-30)
            self._n_cos += 1
        self._before = None

    def read(self) -> tuple[dict[str, float], dict[str, float] | None]:
        overall: dict[str, float] = {}
        groups = None
        if self._n:
            overall["grad_norm"] = self._norm_sum.item() / self._n
            overall["grad_norm_max"] = self._norm_max.item()
            if self._clip > 0:
                overall["clip_frac"] = self._clipped.item() / self._n
        if self._n_detail:
            d = (self._detail[:3] / self._n_detail).tolist()
            cos = (self._detail[3] / max(1, self._n_cos)).tolist()
            overall["weight_norm"], overall["update_ratio"] = d[0][-1], d[1][-1]
            groups = {}
            for i, g in enumerate(self.groups):
                groups[f"{g}.weight_norm"], groups[f"{g}.update_ratio"], groups[f"{g}.grad_norm"] = d[0][i], d[1][i], d[2][i]
            if self._n_cos:
                overall["grad_cos"] = cos[-1]
                groups |= {f"{g}.grad_cos": cos[i] for i, g in enumerate(self.groups)}
        self._new_window()
        return overall, groups

    def group_fields(self) -> list[str]:
        """Column names of `per_group`, for a csv header."""
        return [f"{g}.{s}" for g in self.groups for s in ("weight_norm", "update_ratio", "grad_norm", "grad_cos")]
