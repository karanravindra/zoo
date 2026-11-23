"""Lightning callbacks: EMA of the weights, steps/s, gradient and weight statistics."""

from __future__ import annotations

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback

from zoo.train import ParamStats, Throughput


class EMA(Callback):
    """Exponential moving average of the weights (and a copy of the buffers).

    The average is held as plain tensors rather than a deepcopy of the module, so compiling the model stays
    unaffected. It is swapped in for validation/test, and swapped out again for training. The shadow weights are saved in
    checkpoints under `callbacks["EMA"]`, and `load_into(module)` applies them (for sampling or export)."""

    def __init__(self, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: list[torch.Tensor] | None = None
        self._backup: list[torch.Tensor] | None = None

    @staticmethod
    def _tensors(module: torch.nn.Module) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        return list(module.parameters()), list(module.buffers())

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self.shadow is None:  # a resumed run restores it in on_load_checkpoint
            params, buffers = self._tensors(pl_module)
            self.shadow = [t.detach().clone() for t in params + buffers]

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        params, buffers = self._tensors(pl_module)
        n = len(params)
        torch._foreach_lerp_(self.shadow[:n], [p.detach() for p in params], 1 - self.decay)
        if buffers:
            torch._foreach_copy_(self.shadow[n:], [b.detach() for b in buffers])

    def _swap_in(self, pl_module: L.LightningModule) -> None:
        params, buffers = self._tensors(pl_module)
        live = params + buffers
        self._backup = [t.detach().clone() for t in live]
        with torch.no_grad():
            torch._foreach_copy_(live, self.shadow)

    def _swap_out(self, pl_module: L.LightningModule) -> None:
        params, buffers = self._tensors(pl_module)
        with torch.no_grad():
            torch._foreach_copy_(params + buffers, self._backup)
        self._backup = None

    def on_validation_start(self, trainer, pl_module) -> None:
        if not trainer.sanity_checking and self.shadow is not None:
            self._swap_in(pl_module)

    def on_validation_end(self, trainer, pl_module) -> None:
        if self._backup is not None:
            self._swap_out(pl_module)

    def on_test_start(self, trainer, pl_module) -> None:
        if self.shadow is not None:
            self._swap_in(pl_module)

    def on_test_end(self, trainer, pl_module) -> None:
        if self._backup is not None:
            self._swap_out(pl_module)

    def state_dict(self) -> dict:
        return {"shadow": self.shadow}

    def load_state_dict(self, state_dict: dict) -> None:
        self.shadow = state_dict["shadow"]

    def load_into(self, module: torch.nn.Module) -> None:
        """Copy the averaged weights into `module` (same architecture as the trained one)."""
        params, buffers = self._tensors(module)
        with torch.no_grad():
            torch._foreach_copy_(params + buffers, [t.to(params[0].device) for t in self.shadow])


class ThroughputMonitor(Callback):
    """Logs `steps_per_s` every `every` steps, leaving validation time out of the reading."""

    def __init__(self, every: int = 50) -> None:
        self.every = every
        self.tp = Throughput()

    def on_train_start(self, trainer, pl_module) -> None:
        self.tp.restart()

    def on_validation_start(self, trainer, pl_module) -> None:
        self.tp.pause()

    def on_validation_end(self, trainer, pl_module) -> None:
        self.tp.resume()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if trainer.global_step % self.every == 0:
            # Lightning has synced the device by the time it logs, and `self.log` below forces it for this step.
            pl_module.log("perf/steps_per_s", self.tp.tick(trainer.global_step), on_step=True, on_epoch=False)


class ParamStatsMonitor(Callback):
    """Logs `zoo.train.ParamStats` (gradient norm, weight norm, update ratio, gradient direction) under `stats/`.

    Gradient norms are measured before Lightning's clipping, which runs after `on_before_optimizer_step`."""

    def __init__(self, every: int = 50, detail_every: int = 100) -> None:
        self.every, self.detail_every = every, detail_every
        self.stats: ParamStats | None = None
        self._step = 0

    def on_train_start(self, trainer, pl_module) -> None:
        self.stats = ParamStats(pl_module, every=self.detail_every)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
        self._step = trainer.global_step + 1  # global_step counts finished optimizer steps
        self.stats.grads(self._step)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self.stats.stepped(self._step)
        if self._step % self.every == 0:
            overall, _ = self.stats.read()
            pl_module.log_dict({f"stats/{k}": v for k, v in overall.items()}, on_step=True, on_epoch=False)
