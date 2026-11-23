"""Learning-rate schedules selected by `RunConfig.lr_schedule`."""

from __future__ import annotations

import math

import torch

from zoo.training.config import RunConfig


def make_scheduler(optimizer: torch.optim.Optimizer, cfg: RunConfig, total_steps: int) -> dict:
    """The `lr_scheduler` entry for Lightning's `configure_optimizers`, stepped every optimizer step.

    `cfg.lr_schedule` picks the shape; the optimizer's `lr` is the peak.
      constant   warmup, then the peak for the rest of training (no decay)
      cosine     warmup, then cosine decay to `min_lr_ratio` x peak at the last step
      linear     warmup, then linear decay to `min_lr_ratio` x peak at the last step
      onecycle   `OneCycleLR`: its own warmup and anneal, so `warmup` and `min_lr_ratio` are ignored
    `total_steps` is usually `int(self.trainer.estimated_stepping_batches)`."""
    if cfg.lr_schedule == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[g["lr"] for g in optimizer.param_groups], total_steps=total_steps)
        return {"scheduler": sched, "interval": "step"}

    warmup, floor = cfg.warmup, cfg.min_lr_ratio

    def scale(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        if cfg.lr_schedule == "constant":
            return 1.0
        t = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        decay = 0.5 * (1 + math.cos(math.pi * t)) if cfg.lr_schedule == "cosine" else 1 - t
        return floor + (1 - floor) * decay

    return {"scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, scale), "interval": "step"}
