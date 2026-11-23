"""The runtime settings every training script shares."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class RunConfig:
    """Base for a project's `Config`: `class Config(RunConfig): ...` adds the project's own fields.

    Keep subclasses plain, serializable dataclasses; the whole config is logged to W&B and saved in checkpoints."""

    epochs: int = 10  # passes over the training set
    max_steps: int = -1  # stop after this many optimizer steps (-1 = use epochs)
    seed: int = 0  # Python, NumPy and torch RNGs
    deterministic: bool = False  # deterministic kernels (slower)
    precision: str = "32-true"  # Lightning precision: "32-true", "bf16-mixed", "16-mixed", ...
    grad_clip: float = 0.0  # max gradient norm (0 = off)
    compile: bool = False  # compile the model (see zoo.training.compile_)
    num_workers: int = 4  # DataLoader workers
    log_every: int = 50  # steps between logged train metrics
    wandb: Literal["online", "offline", "disabled"] = "online"  # W&B mode; offline runs upload later with `wandb sync`
    name: str | None = None  # W&B run name
    group: str | None = None  # W&B group, e.g. the stage of a multi-stage project or a sweep's name
    tags: tuple[str, ...] = ()  # W&B tags
    csv: bool = True  # also write metrics.csv locally, a backup if W&B is unreachable
    dev: bool = False  # throwaway run: no W&B, no checkpoints, outputs in a temp dir deleted on exit
