"""DigitNet settings: the project's fields on top of the shared `RunConfig` (epochs, seed, precision, W&B, ...)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from zoo.training import RunConfig


@dataclass
class Config(RunConfig):
    dataset: str = "MNIST"  # class name in zoo.data
    val_size: int = 5000  # held out from the train split for checkpoint selection
    width: int = 32  # channels of the first conv block (the second has twice as many)
    kernel_size: int = 5  # of both convs in each block
    dropout: float = 0.1  # before the linear head
    batch_size: int = 512
    lr: float = 4e-3  # peak of the schedule
    weight_decay: float = 5e-4  # AdamW
    lr_schedule: Literal["constant", "cosine", "linear", "onecycle"] = "onecycle"  # "constant" = no decay
    log_every: int = 20  # steps between logged train metrics
