"""DigitDreamer settings: one config per stage, each on top of the shared `RunConfig`."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from tyro.conf import Positional

from zoo.paths import runs_root
from zoo.training import RunConfig

LRSchedule = Literal["constant", "cosine", "linear", "onecycle"]


@dataclass
class AEConfig(RunConfig):
    """Stage 1: train the autoencoder. Keeps the best checkpoint by val/mse (pass it to `flow` as --ae-ckpt)."""

    dataset: str = "MNIST"  # class name in zoo.data
    val_size: int = 5000  # held out from the train split for checkpoint selection
    width: int = 8
    kernel_size: int = 5  # of every spatial conv
    latent_channels: int = 4  # the latent is latent_channels x 4 x 4
    mask_prob: float = 0.5  # chance per sample of zeroing the latent's detail half, so the other half carries the content
    batch_size: int = 64
    lr: float = 2e-3  # peak of the schedule
    weight_decay: float = 1e-4  # AdamW
    deterministic: bool = True
    lr_schedule: LRSchedule = "onecycle"  # "constant" = no decay
    log_every: int = 20  # steps between logged train metrics


@dataclass(kw_only=True)
class FlowConfig(RunConfig):
    """Stage 2: train the class-conditioned rectified flow on the frozen AE's latents (run `ae` first)."""

    ae_ckpt: str  # the autoencoder's checkpoint (`<run dir>/checkpoints/<best>.ckpt`), required
    width: int = 64
    depth: int = 6
    emb_dim: int = 128
    dropout: float = 0.0
    batch_size: int = 256
    lr: float = 1e-3  # peak of the schedule
    weight_decay: float = 0.0  # AdamW
    class_dropout: float = 0.1  # chance of dropping the label to "null", which enables classifier-free guidance
    sample_steps: int = 50  # Euler steps for the sample grids logged each epoch
    cfg_scale: float = 2.0  # guidance scale for those grids
    epochs: int = 20
    ema: float = 0.999  # decay of the weight average; validation, samples and the sampler use it
    deterministic: bool = True
    lr_schedule: LRSchedule = "onecycle"  # "constant" = no decay
    log_every: int = 100  # steps between logged train metrics


@dataclass
class SampleConfig:
    """Generate a grid of digits (rows = digits) from a flow checkpoint, with the EMA weights if the run had them."""

    ckpt: Positional[Path]  # flow checkpoint
    digits: list[int] = field(default_factory=lambda: list(range(10)))
    per_digit: int = 10  # samples per digit
    steps: int | None = None  # Euler steps (default: the run's sample_steps)
    cfg: float | None = None  # classifier-free guidance scale, 1 = off (default: the run's cfg_scale)
    seed: int = 0
    out: Path = runs_root("digitdreamer") / "sample.png"


@dataclass
class PrepareConfig:
    """Download and cache the dataset, so the training commands start straight away."""

    dataset: str = "MNIST"  # class name in zoo.data
