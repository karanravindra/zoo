"""Logging images to W&B, and loading checkpoints back."""

from __future__ import annotations

import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

from zoo.metrics import to_unit
from zoo.training.callbacks import EMA


def log_images(trainer: L.Trainer, key: str, images: torch.Tensor, nrow: int = 8) -> None:
    """Log a batch of images in [-1, 1] as one grid to W&B (a no-op without a W&B logger, e.g. a dev run)."""
    loggers = [logger for logger in trainer.loggers if isinstance(logger, WandbLogger)]
    if not loggers:
        return
    # A PIL image, not the float grid: wandb.Image casts a float tensor straight to uint8 with no 0-255 scaling, which is black.
    grid = to_pil_image(make_grid(to_unit(images.detach().float().cpu()).clamp(0, 1), nrow=nrow, padding=1))
    for logger in loggers:
        logger.log_image(key, [grid])


def load_module(cls: type[L.LightningModule], path, *, ema: bool = True, map_location="cpu") -> L.LightningModule:
    """`cls.load_from_checkpoint(path)` in eval mode, with the EMA weights applied when the run had `--ema`.

    A checkpoint holds the live weights; the average is stored beside them (see `EMA`). Pass `ema=False` for the live ones."""
    module = cls.load_from_checkpoint(path, map_location=map_location)
    if ema:
        try:
            EMA.load_from_checkpoint(path).load_into(module)
        except KeyError:  # the run did not use EMA
            pass
    return module.eval()
