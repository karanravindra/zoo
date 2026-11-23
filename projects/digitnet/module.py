"""DigitNet: a small CNN for 28x28 digit classification, and its LightningModule."""

from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config

from zoo import data
from zoo.training import compile_


def conv_block(c_in: int, c_out: int, k: int) -> nn.Sequential:
    """Two kxk convs (BN + GELU) followed by 2x2 max-pool, halving spatial size."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, k, padding=k // 2, bias=False),
        nn.BatchNorm2d(c_out),
        nn.GELU(),
        nn.Conv2d(c_out, c_out, k, padding=k // 2, bias=False),
        nn.BatchNorm2d(c_out),
        nn.GELU(),
        nn.MaxPool2d(2),
    )


class DigitNet(nn.Module):
    """28 -> 14 -> 7 -> global pool -> logits."""

    def __init__(self, in_channels: int = 1, num_classes: int = 10, width: int = 32, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        self.features = nn.Sequential(conv_block(in_channels, width, kernel_size), conv_block(width, width * 2, kernel_size))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(width * 2, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def build_model(cfg: Config) -> DigitNet:
    ds = getattr(data, cfg.dataset)
    return DigitNet(ds.image_shape[0], ds.num_classes, cfg.width, cfg.kernel_size, cfg.dropout)


class LitDigitNet(L.LightningModule):
    """Training wrapper. Built from the config as a plain dict, so `load_from_checkpoint(path)` rebuilds it."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = Config(**cfg)
        self.net = build_model(self.cfg)

    def setup(self, stage: str) -> None:
        if self.cfg.compile and stage == "fit":
            compile_(self.net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _step(self, batch, prefix: str) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        on_step = prefix == "train"
        self.log(f"{prefix}/loss", loss, on_step=on_step, on_epoch=True, batch_size=len(y))
        self.log(f"{prefix}/acc", acc, on_step=on_step, on_epoch=True, batch_size=len(y))
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.cfg.lr, total_steps=int(self.trainer.estimated_stepping_batches))
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
