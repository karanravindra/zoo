"""DigitNet: a small CNN for 28x28 digit classification, plus its config and checkpoint format."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
import torch.nn as nn


@dataclass
class Config:
    # data
    dataset: str = "MNIST"  # class name in zoo.data
    val_size: int = 5000  # held out from the train split for checkpoint selection
    # model
    width: int = 32
    kernel_size: int = 5
    dropout: float = 0.1
    # optimization
    batch_size: int = 512
    epochs: int = 10
    lr: float = 4e-3
    weight_decay: float = 5e-4
    # runtime
    seed: int = 0
    deterministic: bool = True
    compile: bool = False  # torch.compile the model with mode="reduce-overhead" (CUDA graphs on GPU)
    dev: bool = False  # dev run: outputs go to a temporary directory deleted on exit, so nothing is logged to runs/
    num_workers: int = 4
    log_every: int = 20  # steps between progress-bar loss updates (each forces a device sync)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Config:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


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

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        width: int = 32,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(in_channels, width, kernel_size),
            conv_block(width, width * 2, kernel_size),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def build_model(cfg: Config) -> DigitNet:
    from zoo import data

    ds = getattr(data, cfg.dataset)
    return DigitNet(
        in_channels=ds.image_shape[0],
        num_classes=ds.num_classes,
        width=cfg.width,
        kernel_size=cfg.kernel_size,
        dropout=cfg.dropout,
    )


def save_checkpoint(path: Path, model: nn.Module, cfg: Config, epoch: int, metrics: dict) -> None:
    """Write a self-describing checkpoint: weights + the config that built them."""
    tmp = path.with_suffix(".tmp")
    torch.save({"config": cfg.to_dict(), "epoch": epoch, "metrics": metrics, "model": model.state_dict()}, tmp)
    tmp.replace(path)


def load_checkpoint(path: str | Path, map_location="cpu") -> tuple[DigitNet, Config, dict]:
    """Rebuild the model from a checkpoint's own config. Returns (model, config, raw checkpoint)."""
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    cfg = Config.from_dict(ckpt["config"])
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    return model, cfg, ckpt
