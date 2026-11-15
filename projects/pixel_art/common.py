"""Shared bits for the pixel_art training scripts: the RGBA image grid."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from zoo.metrics import to_unit


def save_grid(x: torch.Tensor, path: Path, nrow: int, scale: int = 8) -> None:
    """Save RGBA images in [-1, 1] as a grid, composited over a checkerboard so transparency is visible
    and upscaled `scale`x with nearest neighbour."""
    x = to_unit(x.float().cpu())
    rgb, a = x[:, :3], x[:, 3:]
    h, w = x.shape[-2:]
    checker = ((torch.arange(h)[:, None] + torch.arange(w)[None, :]) % 2).float() * 0.2 + 0.6
    img = rgb * a + checker * (1 - a)
    img = F.interpolate(img, scale_factor=scale, mode="nearest")
    save_image(img.clamp(0, 1), path, nrow=nrow, padding=2, pad_value=0.2)
