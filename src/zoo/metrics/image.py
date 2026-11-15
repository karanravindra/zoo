"""Image quality metrics for images in [-1, 1]."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def to_unit(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1] for saving images."""
    return (x + 1) / 2


def psnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-image PSNR in dB for images in [-1, 1] (peak signal range 1 after mapping to [0, 1])."""
    mse = ((to_unit(x) - to_unit(y)) ** 2).flatten(1).mean(1)
    return 10 * torch.log10(1 / mse.clamp_min(1e-10))


def _gaussian_window(channels: int, size: int, sigma: float, device, dtype) -> torch.Tensor:
    g = torch.exp(-((torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2) ** 2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).expand(channels, 1, size, size).contiguous()


def ssim(x: torch.Tensor, y: torch.Tensor, size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Per-image SSIM (Wang et al. 2004; Gaussian window, valid convolution) for images in [-1, 1].

    `size` is the window; use a smaller one (e.g. 7) for tiny images. Each channel is scored separately."""
    x, y = to_unit(x), to_unit(y)
    c1, c2 = 0.01**2, 0.03**2
    w = _gaussian_window(x.shape[1], size, sigma, x.device, x.dtype)
    blur = lambda t: F.conv2d(t, w, groups=t.shape[1])  # noqa: E731
    mx, my = blur(x), blur(y)
    vx, vy, cxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
    s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
    return s.flatten(1).mean(1)
