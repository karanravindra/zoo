"""Helpers for picking and working with PyTorch devices."""

from __future__ import annotations

import torch


def get_device(prefer: str | None = None) -> torch.device:
    """Return the best available device (cuda > mps > cpu).

    Args:
        prefer: Optional device string (e.g. "cuda", "mps", "cpu", "cuda:1").
            If given and available, it is used; otherwise falls back to auto-detection.
    """
    if prefer is not None:
        dev = torch.device(prefer)
        if is_available(dev.type):
            return dev

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def is_available(device_type: str) -> bool:
    """Check whether a device type ("cuda", "mps", "cpu") is available."""
    if device_type == "cuda":
        return torch.cuda.is_available()
    if device_type == "mps":
        return torch.backends.mps.is_available()
    return device_type == "cpu"


def seed_everything(seed: int = 0, deterministic: bool = False) -> None:
    """Seed Python, NumPy and torch RNGs on all available devices.

    With `deterministic=True`, also disable cuDNN autotuning and ask torch for
    deterministic kernels (warning, not erroring, where none exists). This can be slower.
    """
    import os
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Required by cuBLAS for deterministic matmuls; must be set before CUDA initializes.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def synchronize(device: torch.device | str | None = None) -> None:
    """Block until all queued work on the device finishes (useful for timing)."""
    device = torch.device(device) if device is not None else get_device()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def empty_cache(device: torch.device | str | None = None) -> None:
    """Release cached memory on the device, if applicable."""
    device = torch.device(device) if device is not None else get_device()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
