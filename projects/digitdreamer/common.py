"""Shared plumbing for the DigitDreamer training scripts: paths, data, run dirs, plots."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split
from torchvision.transforms import v2

from zoo import data

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
RUNS_ROOT = REPO_ROOT / "runs" / "digitdreamer"

# MNIST is 28x28; pad 2 px of background (0) on each side to 32x32, then uint8 [0, 255] -> float [-1, 1]
TO_PM1 = v2.Compose([v2.Pad(2, fill=0), v2.ToDtype(torch.float32, scale=True), v2.Normalize((0.5,), (0.5,))])


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
    """Per-image SSIM (Wang et al. 2004; Gaussian window, valid convolution) for images in [-1, 1]."""
    x, y = to_unit(x), to_unit(y)
    c1, c2 = 0.01**2, 0.03**2
    w = _gaussian_window(x.shape[1], size, sigma, x.device, x.dtype)
    blur = lambda t: F.conv2d(t, w, groups=t.shape[1])  # noqa: E731
    mx, my = blur(x), blur(y)
    vx, vy, cxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
    s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
    return s.flatten(1).mean(1)


def make_splits(dataset: str, val_size: int, seed: int) -> tuple[Dataset, Dataset, Dataset]:
    """(train, val, test), with val carved deterministically out of the train split."""
    ds_cls = getattr(data, dataset)
    full_train = ds_cls(root=str(DATA_ROOT), train=True, transform=TO_PM1)
    test = ds_cls(root=str(DATA_ROOT), train=False, transform=TO_PM1)
    train, val = random_split(
        full_train, [len(full_train) - val_size, val_size], generator=torch.Generator().manual_seed(seed)
    )
    return train, val, test


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True)
        return out + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def new_run_dir(stage: str, config: dict, device: torch.device, **extra) -> Path:
    run_dir = RUNS_ROOT / stage / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    meta = {"config": config, "git_commit": git_commit(), "device": str(device), **extra}
    (run_dir / "config.json").write_text(json.dumps(meta, indent=2))
    print(f"Run directory: {run_dir}")
    return run_dir


def latest_checkpoint(stage: str, name: str) -> Path:
    """Most recently written `<run>/<name>` under a stage (by mtime, so hand-renamed run dirs work)."""
    runs = sorted((RUNS_ROOT / stage).glob(f"*/{name}"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"No {name} found under {RUNS_ROOT / stage}; train that stage first.")
    return runs[-1]


def plot_curves(rows: list[dict], series: list[str], path: Path, ylabel: str = "loss", log: bool = False) -> None:
    epochs = [r["epoch"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    for key in series:
        ax.plot(epochs, [r[key] for r in rows], label=key)
    ax.set(xlabel="epoch", ylabel=ylabel)
    if log:
        ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
