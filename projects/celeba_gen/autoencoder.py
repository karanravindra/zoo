"""Config, convolutional autoencoder and checkpoints for CelebA-HQ at 256x256.

The encoder halves the resolution three times (256 -> 128 -> 64 -> 32), so the latent is `latent_channels` x 32 x 32:
8x per side, 48x fewer values than the RGB image at the default 4 channels. Nothing in the network is tied to 256 px
except the number of stride-2 stages, so the same code also runs at 128 or 512 with a different latent size.

Images live in [-1, 1]. The model returns the reconstruction in the same range; the training loop clamps it for the
PSNR and SSIM metrics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
import torch.nn as nn

IMAGE_CHANNELS = 3
DOWNSAMPLE = 8  # three stride-2 stages in the encoder, mirrored by three upsamples in the decoder


class _Serializable:
    def to_dict(self) -> dict:
        return asdict(self)  # type: ignore[call-overload]

    @classmethod
    def from_dict(cls, d: dict):
        known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class AEConfig(_Serializable):
    # data (CelebAHQ256; `val_size` images are held out of the train split, the official val split is the test set)
    val_size: int = 1000
    # model
    width: int = 32  # base channel count, must be a multiple of 8 (GroupNorm uses 8 groups)
    kernel_size: int = 3  # for every spatial conv; full-resolution layers dominate the cost, so 3 rather than 5
    latent_channels: int = 4  # latent is latent_channels x 32 x 32
    # optimization
    batch_size: int = 32
    epochs: int = 20
    lr: float = 2e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0  # clip the global gradient norm; 0 disables
    hflip_prob: float = 0.5  # mirror each training image left-right with this probability (val and test are never flipped)
    # runtime
    seed: int = 0
    deterministic: bool = True
    compile: bool = False  # torch.compile the model with mode="reduce-overhead" (CUDA graphs on GPU)
    dev: bool = False  # dev run: outputs go to a temporary directory deleted on exit, so nothing is logged to runs/
    num_workers: int = 4
    log_every: int = 100


def conv_gn(c_in: int, c_out: int, k: int = 3, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, k, stride=stride, padding=k // 2, bias=False),
        nn.GroupNorm(8, c_out),
        nn.SiLU(),
    )


def up_gn(c_in: int, c_out: int, k: int = 3) -> nn.Sequential:
    return nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), conv_gn(c_in, c_out, k))


class Autoencoder(nn.Module):
    """256 -> 128 -> 64 -> 32 conv encoder to a `latent_channels` x 32 x 32 code, mirrored decoder."""

    def __init__(self, in_channels: int = IMAGE_CHANNELS, width: int = 32, latent_channels: int = 4, kernel_size: int = 3):
        super().__init__()
        w, k = width, kernel_size
        self.encoder = nn.Sequential(
            conv_gn(in_channels, w, k),
            conv_gn(w, w, k),  # 256
            conv_gn(w, 2 * w, k, stride=2),  # 128
            conv_gn(2 * w, 2 * w, k),
            conv_gn(2 * w, 4 * w, k, stride=2),  # 64
            conv_gn(4 * w, 4 * w, k),
            conv_gn(4 * w, 4 * w, k, stride=2),  # 32
            conv_gn(4 * w, 4 * w, k),
            nn.Conv2d(4 * w, latent_channels, 1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 4 * w, k, padding=k // 2),
            conv_gn(4 * w, 4 * w, k),
            up_gn(4 * w, 4 * w, k),  # 64
            conv_gn(4 * w, 4 * w, k),
            up_gn(4 * w, 2 * w, k),  # 128
            conv_gn(2 * w, 2 * w, k),
            up_gn(2 * w, w, k),  # 256
            conv_gn(w, w, k),
            nn.Conv2d(w, in_channels, k, padding=k // 2),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def build_ae(cfg: AEConfig) -> Autoencoder:
    return Autoencoder(in_channels=IMAGE_CHANNELS, width=cfg.width, latent_channels=cfg.latent_channels, kernel_size=cfg.kernel_size)


def _atomic_save(obj: dict, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def save_ae(path: Path, model: Autoencoder, cfg: AEConfig, epoch: int, metrics: dict) -> None:
    _atomic_save({"config": cfg.to_dict(), "epoch": epoch, "metrics": metrics, "model": model.state_dict()}, path)


def load_ae(path: str | Path, map_location="cpu") -> tuple[Autoencoder, AEConfig, dict]:
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    cfg = AEConfig.from_dict(ckpt["config"])
    model = build_ae(cfg)
    model.load_state_dict(ckpt["model"])
    return model, cfg, ckpt
