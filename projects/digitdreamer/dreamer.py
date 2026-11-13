"""DigitDreamer: a conv autoencoder plus a class-conditioned rectified flow in its latent space.

Stage 1 (`train_ae.py`): an autoencoder squeezes 1x32x32 digits (28x28 MNIST padded by 2) into a `latent_channels`x4x4 latent.
Stage 2 (`train_flow.py`): a rectified flow learns to transport Gaussian noise to
(normalized) latents along straight lines, conditioned on the digit class. Sampling
integrates the learned velocity with Euler steps, then decodes with the AE decoder.

Conventions: images live in [-1, 1]; flow time runs from t=0 (noise) to t=1 (data).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Serializable:
    def to_dict(self) -> dict:
        return asdict(self)  # type: ignore[call-overload]

    @classmethod
    def from_dict(cls, d: dict):
        known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class AEConfig(_Serializable):
    # data
    dataset: str = "MNIST"  # class name in zoo.data
    val_size: int = 5000
    # model
    width: int = 8
    kernel_size: int = 5  # for every spatial conv in the autoencoder
    latent_channels: int = 4  # latent is latent_channels x LATENT_SIZE x LATENT_SIZE
    # Per training sample, with this probability the second (detail) half of the latent channels is zeroed
    # before decoding, so the first half must carry global information and the second half only refines it.
    mask_prob: float = 0.5
    # optimization
    batch_size: int = 64
    epochs: int = 10
    lr: float = 2e-3
    weight_decay: float = 1e-4
    # runtime
    seed: int = 0
    deterministic: bool = True
    num_workers: int = 4
    log_every: int = 20


@dataclass
class FlowConfig(_Serializable):
    ae_ckpt: str | None = "runs/digitdreamer/ae/latent1/best.pt"  # relative to the repo root; None -> best.pt of the latest AE run
    # model
    width: int = 64
    depth: int = 6
    emb_dim: int = 128
    dropout: float = 0.0
    # optimization
    batch_size: int = 256
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 0.0
    ema_decay: float = 0.999
    class_dropout: float = 0.1  # prob. of replacing the label with "null" (enables CFG)
    # sampling (for per-epoch previews)
    sample_steps: int = 50
    cfg_scale: float = 2.0
    # runtime
    seed: int = 0
    deterministic: bool = True
    num_workers: int = 4  # only used to encode the dataset once
    log_every: int = 100


# --- autoencoder ---------------------------------------------------------------

IMAGE_SIZE = 32  # images are padded to this (see common.TO_PM1)
LATENT_SIZE = 4  # IMAGE_SIZE / 8: three stride-2 stages


def conv_gn(c_in: int, c_out: int, k: int = 3, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, k, stride=stride, padding=k // 2, bias=False),
        nn.GroupNorm(8, c_out),
        nn.SiLU(),
    )


def up_gn(c_in: int, c_out: int, k: int = 3) -> nn.Sequential:
    return nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), conv_gn(c_in, c_out, k))


class Autoencoder(nn.Module):
    """32 -> 16 -> 8 -> 4 conv encoder to a `latent_channels`x4x4 code, mirrored decoder."""

    def __init__(self, in_channels: int = 1, width: int = 8, latent_channels: int = 4, kernel_size: int = 5):
        super().__init__()
        w, k = width, kernel_size
        self.encoder = nn.Sequential(
            conv_gn(in_channels, w, k),
            conv_gn(w, w, k),
            conv_gn(w, 2 * w, k, stride=2),  # 16
            conv_gn(2 * w, 2 * w, k),
            conv_gn(2 * w, 4 * w, k, stride=2),  # 8
            conv_gn(4 * w, 4 * w, k),
            conv_gn(4 * w, 4 * w, k, stride=2),  # 4
            conv_gn(4 * w, 4 * w, k),
            nn.Conv2d(4 * w, latent_channels, 1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 4 * w, k, padding=k // 2),
            conv_gn(4 * w, 4 * w, k),
            up_gn(4 * w, 4 * w, k),  # 8
            conv_gn(4 * w, 4 * w, k),
            up_gn(4 * w, 2 * w, k),  # 16
            conv_gn(2 * w, 2 * w, k),
            up_gn(2 * w, w, k),  # 32
            conv_gn(w, w, k),
            nn.Conv2d(w, in_channels, k, padding=k // 2),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor, mask_prob: float = 0.0) -> torch.Tensor:
        z = self.encode(x)
        if mask_prob > 0:
            z = self.mask_detail(z, mask_prob)
        return self.decode(z)

    @staticmethod
    def mask_detail(z: torch.Tensor, p: float) -> torch.Tensor:
        """Zero the second half of the channels of each sample independently with probability p."""
        keep = (torch.rand(len(z), 1, 1, 1, device=z.device) >= p).to(z.dtype)
        half = z.shape[1] // 2
        return torch.cat([z[:, :half], z[:, half:] * keep], dim=1)


def build_ae(cfg: AEConfig) -> Autoencoder:
    from zoo import data

    ds = getattr(data, cfg.dataset)
    return Autoencoder(in_channels=ds.image_shape[0], width=cfg.width, latent_channels=cfg.latent_channels, kernel_size=cfg.kernel_size)


# --- flow model ----------------------------------------------------------------


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding of t in [0, 1] (scaled by 1000 so frequencies cover the range)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = 1000.0 * t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResBlock(nn.Module):
    """Pre-norm residual conv block; the (time + class) embedding modulates it via scale/shift."""

    def __init__(self, ch: int, emb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, 2 * ch)
        self.norm2 = nn.GroupNorm(32, ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(F.silu(emb))[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.dropout(F.silu(h)))
        return x + h


class FlowNet(nn.Module):
    """Predicts the velocity v(x_t, t, y). Label `num_classes` is the "null" (unconditional) class."""

    def __init__(self, latent_channels: int, num_classes: int, width: int = 64, depth: int = 6, emb_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.time_mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.class_emb = nn.Embedding(num_classes + 1, emb_dim)
        self.inp = nn.Conv2d(latent_channels, width, 3, padding=1)
        self.blocks = nn.ModuleList(ResBlock(width, emb_dim, dropout) for _ in range(depth))
        self.out = nn.Sequential(nn.GroupNorm(32, width), nn.SiLU(), nn.Conv2d(width, latent_channels, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        emb = self.time_mlp(timestep_embedding(t, self.emb_dim)) + self.class_emb(y)
        h = self.inp(x)
        for block in self.blocks:
            h = block(h, emb)
        return self.out(h)


def build_flow(cfg: FlowConfig, ae_cfg: AEConfig) -> FlowNet:
    from zoo import data

    ds = getattr(data, ae_cfg.dataset)
    return FlowNet(
        latent_channels=ae_cfg.latent_channels,
        num_classes=ds.num_classes,
        width=cfg.width,
        depth=cfg.depth,
        emb_dim=cfg.emb_dim,
        dropout=cfg.dropout,
    )


def rectified_flow_loss(model: FlowNet, x1: torch.Tensor, y: torch.Tensor, x0: torch.Tensor | None = None, t: torch.Tensor | None = None) -> torch.Tensor:
    """MSE between predicted and true velocity on the straight path x_t = (1-t) x0 + t x1."""
    if x0 is None:
        x0 = torch.randn_like(x1)
    if t is None:
        t = torch.rand(len(x1), device=x1.device)
    tb = t[:, None, None, None]
    xt = (1 - tb) * x0 + tb * x1
    return F.mse_loss(model(xt, t, y), x1 - x0)


@torch.no_grad()
def sample_latents(model: FlowNet, y: torch.Tensor, shape: tuple[int, ...], steps: int = 50, cfg_scale: float = 1.0, noise: torch.Tensor | None = None) -> torch.Tensor:
    """Euler-integrate dx/dt = v from t=0 (noise) to t=1, with classifier-free guidance.

    cfg_scale=1 is plain conditional sampling; >1 pushes samples further toward the class.
    """
    x = noise if noise is not None else torch.randn(len(y), *shape, device=y.device)
    null = torch.full_like(y, model.num_classes)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((len(y),), i * dt, device=y.device)
        if cfg_scale == 1.0:
            v = model(x, t, y)
        else:
            v_c, v_u = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([y, null])).chunk(2)
            v = v_u + cfg_scale * (v_c - v_u)
        x = x + dt * v
    return x


# --- checkpoints ---------------------------------------------------------------


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


def save_flow(
    path: Path,
    model: FlowNet,
    cfg: FlowConfig,
    ae: Autoencoder,
    ae_cfg: AEConfig,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    epoch: int,
    metrics: dict,
) -> None:
    """Self-contained: bundles the AE and latent stats, so sampling needs only this file."""
    _atomic_save(
        {
            "config": cfg.to_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "model": model.state_dict(),
            "ae": {"config": ae_cfg.to_dict(), "model": ae.state_dict()},
            "latent_mean": latent_mean.cpu(),
            "latent_std": latent_std.cpu(),
        },
        path,
    )


@dataclass
class Dreamer:
    """Everything needed to generate digits: flow, AE, and the latent normalization."""

    flow: FlowNet
    ae: Autoencoder
    latent_mean: torch.Tensor  # (1, C, 1, 1)
    latent_std: torch.Tensor
    cfg: FlowConfig
    ae_cfg: AEConfig

    def to(self, device) -> Dreamer:
        self.flow.to(device).eval()
        self.ae.to(device).eval()
        self.latent_mean = self.latent_mean.to(device)
        self.latent_std = self.latent_std.to(device)
        return self

    @torch.no_grad()
    def generate(self, y: torch.Tensor, steps: int | None = None, cfg_scale: float | None = None, noise: torch.Tensor | None = None) -> torch.Tensor:
        """Class labels -> images in [-1, 1]."""
        shape = (self.ae_cfg.latent_channels, LATENT_SIZE, LATENT_SIZE)
        z = sample_latents(
            self.flow,
            y,
            shape,
            steps=steps or self.cfg.sample_steps,
            cfg_scale=self.cfg.cfg_scale if cfg_scale is None else cfg_scale,
            noise=noise,
        )
        return self.ae.decode(z * self.latent_std + self.latent_mean).clamp(-1, 1)


def load_dreamer(path: str | Path, map_location="cpu") -> Dreamer:
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    cfg = FlowConfig.from_dict(ckpt["config"])
    ae_cfg = AEConfig.from_dict(ckpt["ae"]["config"])
    ae = build_ae(ae_cfg)
    ae.load_state_dict(ckpt["ae"]["model"])
    flow = build_flow(cfg, ae_cfg)
    flow.load_state_dict(ckpt["model"])
    return Dreamer(flow, ae, ckpt["latent_mean"], ckpt["latent_std"], cfg, ae_cfg)
