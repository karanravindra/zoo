"""Stage 2: a class-conditioned rectified flow in the autoencoder's latent space.

It learns to transport Gaussian noise to (normalized) latents along straight lines. Sampling integrates the learned
velocity with Euler steps, then decodes with the AE. Flow time runs from t=0 (noise) to t=1 (data).
"""

from __future__ import annotations

import math

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from ae import LATENT_SIZE, build_ae
from config import AEConfig, FlowConfig

from zoo import data
from zoo.training import compile_, log_images, make_scheduler

SAMPLES_PER_CLASS = 10  # in the sample grids logged each epoch


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
        return x + self.conv2(self.dropout(F.silu(h)))


class FlowNet(nn.Module):
    """Predicts the velocity v(x_t, t, y). Label `num_classes` is the "null" (unconditional) class."""

    def __init__(self, latent_channels: int, num_classes: int, width: int = 64, depth: int = 6, emb_dim: int = 128,
                 dropout: float = 0.0):
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


def rectified_flow_loss(model: FlowNet, x1: torch.Tensor, y: torch.Tensor, x0: torch.Tensor | None = None,
                        t: torch.Tensor | None = None) -> torch.Tensor:
    """MSE between predicted and true velocity on the straight path x_t = (1-t) x0 + t x1."""
    x0 = torch.randn_like(x1) if x0 is None else x0
    t = torch.rand(len(x1), device=x1.device) if t is None else t
    tb = t[:, None, None, None]
    xt = (1 - tb) * x0 + tb * x1
    return F.mse_loss(model(xt, t, y), x1 - x0)


@torch.no_grad()
def sample_latents(model: FlowNet, y: torch.Tensor, shape: tuple[int, ...], steps: int = 50, cfg_scale: float = 1.0,
                   noise: torch.Tensor | None = None) -> torch.Tensor:
    """Euler-integrate dx/dt = v from t=0 (noise) to t=1, with classifier-free guidance.

    cfg_scale=1 is plain conditional sampling; >1 pushes samples further toward the class."""
    x = torch.randn(len(y), *shape, device=y.device) if noise is None else noise
    null = torch.full_like(y, model.num_classes)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((len(y),), i * dt, device=y.device)
        if cfg_scale == 1.0:
            v = model(x, t, y)
        else:
            v_cond, v_uncond = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([y, null])).chunk(2)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        x = x + dt * v
    return x


class LitFlow(L.LightningModule):
    """The flow bundled with the frozen autoencoder and the latent statistics, so a checkpoint is self-contained:
    `load_module(LitFlow, path)` is all that sampling needs."""

    def __init__(self, cfg: dict, ae_cfg: dict):
        super().__init__()
        self.save_hyperparameters()
        self.cfg, self.ae_cfg = FlowConfig.from_dict(cfg), AEConfig.from_dict(ae_cfg)
        self.num_classes = getattr(data, self.ae_cfg.dataset).num_classes
        self.ae = build_ae(self.ae_cfg).requires_grad_(False)  # the training script copies the weights in
        self.flow = FlowNet(self.ae_cfg.latent_channels, self.num_classes, self.cfg.width, self.cfg.depth,
                            self.cfg.emb_dim, self.cfg.dropout)
        self.register_buffer("latent_mean", torch.zeros(1, self.ae_cfg.latent_channels, 1, 1))  # set in on_fit_start
        self.register_buffer("latent_std", torch.ones(1, self.ae_cfg.latent_channels, 1, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        self.ae.eval()  # frozen
        return self

    def setup(self, stage: str) -> None:
        if self.cfg.compile and stage == "fit":
            compile_(self.flow)

    def on_fit_start(self) -> None:
        self.latent_mean.copy_(self.trainer.datamodule.mean)
        self.latent_std.copy_(self.trainer.datamodule.std)

    @torch.no_grad()
    def generate(self, y: torch.Tensor, steps: int | None = None, cfg_scale: float | None = None,
                 noise: torch.Tensor | None = None) -> torch.Tensor:
        """Class labels -> images in [-1, 1]."""
        shape = (self.ae_cfg.latent_channels, LATENT_SIZE, LATENT_SIZE)
        cfg_scale = self.cfg.cfg_scale if cfg_scale is None else cfg_scale
        z = sample_latents(self.flow, y, shape, steps or self.cfg.sample_steps, cfg_scale, noise)
        return self.ae.decode(z * self.latent_std + self.latent_mean).clamp(-1, 1)

    def training_step(self, batch, batch_idx):
        z, y = batch
        # Randomly drop the label to the null class so one model learns both p(z|y) and p(z).
        drop = torch.rand(len(y), device=z.device) < self.cfg.class_dropout
        y = torch.where(drop, torch.full_like(y, self.num_classes), y)
        loss = rectified_flow_loss(self.flow, z, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, batch_size=len(y))
        return loss

    def _eval_step(self, batch, batch_idx: int, prefix: str) -> None:
        z, y = batch
        g = torch.Generator(device=z.device).manual_seed(batch_idx)  # fixed noise and times: comparable across epochs
        x0 = torch.randn(z.shape, device=z.device, generator=g)
        t = torch.rand(len(z), device=z.device, generator=g)
        self.log(f"{prefix}/loss", rectified_flow_loss(self.flow, z, y, x0, t), on_epoch=True, batch_size=len(z))

    def validation_step(self, batch, batch_idx):
        self._eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._eval_step(batch, batch_idx, "test")

    def _log_samples(self, key: str) -> None:
        """Rows are classes, columns are samples; the noise is fixed so epochs are comparable."""
        device = self.latent_mean.device
        y = torch.arange(self.num_classes, device=device).repeat_interleave(SAMPLES_PER_CLASS)
        g = torch.Generator(device=device).manual_seed(1234)
        noise = torch.randn(len(y), self.ae_cfg.latent_channels, LATENT_SIZE, LATENT_SIZE, device=device, generator=g)
        log_images(self.trainer, key, self.generate(y, noise=noise), nrow=SAMPLES_PER_CLASS)

    def on_validation_epoch_end(self) -> None:  # runs with the EMA weights swapped in
        if not self.trainer.sanity_checking:
            self._log_samples("val/samples")

    def on_test_epoch_end(self) -> None:
        self._log_samples("test/samples")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.flow.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        return {"optimizer": opt, "lr_scheduler": make_scheduler(opt, self.cfg, int(self.trainer.estimated_stepping_batches))}
