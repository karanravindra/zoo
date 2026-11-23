"""Stage 1: a conv autoencoder squeezing 1x32x32 digits (28x28 MNIST padded by 2) into a `latent_channels`x4x4 latent."""

from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import AEConfig

from zoo import data
from zoo.metrics import psnr, ssim
from zoo.training import compile_, log_images, make_scheduler

IMAGE_SIZE = 32  # images are padded to this (see data.TO_PM1)
LATENT_SIZE = 4  # IMAGE_SIZE / 8: three stride-2 stages


def conv_gn(c_in: int, c_out: int, k: int = 3, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(c_in, c_out, k, stride=stride, padding=k // 2, bias=False), nn.GroupNorm(8, c_out), nn.SiLU())


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
    ds = getattr(data, cfg.dataset)
    return Autoencoder(ds.image_shape[0], cfg.width, cfg.latent_channels, cfg.kernel_size)


class LitAutoencoder(L.LightningModule):
    """Trained on reconstruction MSE. Built from the config as a plain dict, so `load_from_checkpoint` rebuilds it."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = AEConfig.from_dict(cfg)
        self.net = build_ae(self.cfg)
        self.preview: torch.Tensor | None = None

    def setup(self, stage: str) -> None:
        if self.cfg.compile and stage == "fit":
            compile_(self.net)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        loss = F.mse_loss(self.net(x, self.cfg.mask_prob), x)
        self.log("train/mse", loss, on_step=True, on_epoch=True, batch_size=len(x))
        return loss

    def _eval_step(self, batch, batch_idx: int, prefix: str) -> None:
        x, _ = batch
        recon = self.net(x)
        mse = F.mse_loss(recon, x)
        recon = recon.clamp(-1, 1)  # a copy, so the next call may overwrite the output buffer (--compile: CUDA graphs)
        self.log_dict(
            {
                f"{prefix}/mse": mse,
                f"{prefix}/psnr": psnr(recon, x).mean(),
                f"{prefix}/ssim": ssim(recon, x).mean(),
                f"{prefix}/mse_global": F.mse_loss(self.net(x, 1.0), x),  # detail half masked: what the global half holds
            },
            on_epoch=True,
            batch_size=len(x),
        )
        if batch_idx == 0:
            self.preview = torch.cat([x[:16], recon[:16]])  # the same images every epoch: the loaders do not shuffle

    def validation_step(self, batch, batch_idx):
        self._eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._eval_step(batch, batch_idx, "test")

    def on_validation_epoch_end(self) -> None:
        if not self.trainer.sanity_checking:
            log_images(self.trainer, "val/recon", self.preview, nrow=16)  # top row: images, bottom row: reconstructions

    def on_test_epoch_end(self) -> None:
        log_images(self.trainer, "test/recon", self.preview, nrow=16)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        return {"optimizer": opt, "lr_scheduler": make_scheduler(opt, self.cfg, int(self.trainer.estimated_stepping_batches))}
