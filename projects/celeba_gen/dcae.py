"""DC-AE-style autoencoder (Chen et al., "Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models").

The encoder halves the resolution `log2(downsample)` times (f8, f16, f32, f64, ...) and the latent is
`latent_channels` x R/downsample x R/downsample. Nothing is tied to one resolution except that R must be a multiple of
`downsample`, so a model trained at 128 px runs at 256 px: that gap is what phase 2 of `train_dcae.py` adapts.

Residual autoencoding: every down / up block and both latent projections carry a non-parametric shortcut, and the
layers learn the residual on top of it. The shortcut is a space-to-channel move, not an identity:

  down   conv + pixel_unshuffle   +  pixel_unshuffle, then average groups of channels    (HxWxC -> H/2xW/2xC')
  up     conv + pixel_shuffle     +  repeat channels, then pixel_shuffle                 (HxWxC -> 2Hx2WxC')
  latent conv                     +  average groups of channels (encoder) / repeat them (decoder)

Differences from the paper: residual blocks are plain 3x3 convs with GroupNorm and SiLU (no EfficientViT blocks), and
widths are small by default.

Parameter names are `encoder.{stem, stages.i, bottom, proj}` and `decoder.{proj, bottom, stages.j, out}`; the training
script picks what to train by these prefixes (see `set_trainable`). Images live in [-1, 1] and the output is not clamped.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_CHANNELS = 3
GROUPS = 8  # GroupNorm groups, so every width must be a multiple of this


class _Serializable:
    def to_dict(self) -> dict:
        return asdict(self)  # type: ignore[call-overload]

    @classmethod
    def from_dict(cls, d: dict):
        known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
        return cls(**{k: v for k, v in d.items() if k in known})


def _atomic_save(obj: dict, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


@dataclass
class DCAEConfig(_Serializable):
    downsample: int = 32  # total stride, a power of two: the latent is latent_channels x R/downsample x R/downsample
    latent_channels: int = 32  # must divide the deepest width (the latent shortcut averages / repeats channel groups)
    width: int = 32  # channels at full resolution; doubles at every stage up to max_width (multiple of 8)
    max_width: int = 256  # width cap; max_width / width must be a power of two
    blocks: int = 2  # residual blocks per stage, in the encoder and in the decoder
    kernel_size: int = 3  # for every spatial conv

    def __post_init__(self) -> None:
        d, n = self.downsample, self.n_stages
        if d < 2 or 2**n != d:
            raise ValueError(f"downsample must be a power of two >= 2, got {d}")
        if self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {self.kernel_size}")
        ws = self.widths
        if any(w % GROUPS for w in ws):
            raise ValueError(f"widths {ws} must all be multiples of {GROUPS} (GroupNorm groups)")
        if any(b not in (a, 2 * a) for a, b in zip(ws, ws[1:])):
            raise ValueError(f"widths {ws}: max_width / width must be a power of two")
        if ws[-1] % self.latent_channels:
            raise ValueError(f"latent_channels {self.latent_channels} must divide the deepest width {ws[-1]}")

    @property
    def n_stages(self) -> int:
        return int(math.log2(self.downsample)) if self.downsample > 0 else 0

    @property
    def widths(self) -> list[int]:
        """Channels before each downsample, then at the bottom: n_stages + 1 values."""
        return [min(self.width * 2**i, self.max_width) for i in range(self.n_stages + 1)]


class ResBlock(nn.Module):
    def __init__(self, c: int, k: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.GroupNorm(GROUPS, c),
            nn.SiLU(),
            nn.Conv2d(c, c, k, padding=k // 2),
            nn.GroupNorm(GROUPS, c),
            nn.SiLU(),
            nn.Conv2d(c, c, k, padding=k // 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


def res_blocks(c: int, k: int, n: int) -> nn.Sequential:
    return nn.Sequential(*(ResBlock(c, k) for _ in range(n)))


def _group_mean(x: torch.Tensor, channels: int) -> torch.Tensor:
    """Average consecutive groups of channels down to `channels`."""
    return x.unflatten(1, (channels, x.shape[1] // channels)).mean(2)


class Down(nn.Module):
    """Halve the resolution: (conv, pixel_unshuffle) + (pixel_unshuffle, channel averaging)."""

    def __init__(self, c_in: int, c_out: int, k: int):
        super().__init__()
        self.c_out = c_out
        self.conv = nn.Conv2d(c_in, c_out // 4, k, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pixel_unshuffle(self.conv(x), 2) + _group_mean(F.pixel_unshuffle(x, 2), self.c_out)


class Up(nn.Module):
    """Double the resolution: (conv, pixel_shuffle) + (channel duplicating, pixel_shuffle)."""

    def __init__(self, c_in: int, c_out: int, k: int):
        super().__init__()
        self.repeats = 4 * c_out // c_in
        self.conv = nn.Conv2d(c_in, 4 * c_out, k, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pixel_shuffle(self.conv(x), 2) + F.pixel_shuffle(x.repeat_interleave(self.repeats, dim=1), 2)


class ToLatent(nn.Module):
    """Encoder output: conv + channel averaging."""

    def __init__(self, c_in: int, latent: int, k: int):
        super().__init__()
        self.latent = latent
        self.conv = nn.Conv2d(c_in, latent, k, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + _group_mean(x, self.latent)


class FromLatent(nn.Module):
    """Decoder input: conv + channel duplicating."""

    def __init__(self, latent: int, c_out: int, k: int):
        super().__init__()
        self.repeats = c_out // latent
        self.conv = nn.Conv2d(latent, c_out, k, padding=k // 2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.conv(z) + z.repeat_interleave(self.repeats, dim=1)


class Encoder(nn.Module):
    def __init__(self, cfg: DCAEConfig):
        super().__init__()
        ws, k, n = cfg.widths, cfg.kernel_size, cfg.blocks
        self.stem = nn.Conv2d(IMAGE_CHANNELS, ws[0], k, padding=k // 2)
        self.stages = nn.ModuleList(nn.Sequential(res_blocks(ws[i], k, n), Down(ws[i], ws[i + 1], k)) for i in range(cfg.n_stages))
        self.bottom = res_blocks(ws[-1], k, n)
        self.proj = ToLatent(ws[-1], cfg.latent_channels, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return self.proj(self.bottom(x))


class Decoder(nn.Module):
    def __init__(self, cfg: DCAEConfig):
        super().__init__()
        ws, k, n, s = cfg.widths, cfg.kernel_size, cfg.blocks, cfg.n_stages
        self.proj = FromLatent(cfg.latent_channels, ws[-1], k)
        self.bottom = res_blocks(ws[-1], k, n)
        # stages[0] is the deepest: it upsamples ws[s] -> ws[s-1]
        self.stages = nn.ModuleList(nn.Sequential(Up(ws[s - j], ws[s - j - 1], k), res_blocks(ws[s - j - 1], k, n)) for j in range(s))
        self.out = nn.Sequential(nn.GroupNorm(GROUPS, ws[0]), nn.SiLU(), nn.Conv2d(ws[0], IMAGE_CHANNELS, k, padding=k // 2))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.bottom(self.proj(z))
        for stage in self.stages:
            x = stage(x)
        return self.out(x)


class DCAE(nn.Module):
    def __init__(self, cfg: DCAEConfig):
        super().__init__()
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class PatchDiscriminator(nn.Module):
    """PatchGAN: 4x4 convs (three stride 2, then stride 1) with GroupNorm and LeakyReLU, one logit per patch."""

    def __init__(self, width: int = 64, in_channels: int = IMAGE_CHANNELS):
        super().__init__()
        if width % GROUPS:
            raise ValueError(f"discriminator width must be a multiple of {GROUPS}, got {width}")
        w = width
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, w, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(w, 2 * w, 4, 2, 1),
            nn.GroupNorm(GROUPS, 2 * w),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2 * w, 4 * w, 4, 2, 1),
            nn.GroupNorm(GROUPS, 4 * w),
            nn.LeakyReLU(0.2),
            nn.Conv2d(4 * w, 8 * w, 4, 1, 1),
            nn.GroupNorm(GROUPS, 8 * w),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8 * w, 1, 4, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_dcae(cfg: DCAEConfig) -> DCAE:
    return DCAE(cfg)


def trainable_prefixes(selectors: tuple[str, ...] | list[str], n_stages: int, edge_stages: int = 1) -> list[str]:
    """Expand selectors into parameter-name prefixes. A selector is one of

      all           every parameter
      middle        what sits next to the latent: the last `edge_stages` encoder stages, `encoder.bottom`,
                    `encoder.proj`, `decoder.proj`, `decoder.bottom` and the first `edge_stages` decoder stages
      decoder-head  what sits next to the output: the last `edge_stages` decoder stages and `decoder.out`

    or a literal prefix such as `encoder.stages.4` or `decoder.out`."""
    d = max(0, min(edge_stages, n_stages))
    presets = {
        "all": [""],
        "middle": [
            *(f"encoder.stages.{i}" for i in range(n_stages - d, n_stages)),
            "encoder.bottom",
            "encoder.proj",
            "decoder.proj",
            "decoder.bottom",
            *(f"decoder.stages.{j}" for j in range(d)),
        ],
        "decoder-head": [*(f"decoder.stages.{j}" for j in range(n_stages - d, n_stages)), "decoder.out"],
    }
    return [p for s in selectors for p in presets.get(s, [s])]


def set_trainable(model: nn.Module, selectors: tuple[str, ...] | list[str], n_stages: int, edge_stages: int = 1) -> list[str]:
    """Freeze every parameter except those under the selected prefixes (see `trainable_prefixes`).

    Returns the prefixes. A prefix that matches no parameter raises, so a typo cannot silently freeze everything."""
    prefixes = trainable_prefixes(selectors, n_stages, edge_stages)
    names = [n for n, _ in model.named_parameters()]
    hit = lambda name, p: p == "" or name == p or name.startswith(p + ".")  # noqa: E731
    for p in prefixes:
        if not any(hit(n, p) for n in names):
            groups = sorted({".".join(n.split(".")[:3]) for n in names})
            raise ValueError(f"--train {p!r} matches no parameter; groups look like: {', '.join(groups[:8])}, ...")
    for n, param in model.named_parameters():
        param.requires_grad_(any(hit(n, p) for p in prefixes))
    return prefixes


def save_dcae(path: Path, model: DCAE, cfg: DCAEConfig, train_cfg: dict, epoch: int, metrics: dict) -> None:
    """`train_cfg` is the phase config as a dict (it includes the phase number)."""
    _atomic_save({"model_config": cfg.to_dict(), "train_config": train_cfg, "epoch": epoch, "metrics": metrics, "model": model.state_dict()}, path)


def load_dcae(path: str | Path, map_location="cpu") -> tuple[DCAE, DCAEConfig, dict]:
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    cfg = DCAEConfig.from_dict(ckpt["model_config"])
    model = build_dcae(cfg)
    model.load_state_dict(ckpt["model"])
    return model, cfg, ckpt
