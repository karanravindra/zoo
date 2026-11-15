"""Latent diffusion for pixel-art sprites: a conv autoencoder plus a text-conditioned denoiser in its latent space.

Stage 1 (`train_ae.py`): an autoencoder squeezes 4x16x16 RGBA sprites into a `latent_channels` x
(16 >> downsample)^2 latent (8x4x4 by default).
Stage 2 (`train_ldm.py`): a latent diffusion model learns to turn Gaussian noise into (normalized)
latents, conditioned on the caption. Training uses the rectified-flow / flow-matching parameterization:
x_t interpolates linearly between noise and data and the denoiser predicts the velocity (data - noise).
The denoiser is a small DiT: one token per
latent position, adaLN modulation from time + pooled caption, and cross-attention to the caption
tokens from a text transformer trained from scratch jointly with the denoiser. Sampling integrates the
predicted velocity with Euler steps and classifier-free guidance, then decodes with the AE decoder.

Conventions: images live in [-1, 1] (RGBA, RGB zeroed where transparent); diffusion time runs from t=0
(noise) to t=1 (data).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from pixel_data import CHANNELS, IMAGE_SIZE, NULL_ID, PAD_ID, Tokenizer


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
    val_size: int = 4096  # train sprites held out for validation (the official validation split is the test set)
    # model
    width: int = 32  # must be a multiple of 8 (GroupNorm)
    kernel_size: int = 3  # for every spatial conv in the autoencoder
    downsample: int = 2  # stride-2 stages; latent side = 16 >> downsample
    latent_channels: int = 8
    # Per training sample, with this probability the second (detail) half of the latent channels is zeroed
    # before decoding, so the first half must carry global information and the second half only refines it.
    mask_prob: float = 0.0
    # Alpha in the data is exactly 0 or 255. If set, the decoder's alpha channel is an opacity logit trained with
    # BCE, RGB loss counts opaque pixels only, and decoded images get hard alpha with RGB blanked where transparent.
    # If unset, all four channels are plain MSE regression (how checkpoints saved before this flag were trained).
    binary_alpha: bool = True
    alpha_weight: float = 0.1  # weight of the alpha BCE relative to the opaque-pixel RGB MSE (binary_alpha only)
    # augmentation (training batches only; val/test are never flipped)
    hflip_prob: float = 0.0  # per-sample probability of mirroring a training sprite left-right
    # optimization
    batch_size: int = 256
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0  # max global gradient norm (0 disables); grad_norm columns in metrics.csv show how often it bites
    # runtime
    seed: int = 0
    deterministic: bool = True
    compile: bool = False  # torch.compile the model with mode="reduce-overhead" (CUDA graphs on GPU)
    dev: bool = False  # dev run: outputs go to a temporary directory deleted on exit, so nothing is logged to runs/
    log_every: int = 200

    @property
    def latent_size(self) -> int:
        return IMAGE_SIZE >> self.downsample

    @classmethod
    def from_dict(cls, d: dict) -> AEConfig:
        # keys older configs lack get the values those runs actually used, not today's defaults
        return super().from_dict({"binary_alpha": False, "alpha_weight": 1.0, "grad_clip": 0.0, **d})


@dataclass
class LDMConfig(_Serializable):
    ae_ckpt: str | None = None  # relative to the repo root; None -> best.pt of the latest AE run
    # model
    width: int = 256  # transformer width; also the text encoder width
    depth: int = 8
    heads: int = 4
    text_layers: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    # optimization
    batch_size: int = 512
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 0.0
    ema_decay: float = 0.9995
    uncond_prob: float = 0.1  # prob. of replacing the caption with the empty prompt (enables classifier-free guidance)
    # sampling (for per-epoch previews)
    sample_steps: int = 50
    guidance_scale: float = 3.0
    preview_prompts: int = 16  # validation captions to preview each epoch
    # runtime
    seed: int = 0
    deterministic: bool = True
    compile: bool = False  # torch.compile the model with mode="reduce-overhead" (CUDA graphs on GPU)
    dev: bool = False  # dev run: outputs go to a temporary directory deleted on exit, so nothing is logged to runs/
    log_every: int = 200


# --- autoencoder ---------------------------------------------------------------


def conv_gn(c_in: int, c_out: int, k: int = 3, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, k, stride=stride, padding=k // 2, bias=False),
        nn.GroupNorm(8, c_out),
        nn.SiLU(),
    )


def up_gn(c_in: int, c_out: int, k: int = 3) -> nn.Sequential:
    return nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), conv_gn(c_in, c_out, k))


class Autoencoder(nn.Module):
    """Conv encoder with `downsample` stride-2 stages to a `latent_channels` code, mirrored decoder."""

    def __init__(self, in_channels: int = CHANNELS, width: int = 32, latent_channels: int = 8, kernel_size: int = 3, downsample: int = 2, binary_alpha: bool = False):
        super().__init__()
        self.binary_alpha = binary_alpha
        k = kernel_size
        ch = [width * min(2**i, 4) for i in range(downsample + 1)]
        enc: list[nn.Module] = [conv_gn(in_channels, ch[0], k)]
        for i in range(downsample):
            enc += [conv_gn(ch[i], ch[i + 1], k, stride=2), conv_gn(ch[i + 1], ch[i + 1], k)]
        enc.append(nn.Conv2d(ch[-1], latent_channels, 1))
        dec: list[nn.Module] = [nn.Conv2d(latent_channels, ch[-1], k, padding=k // 2), conv_gn(ch[-1], ch[-1], k)]
        for i in reversed(range(downsample)):
            dec += [up_gn(ch[i + 1], ch[i], k), conv_gn(ch[i], ch[i], k)]
        dec.append(nn.Conv2d(ch[0], in_channels, k, padding=k // 2))
        self.encoder, self.decoder = nn.Sequential(*enc), nn.Sequential(*dec)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor, mask_prob: float = 0.0) -> torch.Tensor:
        z = self.encode(x)
        if mask_prob > 0:
            z = self.mask_detail(z, mask_prob)
        return self.decode(z)

    def to_image(self, out: torch.Tensor) -> torch.Tensor:
        """Raw decoder output -> RGBA image in [-1, 1]. With binary_alpha, alpha is the thresholded logit and RGB is
        set to -1 (uint8 0) where transparent, matching the data."""
        if not self.binary_alpha:
            return out.clamp(-1, 1)
        opaque = out[:, 3:] > 0
        rgb = torch.where(opaque, out[:, :3].clamp(-1, 1), -1.0)
        return torch.cat([rgb, opaque.to(out.dtype) * 2 - 1], dim=1)

    def loss(self, out: torch.Tensor, x: torch.Tensor, alpha_weight: float = 1.0) -> torch.Tensor:
        """Training loss of raw decoder output `out` against target images `x` in [-1, 1]."""
        if not self.binary_alpha:
            return F.mse_loss(out, x)
        opaque = (x[:, 3:] > 0).to(out.dtype)
        rgb_mse = ((out[:, :3] - x[:, :3]) ** 2 * opaque).sum() / (3 * opaque.sum()).clamp_min(1)
        return rgb_mse + alpha_weight * F.binary_cross_entropy_with_logits(out[:, 3:], opaque)

    @staticmethod
    def mask_detail(z: torch.Tensor, p: float) -> torch.Tensor:
        """Zero the second half of the channels of each sample independently with probability p."""
        keep = (torch.rand(len(z), 1, 1, 1, device=z.device) >= p).to(z.dtype)
        half = z.shape[1] // 2
        return torch.cat([z[:, :half], z[:, half:] * keep], dim=1)


def build_ae(cfg: AEConfig) -> Autoencoder:
    return Autoencoder(CHANNELS, cfg.width, cfg.latent_channels, cfg.kernel_size, cfg.downsample, cfg.binary_alpha)


# --- denoiser ----------------------------------------------------------------


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding of t in [0, 1] (scaled by 1000 so frequencies cover the range)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = 1000.0 * t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def uncond_tokens(n: int, length: int, device) -> torch.Tensor:
    """The unconditional (empty) prompt: a single <null> token, rest padding."""
    t = torch.zeros(n, length, dtype=torch.long, device=device)
    t[:, 0] = NULL_ID
    return t


class TextEncoder(nn.Module):
    """Token embedding + learned positions + pre-norm transformer. Returns (features, valid-token mask)."""

    def __init__(self, vocab_size: int, max_len: int, dim: int, layers: int, heads: int):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, dim, padding_idx=PAD_ID)
        self.pos = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(dim), enable_nested_tensor=False)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pad = tokens == PAD_ID
        h = self.encoder(self.tok(tokens) + self.pos[:, : tokens.shape[1]], src_key_padding_mask=pad)
        return h, ~pad


class Attention(nn.Module):
    """Multi-head attention; `ctx` makes it cross-attention, `mask` is (B, 1, 1, Lk) with True = attend."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor | None = None, mask: torch.Tensor | None = None) -> torch.Tensor:
        ctx = x if ctx is None else ctx
        b, n, d = x.shape
        q = self.q(x).view(b, n, self.heads, -1).transpose(1, 2)
        k, v = self.kv(ctx).view(b, ctx.shape[1], 2, self.heads, -1).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(out.transpose(1, 2).reshape(b, n, d))


class Block(nn.Module):
    """Self-attn, cross-attn to the caption, MLP; each branch is adaLN-modulated and gated by the condition (zero-init)."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        ln = lambda: nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)  # noqa: E731
        self.n1, self.n2, self.n3 = ln(), ln(), ln()
        self.self_attn, self.cross_attn = Attention(dim, heads), Attention(dim, heads)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))
        nn.init.zeros_(self.mod[-1].weight)
        nn.init.zeros_(self.mod[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor, ctx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        sh1, sc1, g1, sh2, sc2, g2, sh3, sc3, g3 = self.mod(c)[:, None].chunk(9, dim=-1)
        x = x + g1 * self.self_attn(self.n1(x) * (1 + sc1) + sh1)
        x = x + g2 * self.cross_attn(self.n2(x) * (1 + sc2) + sh2, ctx, mask)
        return x + g3 * self.mlp(self.n3(x) * (1 + sc3) + sh3)


class Denoiser(nn.Module):
    """Predicts the velocity v(x_t, t, caption). Captions are token ids; `uncond_tokens` is the empty prompt."""

    def __init__(
        self,
        latent_channels: int,
        latent_size: int,
        vocab_size: int,
        max_tokens: int,
        width: int = 256,
        depth: int = 8,
        heads: int = 4,
        text_layers: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.width = width
        self.text = TextEncoder(vocab_size, max_tokens, width, text_layers, heads)
        self.time_mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.text_pool = nn.Linear(width, width)
        self.inp = nn.Linear(latent_channels, width)
        self.pos = nn.Parameter(torch.randn(1, latent_size**2, width) * 0.02)
        self.blocks = nn.ModuleList(Block(width, heads, mlp_ratio, dropout) for _ in range(depth))
        self.norm_out = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.mod_out = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.out = nn.Linear(width, latent_channels)
        for m in (self.mod_out[-1], self.out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        ctx, valid = self.text(tokens)
        pooled = (ctx * valid[..., None]).sum(1) / valid.sum(1, keepdim=True)
        cond = self.time_mlp(timestep_embedding(t, self.width)) + self.text_pool(pooled)
        tok = self.inp(x.flatten(2).transpose(1, 2)) + self.pos
        mask = valid[:, None, None, :]
        for block in self.blocks:
            tok = block(tok, cond, ctx, mask)
        shift, scale = self.mod_out(cond)[:, None].chunk(2, dim=-1)
        tok = self.out(self.norm_out(tok) * (1 + scale) + shift)
        return tok.transpose(1, 2).reshape(b, c, h, w)


def build_denoiser(cfg: LDMConfig, ae_cfg: AEConfig, tokenizer: Tokenizer) -> Denoiser:
    return Denoiser(
        latent_channels=ae_cfg.latent_channels,
        latent_size=ae_cfg.latent_size,
        vocab_size=len(tokenizer),
        max_tokens=tokenizer.max_len,
        width=cfg.width,
        depth=cfg.depth,
        heads=cfg.heads,
        text_layers=cfg.text_layers,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
    )


def diffusion_loss(model: Denoiser, x1: torch.Tensor, tokens: torch.Tensor, x0: torch.Tensor | None = None, t: torch.Tensor | None = None) -> torch.Tensor:
    """Denoising loss: MSE between the predicted and true velocity (x1 - x0) at x_t = (1-t) x0 + t x1, x0 ~ N(0, I)."""
    if x0 is None:
        x0 = torch.randn_like(x1)
    if t is None:
        t = torch.rand(len(x1), device=x1.device)
    tb = t[:, None, None, None]
    xt = (1 - tb) * x0 + tb * x1
    return F.mse_loss(model(xt, t, tokens), x1 - x0)


@torch.no_grad()
def sample_latents(model: Denoiser, tokens: torch.Tensor, shape: tuple[int, ...], steps: int = 50, guidance_scale: float = 1.0, noise: torch.Tensor | None = None) -> torch.Tensor:
    """Euler-integrate dx/dt = v from t=0 (noise) to t=1, with classifier-free guidance.

    guidance_scale=1 is plain conditional sampling; >1 pushes samples further toward the caption.
    """
    x = noise if noise is not None else torch.randn(len(tokens), *shape, device=tokens.device)
    uncond = uncond_tokens(len(tokens), tokens.shape[1], tokens.device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((len(tokens),), i * dt, device=tokens.device)
        if guidance_scale == 1.0:
            v = model(x, t, tokens)
        else:
            v_c, v_u = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([tokens, uncond])).chunk(2)
            v = v_u + guidance_scale * (v_c - v_u)
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


def save_ldm(
    path: Path,
    model: Denoiser,
    cfg: LDMConfig,
    ae: Autoencoder,
    ae_cfg: AEConfig,
    tokenizer: Tokenizer,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    epoch: int,
    metrics: dict,
) -> None:
    """Self-contained: bundles the AE, tokenizer and latent stats, so sampling needs only this file."""
    _atomic_save(
        {
            "config": cfg.to_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "model": model.state_dict(),
            "ae": {"config": ae_cfg.to_dict(), "model": ae.state_dict()},
            "tokenizer": tokenizer.to_dict(),
            "latent_mean": latent_mean.cpu(),
            "latent_std": latent_std.cpu(),
        },
        path,
    )


@dataclass
class LatentDiffusion:
    """Everything needed to generate sprites: denoiser, AE, tokenizer, and the latent normalization."""

    denoiser: Denoiser
    ae: Autoencoder
    tokenizer: Tokenizer
    latent_mean: torch.Tensor  # (1, C, 1, 1)
    latent_std: torch.Tensor
    cfg: LDMConfig
    ae_cfg: AEConfig

    def to(self, device) -> LatentDiffusion:
        self.denoiser.to(device).eval()
        self.ae.to(device).eval()
        self.latent_mean = self.latent_mean.to(device)
        self.latent_std = self.latent_std.to(device)
        return self

    @property
    def device(self) -> torch.device:
        return self.latent_mean.device

    def encode_prompts(self, prompts: list[str]) -> torch.Tensor:
        return torch.from_numpy(self.tokenizer.encode(prompts)).long().to(self.device)

    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, steps: int | None = None, guidance_scale: float | None = None, noise: torch.Tensor | None = None) -> torch.Tensor:
        """Token ids (N, L) -> RGBA images in [-1, 1]."""
        s = self.ae_cfg.latent_size
        z = sample_latents(
            self.denoiser,
            tokens.long(),
            (self.ae_cfg.latent_channels, s, s),
            steps=steps or self.cfg.sample_steps,
            guidance_scale=self.cfg.guidance_scale if guidance_scale is None else guidance_scale,
            noise=noise,
        )
        return self.ae.to_image(self.ae.decode(z * self.latent_std + self.latent_mean))

    def generate_prompts(self, prompts: list[str], **kw) -> torch.Tensor:
        return self.generate(self.encode_prompts(prompts), **kw)


def load_ldm(path: str | Path, map_location="cpu") -> LatentDiffusion:
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    cfg = LDMConfig.from_dict(ckpt["config"])
    ae_cfg = AEConfig.from_dict(ckpt["ae"]["config"])
    tokenizer = Tokenizer.from_dict(ckpt["tokenizer"])
    ae = build_ae(ae_cfg)
    ae.load_state_dict(ckpt["ae"]["model"])
    denoiser = build_denoiser(cfg, ae_cfg, tokenizer)
    denoiser.load_state_dict(ckpt["model"])
    return LatentDiffusion(denoiser, ae, tokenizer, ckpt["latent_mean"], ckpt["latent_std"], cfg, ae_cfg)
