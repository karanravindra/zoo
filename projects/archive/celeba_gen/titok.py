"""TiTok-style 1D tokenizer (Yu et al., "An Image is Worth 32 Tokens for Reconstruction and Generation") with FSQ.

Encoder: patchify the image, add a position embedding, append `num_tokens` learned latent tokens, run a ViT and keep only
the latent tokens. Each one is projected to `len(levels)` channels and quantized with FSQ (finite scalar quantization:
every channel is bounded with tanh and rounded to one of `levels[i]` values, with a straight-through gradient). The latent
is `num_tokens` discrete tokens, a codebook index in [0, prod(levels)) each, whatever the image size. There is no codebook
and no commitment loss, so the only training losses are the reconstruction ones.

Decoder: the quantized tokens are projected up and appended to one learned mask token per patch (plus a position
embedding), a second ViT mixes them, and the mask-token outputs are unpatchified back to pixels.

Image size invariance. Only the patch grid depends on the image size: H/p x W/p patches, any aspect ratio. The position
embeddings are a learned `pos_grid x pos_grid` grid that is bilinearly resized (antialiased when it shrinks) to the patch
grid, so a position keeps its place in the image at any resolution, as in VibeToken (Patel et al., CVPR 2026). The latent
tokens have their own learned embeddings, and there are always `num_tokens` of them. A model trained at 128 px therefore
runs at 256 px as is; what it gets wrong there is what phase 2 of `train_titok.py` adapts. Patch size is fixed, so the
patch count (and the attention cost) grows with the image area. Above 256 px that would call for an adaptive patch
embedding too (VibeToken's grid alone was not enough at 1024 px).

Parameter names are `encoder.{patch_embed, pos, latents, blocks.i, norm, proj}` and
`decoder.{proj, latent_pos, mask, pos, blocks.j, norm, out}`; the training script picks what to train by these prefixes
(see `set_trainable`). Images live in [-1, 1] and the output is not clamped.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_CHANNELS = 3
GROUPS = 8  # GroupNorm groups in the discriminator, so its widths must be multiples of this


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
class TiTokConfig(_Serializable):
    patch_size: int = 16  # pixels per patch side; every image side must be a multiple of it
    width: int = 384  # transformer width (a multiple of `heads`)
    depth: int = 6  # transformer blocks, in the encoder and in the decoder
    heads: int = 6  # attention heads
    mlp_ratio: float = 4.0  # MLP hidden width over `width`
    num_tokens: int = 64  # latent tokens K per image, whatever its size
    levels: tuple[int, ...] = (8, 8, 8, 5, 5, 5)  # FSQ levels per latent channel: len = channels per token, prod = codebook size
    pos_grid: int = 16  # side of the learned position grid, resized to each image's patch grid (16 = the grid of a 256 px image)

    def __post_init__(self) -> None:
        self.levels = tuple(self.levels)  # a checkpoint or config.json may hand back a list
        if self.patch_size < 1 or self.pos_grid < 1 or self.num_tokens < 1 or self.depth < 1:
            raise ValueError("patch_size, pos_grid, num_tokens and depth must be at least 1")
        if self.heads < 1 or self.width % self.heads:
            raise ValueError(f"width {self.width} must be a multiple of heads {self.heads}")
        if not self.levels or any(n < 2 for n in self.levels):
            raise ValueError(f"levels must be a non-empty tuple of integers >= 2, got {self.levels}")

    @property
    def codebook_size(self) -> int:
        return math.prod(self.levels)


class FSQ(nn.Module):
    """Finite scalar quantization. `forward` maps (..., len(levels)) to the same shape with channel i on the grid of
    `levels[i]` values spanning [-1, 1] (an even level count is shifted by half a step, as in the FSQ paper)."""

    def __init__(self, levels: tuple[int, ...]):
        super().__init__()
        lv = torch.tensor(levels)
        # buffers are not persistent: they follow from the config, so state_dict keys stay free of them
        self.register_buffer("levels", lv, persistent=False)
        self.register_buffer("half_width", lv // 2, persistent=False)  # not `half`: that is an nn.Module method
        self.register_buffer("basis", torch.cumprod(torch.tensor([1, *levels[:-1]]), 0), persistent=False)

    def bound(self, z: torch.Tensor) -> torch.Tensor:
        half = (self.levels - 1) * (1 - 1e-3) / 2
        offset = (self.levels % 2 == 0).float() * 0.5
        return (z + (offset / half).atanh()).tanh() * half - offset

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.bound(z.float())
        return (z + (z.round() - z).detach()) / self.half_width  # straight-through rounding

    def to_indices(self, q: torch.Tensor) -> torch.Tensor:
        """Quantized codes (..., C) -> one codebook index per token (...)."""
        digits = (q * self.half_width + self.half_width).round().long()  # 0 .. levels[i] - 1
        return (digits * self.basis).sum(-1)

    def from_indices(self, idx: torch.Tensor) -> torch.Tensor:
        digits = (idx.unsqueeze(-1) // self.basis) % self.levels
        return (digits.float() - self.half_width) / self.half_width


class Block(nn.Module):
    """Pre-norm transformer block with full self-attention."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        x = x + self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, c))
        return x + self.mlp(self.norm2(x))


def patch_grid(size: tuple[int, int] | torch.Size, patch_size: int) -> tuple[int, int]:
    h, w = size
    if h % patch_size or w % patch_size:
        raise ValueError(f"image size {h}x{w} must be a multiple of the patch size {patch_size}")
    return h // patch_size, w // patch_size


def grid_tokens(grid: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """The learned (1, C, G, G) position grid resized to h x w patches, as (1, h*w, C). Bilinear and antialiased, so each
    position keeps its place in the image whatever the resolution; a grid already of that size is used as is."""
    if tuple(grid.shape[-2:]) != (h, w):
        grid = F.interpolate(grid, size=(h, w), mode="bilinear", antialias=True, align_corners=False)
    return grid.flatten(2).transpose(1, 2)


def _embedding(*shape: int) -> nn.Parameter:
    return nn.Parameter(nn.init.trunc_normal_(torch.empty(*shape), std=0.02))


class Encoder(nn.Module):
    def __init__(self, cfg: TiTokConfig):
        super().__init__()
        self.patch_size, self.num_tokens = cfg.patch_size, cfg.num_tokens
        self.patch_embed = nn.Conv2d(IMAGE_CHANNELS, cfg.width, cfg.patch_size, stride=cfg.patch_size)
        self.pos = _embedding(1, cfg.width, cfg.pos_grid, cfg.pos_grid)
        self.latents = _embedding(1, cfg.num_tokens, cfg.width)
        self.blocks = nn.ModuleList(Block(cfg.width, cfg.heads, cfg.mlp_ratio) for _ in range(cfg.depth))
        self.norm = nn.LayerNorm(cfg.width)
        self.proj = nn.Linear(cfg.width, len(cfg.levels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = patch_grid(x.shape[-2:], self.patch_size)
        t = self.patch_embed(x).flatten(2).transpose(1, 2) + grid_tokens(self.pos, h, w)  # (B, h*w, width)
        t = torch.cat([t, self.latents.expand(len(t), -1, -1)], dim=1)
        for block in self.blocks:
            t = block(t)
        return self.proj(self.norm(t[:, -self.num_tokens :]))  # (B, K, len(levels)), before quantization


class Decoder(nn.Module):
    def __init__(self, cfg: TiTokConfig):
        super().__init__()
        self.patch_size = cfg.patch_size
        self.proj = nn.Linear(len(cfg.levels), cfg.width)
        self.latent_pos = _embedding(1, cfg.num_tokens, cfg.width)
        self.mask = _embedding(1, 1, cfg.width)
        self.pos = _embedding(1, cfg.width, cfg.pos_grid, cfg.pos_grid)
        self.blocks = nn.ModuleList(Block(cfg.width, cfg.heads, cfg.mlp_ratio) for _ in range(cfg.depth))
        self.norm = nn.LayerNorm(cfg.width)
        self.out = nn.Linear(cfg.width, IMAGE_CHANNELS * cfg.patch_size**2)

    def forward(self, z: torch.Tensor, size: tuple[int, int] | torch.Size) -> torch.Tensor:
        """Quantized tokens (B, K, len(levels)) -> an image of `size` = (H, W) pixels."""
        h, w = patch_grid(size, self.patch_size)
        patches = (self.mask + grid_tokens(self.pos, h, w)).expand(len(z), -1, -1)  # one mask token per patch
        t = torch.cat([patches, self.proj(z) + self.latent_pos], dim=1)
        for block in self.blocks:
            t = block(t)
        t = self.out(self.norm(t[:, : h * w]))  # (B, h*w, 3*p*p)
        return F.pixel_shuffle(t.transpose(1, 2).reshape(len(z), -1, h, w), self.patch_size)


class TiTok(nn.Module):
    def __init__(self, cfg: TiTokConfig):
        super().__init__()
        self.encoder = Encoder(cfg)
        self.quantizer = FSQ(cfg.levels)
        self.decoder = Decoder(cfg)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Images (B, 3, H, W) -> quantized tokens (B, K, len(levels))."""
        return self.quantizer(self.encoder(x))

    def decode(self, z: torch.Tensor, size: tuple[int, int] | torch.Size) -> torch.Tensor:
        """Quantized tokens -> images of `size` = (H, W); it need not be the size they were encoded from."""
        return self.decoder(z, size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x), x.shape[-2:])

    @torch.no_grad()
    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """Images -> codebook indices (B, K), each in [0, prod(levels))."""
        return self.quantizer.to_indices(self.encode(x))

    @torch.no_grad()
    def detokenize(self, idx: torch.Tensor, size: tuple[int, int] | torch.Size) -> torch.Tensor:
        return self.decode(self.quantizer.from_indices(idx), size)


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


def build_titok(cfg: TiTokConfig) -> TiTok:
    return TiTok(cfg)


def trainable_prefixes(selectors: tuple[str, ...] | list[str], depth: int, edge_blocks: int = 2) -> list[str]:
    """Expand selectors into parameter-name prefixes. A selector is one of

      all           every parameter
      middle        what sits next to the latent: the last `edge_blocks` encoder blocks, `encoder.norm`,
                    `encoder.proj`, `decoder.proj`, `decoder.pos` and the first `edge_blocks` decoder blocks. The
                    encoder's patch embedding, position grid, latent tokens and early blocks stay frozen, so autograd
                    keeps none of their activations
      decoder-head  what sits next to the output: the last `edge_blocks` decoder blocks, `decoder.norm`, `decoder.out`

    or a literal prefix such as `encoder.blocks.5` or `decoder.out`."""
    d = max(0, min(edge_blocks, depth))
    presets = {
        "all": [""],
        "middle": [
            *(f"encoder.blocks.{i}" for i in range(depth - d, depth)),
            "encoder.norm",
            "encoder.proj",
            "decoder.proj",
            "decoder.pos",
            *(f"decoder.blocks.{j}" for j in range(d)),
        ],
        "decoder-head": [*(f"decoder.blocks.{j}" for j in range(depth - d, depth)), "decoder.norm", "decoder.out"],
    }
    return [p for s in selectors for p in presets.get(s, [s])]


def set_trainable(model: nn.Module, selectors: tuple[str, ...] | list[str], depth: int, edge_blocks: int = 2) -> list[str]:
    """Freeze every parameter except those under the selected prefixes (see `trainable_prefixes`).

    Returns the prefixes. A prefix that matches no parameter raises, so a typo cannot silently freeze everything."""
    prefixes = trainable_prefixes(selectors, depth, edge_blocks)
    names = [n for n, _ in model.named_parameters()]
    hit = lambda name, p: p == "" or name == p or name.startswith(p + ".")  # noqa: E731
    for p in prefixes:
        if not any(hit(n, p) for n in names):
            groups = sorted({".".join(n.split(".")[:3]) for n in names})
            raise ValueError(f"--train {p!r} matches no parameter; groups look like: {', '.join(groups[:8])}, ...")
    for n, param in model.named_parameters():
        param.requires_grad_(any(hit(n, p) for p in prefixes))
    return prefixes


def save_titok(path: Path, model: TiTok, cfg: TiTokConfig, train_cfg: dict, epoch: int, metrics: dict) -> None:
    """`train_cfg` is the phase config as a dict (it includes the phase number)."""
    _atomic_save({"model_config": cfg.to_dict(), "train_config": train_cfg, "epoch": epoch, "metrics": metrics, "model": model.state_dict()}, path)


def load_titok(path: str | Path, map_location="cpu") -> tuple[TiTok, TiTokConfig, dict]:
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    cfg = TiTokConfig.from_dict(ckpt["model_config"])
    model = build_titok(cfg)
    model.load_state_dict(ckpt["model"])
    return model, cfg, ckpt
