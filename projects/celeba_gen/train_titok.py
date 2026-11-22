"""Train the TiTok-style tokenizer on CelebA-HQ in three phases, with the resolution changing between them.

Run from anywhere, one subcommand per phase (`--help` after a subcommand lists its flags):

  uv run projects/celeba_gen/train_titok.py phase1
  uv run projects/celeba_gen/train_titok.py phase2      # starts from the latest phase1 best.pt
  uv run projects/celeba_gen/train_titok.py phase3      # starts from the latest phase2 best.pt

  phase1  train every layer at low resolution (default 128 px) with L1 + LPIPS. Defines the architecture (--model.*).
  phase2  adapt to high resolution (default 256 px): only the layers next to the latent are tuned ("middle": the encoder's
          last blocks and projection, the decoder's input projection, position grid and first blocks), the rest is
          frozen. Same losses, no GAN.
  phase3  local refinement: only the decoder head is tuned, at low resolution, with a PatchGAN loss added. The latent
          tokens do not move, so nothing is learned about resolution and the GAN never sees high-resolution images.

The tokenizer takes any image size that is a multiple of its patch size (the position grid is resized, see `titok.py`), so
the phases differ only in resolution, what is trained and the loss. The phases are the same loop with different defaults:
what is trained (--train), the resolution, the loss weights and the optimizer are flags on every phase, so e.g.
`phase2 --gan-weight 0.05` or `phase3 --train all` also work. The architecture is stored in the checkpoint: phases 2 and 3
rebuild the model from `--init` (default: the latest best.pt of the previous phase). The CelebA-HQ cache is 256 px, so
lower resolutions are made by an antialiased bicubic downscale on the GPU and no phase can train above 256.

Each run gets its own directory, `<repo>/runs/celeba_gen/titok-phase<N>/<timestamp>/` (or `<name>/` with `--name`; each
phase takes its own `--name`), containing:
  config.json    the phase config (with the phase number) plus git commit, device and init checkpoint
  metrics.csv    one row per epoch; val_* columns carry the resolution, e.g. val_psnr_256. Phases 2 and 3 also log an
                 epoch-0 row: the init checkpoint before any training
  best.pt        checkpoint chosen by --best-by (rFID by default; rFID / MSE / LPIPS at the training resolution, or the
                 last epoch)
  last.pt        checkpoint from the final epoch
  results.json   test-set rFID, MSE, PSNR, SSIM and LPIPS of best.pt at every evaluation resolution
  loss.png, rfid.png, psnr.png, ssim.png, lpips.png, (gan.png)   curves
  recon.png      test images (top row) and their reconstructions (bottom row) at the training resolution
"""

import json
import warnings
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import ClassVar, Literal

import lpips
import torch
import torch.nn.functional as F
import tyro
from torch.utils.data import Dataset
from torchinfo import summary
from torchvision.utils import save_image

from celeba import SIZE, Batches, load_split, make_splits
from rfid import FeatureStats, Inception, frechet_distance
from titok import PatchDiscriminator, TiTokConfig, build_titok, load_titok, save_titok, set_trainable

from zoo.metrics import psnr, ssim, to_unit
from zoo.monitor import GPU_FIELDS, GpuMonitor
from zoo.plot import plot_curves
from zoo.runs import MetricsLogger, find_finished_run, latest_checkpoint, new_run_dir
from zoo.train import STAT_FIELDS, ParamStats, Throughput, update_ema
from zoo.utils.device import get_device, seed_everything

PROJECT = "celeba_gen"
SHOWN = ("rfid", "psnr", "lpips")  # evaluation metrics printed every epoch (all of them go to metrics.csv)


@dataclass
class PhaseConfig:
    """Flags shared by every phase."""

    phase: ClassVar[int] = 1
    val_size: int = 1000  # images held out of the train split for validation (the official val split is the test set)
    resolution: int = 128  # training resolution, a multiple of the model's patch size and at most 256
    eval_resolutions: tuple[int, ...] = (128, 256)  # validation and test resolutions logged every epoch (plus --resolution)

    train: tuple[str, ...] = ("all",)  # all | middle | decoder-head, or parameter-name prefixes (encoder.blocks.5 decoder.out ...)
    edge_blocks: int = 2  # how many transformer blocks next to the latent / the output count for `middle` and `decoder-head`

    l1_weight: float = 1.0  # L1 between reconstruction and image (a loss weight of 0 turns its term off)
    mse_weight: float = 0.0  # MSE between reconstruction and image
    lpips_weight: float = 1.0  # LPIPS weight; val_lpips is logged either way
    lpips_net: Literal["vgg", "alex", "squeeze"] = "vgg"  # LPIPS backbone, for the loss and val_lpips (values differ between backbones)
    rfid: bool = True  # rFID in every evaluation (Inception features of the images vs their reconstructions); downloads the Inception weights on first use
    gan_weight: float = 0.0  # generator weight of the PatchGAN hinge loss; > 0 also trains a discriminator
    disc_lr: float = 1e-4  # discriminator learning rate (constant; it uses --betas)
    disc_width: int = 64  # discriminator base channels, a multiple of 8

    batch_size: int = 64  # images per step
    epochs: int = 20  # passes over the training split
    optimizer: Literal["adamw", "muon"] = "adamw"  # muon: Muon on the 2D weight matrices inside the transformer blocks, AdamW on every other parameter
    lr: float = 2e-4  # peak learning rate; Muon uses it too, with torch's match_rms_adamw scaling, so an AdamW value carries over
    muon_momentum: float = 0.95  # Muon momentum (--optimizer muon)
    betas: tuple[float, float] = (0.9, 0.999)  # AdamW betas for the tokenizer (with --optimizer muon: for the parameters Muon does not take)
    weight_decay: float = 1e-4  # weight decay for AdamW and Muon
    lr_decay: bool = True  # one-cycle schedule (warmup then decay to ~0); False holds lr constant at --lr
    ema: bool = False  # keep an EMA of the weights; validation, checkpoints and test use the EMA weights
    ema_decay: float = 0.999  # per-step EMA decay (used with --ema)
    grad_clip: float = 1.0  # clip the global gradient norm of the tokenizer; 0 disables
    hflip_prob: float = 0.5  # mirror each training image left-right with this probability (val and test are never flipped)
    best_by: Literal["rfid", "mse", "lpips", "last"] = "rfid"  # which epoch best.pt keeps: lowest validation rFID, MSE or LPIPS at --resolution, or the last

    seed: int = 0  # seeds the split, the loader and the weights
    deterministic: bool = True  # deterministic kernels where torch has them (warns where it has none; can be slower)
    compile: bool = False  # torch.compile the tokenizer with mode="reduce-overhead" (the discriminator and LPIPS stay eager)
    bf16: bool = True  # bfloat16 autocast (cuda only) around the tokenizer, the LPIPS loss and the discriminator; weights, optimizer, loss arithmetic and the LPIPS / rFID metrics stay fp32
    name: str | None = None  # run directory name under runs/celeba_gen/titok-phase<N>/ instead of the timestamp; -2, -3, ... is appended if it exists
    dev: bool = False  # dev run: outputs go to a temporary directory deleted on exit, so nothing is logged to runs/ (--name is ignored)
    skip_duplicate: bool = True  # skip the run if a finished run of this phase has the same config and init checkpoint (name, compile, log_every and dev do not count); --no-skip-duplicate trains anyway. Never skips --dev
    log_every: int = 100  # print the running training loss every this many steps

    def to_dict(self) -> dict:
        return {"phase": self.phase, **asdict(self)}


@dataclass
class Phase1Config(PhaseConfig):
    """Phase 1: train every layer at low resolution with L1 + LPIPS."""

    phase: ClassVar[int] = 1
    model: TiTokConfig = field(default_factory=TiTokConfig)  # the architecture; later phases read it from the checkpoint


@dataclass
class _LaterPhase(PhaseConfig):
    init: str | None = None  # checkpoint to start from; default: the latest best.pt of the previous phase


@dataclass
class Phase2Config(_LaterPhase):
    """Phase 2: adapt to high resolution, tuning only the layers next to the latent."""

    phase: ClassVar[int] = 2
    resolution: int = 256  # training resolution, a multiple of the model's patch size and at most 256
    train: tuple[str, ...] = ("middle",)  # all | middle | decoder-head, or parameter-name prefixes (encoder.blocks.5 decoder.out ...)
    batch_size: int = 32  # images per step
    epochs: int = 5  # passes over the training split
    lr: float = 1e-4  # peak AdamW learning rate
    lr_decay: bool = False  # one-cycle schedule (warmup then decay to ~0); False holds lr constant at --lr


@dataclass
class Phase3Config(_LaterPhase):
    """Phase 3: refine local detail by tuning only the decoder head with a PatchGAN loss, at low resolution."""

    phase: ClassVar[int] = 3
    train: tuple[str, ...] = ("decoder-head",)  # all | middle | decoder-head, or parameter-name prefixes (encoder.blocks.5 decoder.out ...)
    gan_weight: float = 0.05  # generator weight of the PatchGAN hinge loss; > 0 also trains a discriminator
    batch_size: int = 32  # images per step
    epochs: int = 5  # passes over the training split
    lr: float = 5e-5  # peak AdamW learning rate
    betas: tuple[float, float] = (0.5, 0.9)  # AdamW betas for the tokenizer
    lr_decay: bool = False  # one-cycle schedule (warmup then decay to ~0); False holds lr constant at --lr


Phase = Phase1Config | Phase2Config | Phase3Config


def add_terms(sums: dict[str, torch.Tensor], n: int, **terms: torch.Tensor) -> None:
    """Add each term times the batch size `n` to the running sums (kept on the device until read)."""
    for k, v in terms.items():
        sums[k] = sums.get(k, 0) + v.detach() * n


def resize(x: torch.Tensor, res: int) -> torch.Tensor:
    """Antialiased bicubic downscale of a batch in [-1, 1] (the cache is 256 px)."""
    if x.shape[-1] == res:
        return x
    return F.interpolate(x, size=(res, res), mode="bicubic", antialias=True, align_corners=False).clamp(-1, 1)


def build_lpips(device, net: str = "vgg") -> lpips.LPIPS:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # lpips calls torchvision with the deprecated `pretrained` argument
        model = lpips.LPIPS(net=net, verbose=False)
    return model.to(device).eval().requires_grad_(False)  # expects images in [-1, 1]


# config keys that do not change what a run learns; `init` is compared as the resolved checkpoint instead
DUPLICATE_IGNORE = {"name", "dev", "skip_duplicate", "compile", "log_every", "init"}


def is_muon_param(name: str, p: torch.Tensor) -> bool:
    """Muon orthogonalizes 2D matrices (torch's Muon rejects anything else), so it gets the weights inside the transformer
    blocks. AdamW keeps the rest: the patch embedding (a conv), position grids, latent and mask tokens, norms, biases, and
    the matrices that touch the image or the FSQ channels (`encoder.proj`, `decoder.proj`, `decoder.out`)."""
    return p.ndim == 2 and ".blocks." in name


def build_optimizers(model: torch.nn.Module, cfg: PhaseConfig) -> list[torch.optim.Optimizer]:
    """AdamW over the trainable parameters, or with `--optimizer muon` Muon over the block matrices plus AdamW over the
    rest. Muon uses `adjust_lr_fn="match_rms_adamw"`, which scales its step to AdamW's size, so `--lr` and `--weight-decay`
    mean the same for both."""
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    fused = next(model.parameters()).is_cuda  # one fused kernel for the update instead of a launch per tensor
    adamw = lambda ps: torch.optim.AdamW(ps, lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay, fused=fused)  # noqa: E731
    if cfg.optimizer == "adamw":
        return [adamw([p for _, p in named])]
    muon = [p for n, p in named if is_muon_param(n, p)]
    rest = [p for n, p in named if not is_muon_param(n, p)]
    print(f"Muon: {sum(p.numel() for p in muon) / 1e6:.2f}M parameters in {len(muon)} matrices; AdamW: {sum(p.numel() for p in rest) / 1e6:.2f}M in {len(rest)} tensors")
    if not muon:
        print("--optimizer muon: no trainable 2D matrix inside a transformer block, so everything trains with AdamW")
    opts = []
    if muon:
        opts.append(torch.optim.Muon(muon, lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.muon_momentum, adjust_lr_fn="match_rms_adamw"))
    if rest:
        opts.append(adamw(rest))
    return opts


@torch.no_grad()
def evaluate(
    model,
    dl,
    device,
    resolutions: list[int],
    lp,
    inception: Inception | None = None,
    amp: Callable[[], AbstractContextManager] = nullcontext,
    real: dict[int, FeatureStats] | None = None,
) -> dict[str, float]:
    """MSE (per value), PSNR, SSIM and LPIPS (mean over images) of reconstructions clamped to [-1, 1], per resolution,
    and with `inception` the rFID between the images and their reconstructions (computed over the whole `dl`).
    `amp` wraps the model call (bf16 autocast, as in training); the reconstruction is cast back to fp32 first, and LPIPS
    and Inception always run in fp32. `real` caches the Inception statistics of the images per resolution: the images
    never change, so pass the same dict on every call with the same `dl` and they go through Inception (and their
    covariance through its eigendecomposition) on the first call only."""
    model.eval()
    sums = {r: torch.zeros(4, device=device) for r in resolutions}  # se, psnr, ssim, lpips
    real = {} if real is None else real
    new = [r for r in resolutions if r not in real] if inception is not None else []  # resolutions whose image statistics are still to be computed
    feats = {r: (FeatureStats(device) if r in new else real[r], FeatureStats(device)) for r in resolutions} if inception is not None else {}  # images, reconstructions
    n_img = 0
    for x in dl:
        for r in resolutions:
            xr = resize(x, r)
            # Finish with each output before the next model call: with --compile (CUDA graphs) a call overwrites the
            # previous call's output buffer.
            with amp():
                recon = model(xr)
            recon = recon.float()
            s = sums[r]
            s[0] += F.mse_loss(recon, xr, reduction="sum")
            recon = recon.clamp(-1, 1)
            s[1] += psnr(recon, xr).sum()
            s[2] += ssim(recon, xr).sum()
            s[3] += lp(recon, xr).sum()
            if inception is not None:
                if r in new:
                    feats[r][0].update(inception(xr))
                feats[r][1].update(inception(recon))
        n_img += len(x)
    real.update({r: feats[r][0] for r in new})
    out = {}
    for r in resolutions:
        se, psnr_sum, ssim_sum, lpips_sum = sums[r].tolist()
        out |= {f"mse_{r}": se / (n_img * 3 * r * r), f"psnr_{r}": psnr_sum / n_img, f"ssim_{r}": ssim_sum / n_img, f"lpips_{r}": lpips_sum / n_img}
        if inception is not None:
            out[f"rfid_{r}"] = frechet_distance(*feats[r])
    return out


@torch.no_grad()
def save_recon(model, ds: Dataset, device, path: Path, res: int, n: int = 8, amp: Callable[[], AbstractContextManager] = nullcontext) -> None:
    model.eval()
    x = resize(torch.stack([ds[i][0] for i in range(n)]).to(device), res)
    with amp():
        recon = model(x)
    save_image(to_unit(torch.cat([x, recon.float().clamp(-1, 1)])), path, nrow=n, padding=1)


def metric_fields(resolutions: list[int]) -> list[str]:
    val = [f"val_{m}_{r}" for r in resolutions for m in ("rfid", "mse", "psnr", "ssim", "lpips")]
    train = ["train_loss", "train_l1", "train_mse", "train_lpips", "train_psnr", "train_gan", "train_d_loss"]
    return ["epoch", "lr", *train, *val, *STAT_FIELDS, *GPU_FIELDS, "steps_per_s", "epoch_time_s"]


def main(cfg: Phase):
    phase = cfg.phase
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    device = get_device()
    amp_on = cfg.bf16 and device.type == "cuda"
    amp = lambda: torch.autocast(device.type, torch.bfloat16, enabled=amp_on)  # noqa: E731
    print(f"Using device: {device}" + (" (bf16 autocast)" if amp_on else ""))
    if cfg.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if cfg.name is not None and (cfg.name in {"", ".", ".."} or Path(cfg.name).name != cfg.name):
        raise ValueError(f"--name must be a plain directory name without '/', got {cfg.name!r}")  # it would otherwise leave or nest inside runs/
    if cfg.l1_weight == cfg.mse_weight == cfg.lpips_weight == cfg.gan_weight == 0:
        raise ValueError("every loss weight is 0, there is nothing to train on")
    if cfg.best_by == "rfid" and not cfg.rfid:
        raise ValueError("--best-by rfid needs rFID: drop --no-rfid, or pick another --best-by")
    resolutions = sorted({cfg.resolution, *cfg.eval_resolutions})

    # model: built from --model in phase 1, otherwise loaded from the previous phase
    init_path = None
    if isinstance(cfg, Phase1Config):
        model_cfg = cfg.model
        model = build_titok(model_cfg)
    else:
        init_path = Path(cfg.init) if cfg.init else latest_checkpoint(PROJECT, "best.pt", stage=f"titok-phase{phase - 1}")
        model, model_cfg, ckpt = load_titok(init_path)
        print(f"Starting from {init_path} (phase {ckpt['train_config'].get('phase', '?')}, epoch {ckpt['epoch']})")
    for r in resolutions:
        if r > SIZE or r % model_cfg.patch_size:
            raise ValueError(f"resolution {r} must be a multiple of patch size {model_cfg.patch_size} and at most {SIZE} (the cache size)")
    model = model.to(device)

    prefixes = set_trainable(model, cfg.train, model_cfg.depth, cfg.edge_blocks)
    n_all = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Phase {phase}: training {n_train / 1e6:.2f}M of {n_all / 1e6:.2f}M parameters ({', '.join(p or 'all' for p in prefixes)})")
    if n_train == 0:
        raise ValueError("--train selects no parameters")
    print(f"{model_cfg.num_tokens} tokens of {len(model_cfg.levels)} channels per image (codebook {model_cfg.codebook_size}); patch grid {cfg.resolution // model_cfg.patch_size}x{cfg.resolution // model_cfg.patch_size} at {cfg.resolution} px")

    if cfg.skip_duplicate and not cfg.dev:
        same = find_finished_run(PROJECT, cfg.to_dict(), stage=f"titok-phase{phase}", init=init_path, ignore=DUPLICATE_IGNORE)
        if same is not None:
            print(f"Skipping: {same} already ran this config (--no-skip-duplicate to train anyway)")
            return

    run_dir = new_run_dir(PROJECT, cfg.to_dict(), device, dev=cfg.dev, stage=f"titok-phase{phase}", name=cfg.name, init=str(init_path) if init_path else None)
    train_cfg = cfg.to_dict()

    train_ds, val_ds, test_ds = make_splits(cfg.val_size, cfg.seed)
    print(f"train {len(train_ds)}, val {len(val_ds)}, test {len(test_ds)} (cache {SIZE}x{SIZE}, training at {cfg.resolution}x{cfg.resolution})")
    print(f"Loading the train and validation images into device memory ({device})")
    train_dl = Batches(load_split(train_ds, device), cfg.batch_size, shuffle=True, drop_last=True, seed=cfg.seed)
    val_dl = Batches(load_split(val_ds, device), cfg.batch_size)

    summary(model, input_size=(cfg.batch_size, 3, cfg.resolution, cfg.resolution), device=device, depth=2)
    ema_model = deepcopy(model).eval() if cfg.ema else None  # before compile, so it shares no compiled state
    if cfg.compile:
        model.compile(mode="reduce-overhead", dynamic=False)  # in place, so state_dict keys and checkpoints are unchanged
    eval_model = ema_model if ema_model is not None else model  # the weights that are validated and saved

    lp = build_lpips(device, cfg.lpips_net)
    inception = Inception(device) if cfg.rfid else None
    use_gan = cfg.gan_weight > 0
    disc = PatchDiscriminator(cfg.disc_width).to(device) if use_gan else None
    d_opt = torch.optim.Adam(disc.parameters(), lr=cfg.disc_lr, betas=cfg.betas, fused=device.type == "cuda") if disc is not None else None

    steps_per_epoch = len(train_dl)
    params = [p for p in model.parameters() if p.requires_grad]
    opts = build_optimizers(model, cfg)
    scheds = [
        torch.optim.lr_scheduler.OneCycleLR(o, max_lr=cfg.lr, total_steps=cfg.epochs * steps_per_epoch, cycle_momentum=False)  # keep --betas as given
        if cfg.lr_decay
        else torch.optim.lr_scheduler.ConstantLR(o, factor=1.0, total_iters=0)
        for o in opts
    ]

    best_val, best_epoch = float("inf"), 0
    ref_val: dict[int, FeatureStats] = {}  # Inception statistics of the validation images, filled by the first evaluation
    stats = ParamStats(model, every=100)  # gradient / weight statistics of the trainable parameters
    gpu = GpuMonitor(device)
    global_step = 0
    with MetricsLogger(run_dir / "metrics.csv", metric_fields(resolutions)) as logger:
        rows = logger.rows

        if phase > 1:  # the init checkpoint before any training: what this phase starts from
            val = evaluate(eval_model, val_dl, device, resolutions, lp, inception, amp, ref_val)
            logger.log({"epoch": 0, **{f"val_{k}": v for k, v in val.items()}})
            print("epoch 0 (init): " + " ".join(f"{k}={v:.4f}" for k, v in val.items() if k.startswith(SHOWN)))

        for epoch in range(1, cfg.epochs + 1):
            tp = Throughput()  # rolling steps/s window
            lr = scheds[0].get_last_lr()[0]  # every optimizer follows the same schedule
            model.train()
            sums: dict[str, torch.Tensor] = {}
            seen = 0
            for step, x in enumerate(train_dl, 1):
                n = len(x)
                if cfg.hflip_prob > 0:
                    flip = torch.rand(n, device=device) < cfg.hflip_prob
                    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
                x = resize(x, cfg.resolution)
                with amp():
                    recon = model(x)
                recon = recon.float()  # the losses and the discriminator input are fp32

                l1 = F.l1_loss(recon if cfg.l1_weight else recon.detach(), x)  # a term with weight 0 is only logged, so backward skips it
                mse = F.mse_loss(recon if cfg.mse_weight else recon.detach(), x)
                loss = cfg.l1_weight * l1 + cfg.mse_weight * mse
                if cfg.lpips_weight > 0:
                    with amp():
                        lpips_loss = lp(recon, x).float().mean()
                    loss = loss + cfg.lpips_weight * lpips_loss
                    add_terms(sums, n, lpips=lpips_loss)
                if disc is not None:
                    disc.requires_grad_(False)  # the generator step must not build discriminator gradients
                    with amp():
                        g_gan = -disc(recon).float().mean()
                    loss = loss + cfg.gan_weight * g_gan
                    add_terms(sums, n, gan=g_gan)
                for o in opts:
                    o.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip if cfg.grad_clip > 0 else float("inf"))
                global_step += 1
                stats.grads(global_step, gn, cfg.grad_clip)
                for o in opts:
                    o.step()
                stats.stepped(global_step)
                for s in scheds:
                    s.step()
                if ema_model is not None:
                    update_ema(ema_model, model, cfg.ema_decay)

                if disc is not None and d_opt is not None:
                    disc.requires_grad_(True)
                    with amp():
                        d_real, d_fake = disc(x).float(), disc(recon.detach()).float()
                    d_loss = F.relu(1 - d_real).mean() + F.relu(1 + d_fake).mean()  # hinge
                    d_opt.zero_grad(set_to_none=True)
                    d_loss.backward()
                    d_opt.step()
                    add_terms(sums, n, d_loss=d_loss)

                add_terms(sums, n, loss=loss, l1=l1, mse=mse, psnr=psnr(recon.detach().clamp(-1, 1), x).mean())
                seen += n
                if step % cfg.log_every == 0:
                    mean = {k: v.item() / seen for k, v in sums.items()}  # syncs the device, so the timing below is real
                    steps_per_s_now = tp.tick(step)
                    extra = "".join(f" {k}={mean[k]:.4f}" for k in ("lpips", "gan", "d_loss") if k in mean)
                    print(f"epoch {epoch}/{cfg.epochs} step {step}/{steps_per_epoch}: loss={mean['loss']:.4f} l1={mean['l1']:.4f}{extra} psnr={mean['psnr']:.2f}dB grad_norm={gn.item():.3f} ({steps_per_s_now:.1f} steps/s)")

            mean = {k: v.item() / seen for k, v in sums.items()}  # sync the device so the timing below covers all training work
            steps_per_s = tp.rate(steps_per_epoch)
            train_stats, gpu_stats = stats.read()[0], gpu.read()  # read before validation, so both cover training only
            val = evaluate(eval_model, val_dl, device, resolutions, lp, inception, amp, ref_val)
            row = {
                "epoch": epoch,
                "lr": lr,
                **{f"train_{k}": mean.get(k) for k in ("loss", "l1", "mse", "lpips", "psnr", "gan", "d_loss")},
                **{f"val_{k}": v for k, v in val.items()},
                **train_stats,
                **gpu_stats,
                "steps_per_s": steps_per_s,
                "epoch_time_s": tp.elapsed(),
            }
            logger.log(row)
            shown = " ".join(f"{k}={v:.4f}" for k, v in val.items() if k.startswith(SHOWN))
            print(f"epoch {epoch}: train_loss={mean['loss']:.4f} train_psnr={mean['psnr']:.2f}dB val {shown} ({row['steps_per_s']:.1f} steps/s, {row['epoch_time_s']:.1f}s)")

            score = -epoch if cfg.best_by == "last" else val[f"{cfg.best_by}_{cfg.resolution}"]
            if score < best_val:
                best_val, best_epoch = score, epoch
                save_titok(run_dir / "best.pt", eval_model, model_cfg, train_cfg, epoch, row)

    save_titok(run_dir / "last.pt", eval_model, model_cfg, train_cfg, cfg.epochs, rows[-1])
    loss_series = ["train_loss", "train_l1"] + [f"train_{k}" for k, w in (("mse", cfg.mse_weight), ("lpips", cfg.lpips_weight)) if w > 0]
    plot_curves(rows, loss_series, run_dir / "loss.png", ylabel="loss", log=True)
    if cfg.rfid:
        plot_curves(rows, [f"val_rfid_{r}" for r in resolutions], run_dir / "rfid.png", ylabel="validation rFID", log=True)
    plot_curves(rows, [f"val_psnr_{r}" for r in resolutions], run_dir / "psnr.png", ylabel="validation PSNR (dB)")
    plot_curves(rows, [f"val_ssim_{r}" for r in resolutions], run_dir / "ssim.png", ylabel="validation SSIM")
    plot_curves(rows, [f"val_lpips_{r}" for r in resolutions], run_dir / "lpips.png", ylabel="validation LPIPS")
    if use_gan:
        plot_curves(rows, ["train_gan", "train_d_loss"], run_dir / "gan.png", ylabel="adversarial loss")

    best_model, _, _ = load_titok(run_dir / "best.pt")
    best_model = best_model.to(device)
    test_dl = Batches(load_split(test_ds, device), cfg.batch_size)
    test = evaluate(best_model, test_dl, device, resolutions, lp, inception, amp)
    save_recon(best_model, test_ds, device, run_dir / "recon.png", cfg.resolution, amp=amp)
    (run_dir / "results.json").write_text(
        json.dumps({"phase": phase, "best_by": cfg.best_by, "best_epoch": best_epoch, **{f"test_{k}": v for k, v in test.items()}}, indent=2)
    )
    shown = " ".join(f"{k}={v:.4f}" for k, v in test.items() if k.startswith(SHOWN))
    print(f"best epoch {best_epoch} ({cfg.best_by}): test {shown}")
    print(f"Outputs in {run_dir}")


if __name__ == "__main__":
    main(tyro.extras.subcommand_cli_from_dict({"phase1": Phase1Config, "phase2": Phase2Config, "phase3": Phase3Config}, description=__doc__))
