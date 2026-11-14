"""Stage 2: train the text-conditioned latent diffusion model (denoiser) in the frozen autoencoder's latent space.

Run from anywhere (after train_ae.py): `uv run projects/pixel_art/train_ldm.py`
Uses `LDMConfig.ae_ckpt`, or best.pt of the latest AE run if it is None.

The AE is deterministic, so every training sprite is encoded once up front; the denoiser (and its
text encoder) then trains on cached latents (per-channel standardized so they match the
unit-Gaussian source). Captions are randomly replaced by the empty prompt for classifier-free guidance.

Each run gets its own directory, `<repo>/runs/pixel_art/ldm/<timestamp>/`, containing:
  config.json    the full config plus git commit, device, and the AE checkpoint used
  metrics.csv    one row per epoch
  prompts.txt    the validation captions previewed each epoch (one per grid row)
  samples/       a grid after every epoch, from the EMA model: one row per prompt, first column the
                 real sprite for that caption, then 4 samples
  best.pt        self-contained checkpoint (EMA denoiser + AE + tokenizer + latent stats) with the lowest
                 validation loss
  last.pt        same, from the final epoch; load either with `load_ldm`
  results.json   best epoch, validation and test loss of best.pt
  curves.png     loss curves
  samples.png    final sample grid
"""

import csv
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch
import tyro
from torchinfo import summary

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO_ROOT, latest_checkpoint, new_run_dir, plot_curves, save_grid  # noqa: E402
from ldm import (  # noqa: E402
    LatentDiffusion,
    LDMConfig,
    build_denoiser,
    diffusion_loss,
    load_ae,
    load_ldm,
    save_ldm,
    uncond_tokens,
)
from pixel_data import load_data, load_tokenizer, to_pm1  # noqa: E402

from zoo.utils.device import get_device, seed_everything  # noqa: E402

METRIC_FIELDS = ["epoch", "lr", "train_loss", "val_loss", "steps_per_s", "epoch_time_s"]
SAMPLES_PER_PROMPT = 4


@torch.no_grad()
def encode_images(ae, images: torch.Tensor, idx: torch.Tensor | None, batch_size: int = 4096) -> torch.Tensor:
    """Encode (a subset of) a uint8 image tensor to latents, kept on the device."""
    ae.eval()
    n = len(images) if idx is None else len(idx)
    return torch.cat([ae.encode(to_pm1(images[idx[i : i + batch_size] if idx is not None else slice(i, i + batch_size)])) for i in range(0, n, batch_size)])


@torch.no_grad()
def eval_loss(model, z, tokens, batch_size: int, seed: int = 0) -> float:
    """Denoising loss on held-out latents with fixed noise and times, so it is comparable across epochs."""
    model.eval()
    g = torch.Generator(device=z.device).manual_seed(seed)
    total, n = 0.0, 0
    for i in range(0, len(z), batch_size):
        zb, tb = z[i : i + batch_size], tokens[i : i + batch_size].long()
        x0 = torch.randn(zb.shape, device=z.device, generator=g)
        t = torch.rand(len(zb), device=z.device, generator=g)
        total += diffusion_loss(model, zb, tb, x0, t).item() * len(zb)
        n += len(zb)
    return total / n


@torch.no_grad()
def update_ema(ema, model, decay: float) -> None:
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.lerp_(pm, 1 - decay)
    for be, bm in zip(ema.buffers(), model.buffers()):
        be.copy_(bm)


def save_samples(ldm: LatentDiffusion, real: torch.Tensor, tokens: torch.Tensor, path: Path) -> None:
    """One row per prompt: the real sprite, then samples. The noise is fixed so epochs are comparable."""
    device = ldm.device
    tok = tokens.long().repeat_interleave(SAMPLES_PER_PROMPT, dim=0)
    g = torch.Generator(device=device).manual_seed(1234)
    s = ldm.ae_cfg.latent_size
    noise = torch.randn(len(tok), ldm.ae_cfg.latent_channels, s, s, device=device, generator=g)
    imgs = ldm.generate(tok, noise=noise).view(len(real), SAMPLES_PER_PROMPT, *real.shape[1:])
    save_grid(torch.cat([real[:, None], imgs], dim=1).flatten(0, 1), path, nrow=SAMPLES_PER_PROMPT + 1)


def main(cfg: LDMConfig):
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    device = get_device()
    print(f"Using device: {device}")

    ae_path = REPO_ROOT / cfg.ae_ckpt if cfg.ae_ckpt else latest_checkpoint("ae", "best.pt")
    ae, ae_cfg, _ = load_ae(ae_path)
    ae = ae.to(device).eval().requires_grad_(False)
    print(f"Autoencoder: {ae_path}")
    run_dir = new_run_dir("ldm", cfg.to_dict(), device, ae_ckpt=str(ae_path))
    (run_dir / "samples").mkdir()

    # Same train/val split the AE saw, so the denoiser never trains on AE-val images.
    d = load_data(ae_cfg.val_size, ae_cfg.seed).to(device)
    tokenizer = load_tokenizer()
    z_train = encode_images(ae, d.images, d.train_idx)
    z_val = encode_images(ae, d.images, d.val_idx)
    z_test = encode_images(ae, d.test_images, None)
    tok_train, tok_val, tok_test = d.tokens[d.train_idx], d.tokens[d.val_idx], d.test_tokens
    mean = z_train.mean(dim=(0, 2, 3), keepdim=True)
    std = z_train.std(dim=(0, 2, 3), keepdim=True)
    z_train, z_val, z_test = (z_train - mean) / std, (z_val - mean) / std, (z_test - mean) / std
    print(f"Latents: train {tuple(z_train.shape)}, val {tuple(z_val.shape)}; vocab {len(tokenizer)} tokens")

    n_prev = cfg.preview_prompts
    prev_tokens, prev_real = d.test_tokens[:n_prev], to_pm1(d.test_images[:n_prev])
    (run_dir / "prompts.txt").write_text("\n".join(d.test_captions[:n_prev]) + "\n")

    model = build_denoiser(cfg, ae_cfg, tokenizer).to(device)
    summary(
        model,
        input_data=(z_train[: cfg.batch_size], torch.rand(cfg.batch_size, device=device), tok_train[: cfg.batch_size].long()),
        device=device,
        depth=2,
    )
    ema = deepcopy(model).eval().requires_grad_(False)
    if cfg.compile:
        model.compile(mode="reduce-overhead", dynamic=False)  # in place, so state_dict keys and checkpoints are unchanged

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = len(z_train) // cfg.batch_size
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.epochs * steps_per_epoch)
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    # LatentDiffusion wraps the EMA model, so previews and the final checkpoint use the EMA weights.
    ldm = LatentDiffusion(ema, ae, tokenizer, mean, std, cfg, ae_cfg).to(device)
    uncond = uncond_tokens(1, tokenizer.max_len, device)

    rows: list[dict] = []
    best_val, best_epoch = float("inf"), 0
    with open(run_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        writer.writeheader()

        for epoch in range(1, cfg.epochs + 1):
            t0 = time.perf_counter()
            t_log, n_log = t0, 0  # window for the rolling steps/s
            lr = sched.get_last_lr()[0]
            model.train()
            perm = torch.randperm(len(z_train), device=device, generator=gen)
            loss_sum = torch.zeros((), device=device)
            seen = 0
            for step in range(steps_per_epoch):
                idx = perm[step * cfg.batch_size : (step + 1) * cfg.batch_size]
                z, tok = z_train[idx], tok_train[idx].long()
                # Randomly drop the caption to the empty prompt so one model learns both p(z|text) and p(z).
                drop = torch.rand(len(idx), device=device, generator=gen) < cfg.uncond_prob
                tok = torch.where(drop[:, None], uncond, tok)

                loss = diffusion_loss(model, z, tok, torch.randn(z.shape, device=device, generator=gen))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                update_ema(ema, model, cfg.ema_decay)

                loss_sum += loss.detach() * len(idx)
                seen += len(idx)
                if (step + 1) % cfg.log_every == 0:
                    mean_loss = loss_sum.item() / seen  # syncs the device, so the timing below is real
                    now = time.perf_counter()
                    steps_per_s_now = (step + 1 - n_log) / (now - t_log)
                    t_log, n_log = now, step + 1
                    print(f"epoch {epoch}/{cfg.epochs} step {step + 1}/{steps_per_epoch}: train_loss={mean_loss:.4f} ({steps_per_s_now:.1f} steps/s)")

            loss_sum.item()  # sync the device so the timing below covers all training work
            steps_per_s = steps_per_epoch / (time.perf_counter() - t0)
            val_loss = eval_loss(ema, z_val, tok_val, cfg.batch_size)
            row = {
                "epoch": epoch,
                "lr": lr,
                "train_loss": loss_sum.item() / seen,
                "val_loss": val_loss,
                "steps_per_s": steps_per_s,
                "epoch_time_s": time.perf_counter() - t0,
            }
            rows.append(row)
            writer.writerow(row)
            f.flush()
            print(f"epoch {epoch}: train_loss={row['train_loss']:.4f} val_loss={val_loss:.4f} ({row['steps_per_s']:.1f} steps/s, {row['epoch_time_s']:.1f}s)")
            save_samples(ldm, prev_real, prev_tokens, run_dir / "samples" / f"epoch_{epoch:03d}.png")
            if val_loss < best_val:
                best_val, best_epoch = val_loss, epoch
                save_ldm(run_dir / "best.pt", ema, cfg, ae, ae_cfg, tokenizer, mean, std, epoch, row)

    save_ldm(run_dir / "last.pt", ema, cfg, ae, ae_cfg, tokenizer, mean, std, cfg.epochs, rows[-1])
    plot_curves(rows, ["train_loss", "val_loss"], run_dir / "curves.png", ylabel="denoising loss (velocity MSE)")
    save_samples(ldm, prev_real, prev_tokens, run_dir / "samples.png")
    test_loss = eval_loss(load_ldm(run_dir / "best.pt").denoiser.to(device), z_test, tok_test, cfg.batch_size)
    (run_dir / "results.json").write_text(
        json.dumps({"best_epoch": best_epoch, "best_val_loss": best_val, "test_loss": test_loss}, indent=2)
    )
    print(f"best epoch {best_epoch}: val_loss={best_val:.4f} test_loss={test_loss:.4f}")
    print(f"Outputs in {run_dir}")


if __name__ == "__main__":
    main(tyro.cli(LDMConfig, description=__doc__))
