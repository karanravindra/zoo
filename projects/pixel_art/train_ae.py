"""Stage 1: train the autoencoder on RGBA sprites.

Run from anywhere (after building the cache with pixel_data.py): `uv run projects/pixel_art/train_ae.py`

Each run gets its own directory, `<repo>/runs/pixel_art/ae/<timestamp>/`, containing:
  config.json    the full config plus git commit and device
  metrics.csv    one row per epoch
  best.pt        checkpoint with the lowest validation MSE (includes its config)
  last.pt        checkpoint from the final epoch
  results.json   test-set (official validation split) MSE, PSNR and SSIM of best.pt
  curves.png     loss curves
  psnr.png, ssim.png  train and validation PSNR / validation SSIM per epoch
  recon.png      test sprites (top row) and their reconstructions (bottom row), on a checkerboard
"""

import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchinfo import summary

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import new_run_dir, plot_curves, psnr, save_grid, ssim  # noqa: E402
from ldm import AEConfig, build_ae, load_ae, save_ae  # noqa: E402
from pixel_data import CHANNELS, IMAGE_SIZE, load_data, to_pm1  # noqa: E402
from zoo.utils.device import get_device, seed_everything  # noqa: E402

METRIC_FIELDS = ["epoch", "lr", "train_mse", "train_psnr", "val_mse", "val_mse_global", "val_psnr", "val_ssim", "steps_per_s", "epoch_time_s"]


@torch.no_grad()
def evaluate(model, images: torch.Tensor, idx: torch.Tensor | None, batch_size: int) -> dict[str, float]:
    """MSE (per value), and PSNR / SSIM (mean over images) of clamped reconstructions.

    `idx` selects rows of `images` (None = all). `mse_global` masks the detail channels."""
    model.eval()
    n = len(images) if idx is None else len(idx)
    dev = images.device
    se, se_global, psnr_sum, ssim_sum = (torch.zeros((), device=dev) for _ in range(4))
    for i in range(0, n, batch_size):
        sel = slice(i, i + batch_size) if idx is None else idx[i : i + batch_size]
        x = to_pm1(images[sel])
        recon = model(x)
        se += F.mse_loss(recon, x, reduction="sum")
        se_global += F.mse_loss(model(x, 1.0), x, reduction="sum")
        recon = recon.clamp(-1, 1)
        psnr_sum += psnr(recon, x).sum()
        ssim_sum += ssim(recon, x).sum()
    return {"mse": se.item() / (n * CHANNELS * IMAGE_SIZE**2), "mse_global": se_global.item() / (n * CHANNELS * IMAGE_SIZE**2), "psnr": psnr_sum.item() / n, "ssim": ssim_sum.item() / n}


@torch.no_grad()
def save_recon(model, images: torch.Tensor, path: Path, n: int = 16) -> None:
    model.eval()
    x = to_pm1(images[:n])
    save_grid(torch.cat([x, model(x).clamp(-1, 1)]), path, nrow=n)


def main(cfg: AEConfig = AEConfig()):
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    device = get_device()
    print(f"Using device: {device}")
    run_dir = new_run_dir("ae", cfg.to_dict(), device)

    d = load_data(cfg.val_size, cfg.seed).to(device)  # the whole dataset lives on the device as uint8
    print(f"train {len(d.train_idx)}, val {len(d.val_idx)}, test {len(d.test_images)}")

    model = build_ae(cfg).to(device)
    summary(model, input_size=(cfg.batch_size, CHANNELS, IMAGE_SIZE, IMAGE_SIZE), device=device, depth=2)

    steps_per_epoch = len(d.train_idx) // cfg.batch_size
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.epochs * steps_per_epoch)
    gen = torch.Generator(device=device).manual_seed(cfg.seed)

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
            perm = d.train_idx[torch.randperm(len(d.train_idx), device=device, generator=gen)]
            loss_sum = torch.zeros((), device=device)
            psnr_sum = torch.zeros((), device=device)
            seen = 0
            for step in range(steps_per_epoch):
                x = to_pm1(d.images[perm[step * cfg.batch_size : (step + 1) * cfg.batch_size]])
                recon = model(x, cfg.mask_prob)
                loss = F.mse_loss(recon, x)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()

                loss_sum += loss.detach() * len(x)
                psnr_sum += psnr(recon.detach().clamp(-1, 1), x).sum()
                seen += len(x)
                if (step + 1) % cfg.log_every == 0:
                    mean_loss = loss_sum.item() / seen  # syncs the device, so the timing below is real
                    now = time.perf_counter()
                    steps_per_s_now = (step + 1 - n_log) / (now - t_log)
                    t_log, n_log = now, step + 1
                    print(f"epoch {epoch}/{cfg.epochs} step {step + 1}/{steps_per_epoch}: train_mse={mean_loss:.5f} train_psnr={psnr_sum.item() / seen:.2f}dB ({steps_per_s_now:.1f} steps/s)")

            loss_sum.item()  # sync the device so the timing below covers all training work
            steps_per_s = steps_per_epoch / (time.perf_counter() - t0)
            val = evaluate(model, d.images, d.val_idx, cfg.batch_size)
            val_mse = val["mse"]
            row = {
                "epoch": epoch,
                "lr": lr,
                "train_mse": loss_sum.item() / seen,
                "train_psnr": psnr_sum.item() / seen,
                "val_mse": val_mse,
                "val_mse_global": val["mse_global"],
                "val_psnr": val["psnr"],
                "val_ssim": val["ssim"],
                "steps_per_s": steps_per_s,
                "epoch_time_s": time.perf_counter() - t0,
            }
            rows.append(row)
            writer.writerow(row)
            f.flush()
            print(
                f"epoch {epoch}: train_mse={row['train_mse']:.5f} train_psnr={row['train_psnr']:.2f}dB val_mse={val_mse:.5f} val_mse_global={val['mse_global']:.5f} "
                f"val_psnr={val['psnr']:.2f}dB val_ssim={val['ssim']:.4f} ({row['steps_per_s']:.1f} steps/s, {row['epoch_time_s']:.1f}s)"
            )

            if val_mse < best_val:
                best_val, best_epoch = val_mse, epoch
                save_ae(run_dir / "best.pt", model, cfg, epoch, row)

    save_ae(run_dir / "last.pt", model, cfg, cfg.epochs, rows[-1])
    plot_curves(rows, ["train_mse", "val_mse", "val_mse_global"], run_dir / "curves.png", ylabel="MSE", log=True)
    plot_curves(rows, ["train_psnr", "val_psnr"], run_dir / "psnr.png", ylabel="PSNR (dB)")
    plot_curves(rows, ["val_ssim"], run_dir / "ssim.png", ylabel="SSIM")

    best_model, _, _ = load_ae(run_dir / "best.pt")
    best_model = best_model.to(device)
    test = evaluate(best_model, d.test_images, None, cfg.batch_size)
    save_recon(best_model, d.test_images, run_dir / "recon.png")
    (run_dir / "results.json").write_text(
        json.dumps(
            {"best_epoch": best_epoch, "best_val_mse": best_val, **{f"test_{k}": v for k, v in test.items()}}, indent=2
        )
    )
    print(
        f"best epoch {best_epoch}: val_mse={best_val:.5f} "
        f"test_mse={test['mse']:.5f} test_psnr={test['psnr']:.2f}dB test_ssim={test['ssim']:.4f}"
    )
    print(f"Outputs in {run_dir}")


if __name__ == "__main__":
    main()
