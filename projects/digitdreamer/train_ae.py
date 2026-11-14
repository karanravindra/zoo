"""Stage 1: train the autoencoder.

Run from anywhere: `uv run projects/digitdreamer/train_ae.py`

Each run gets its own directory, `<repo>/runs/digitdreamer/ae/<timestamp>/`, containing:
  config.json    the full config plus git commit and device
  metrics.csv    one row per epoch
  best.pt        checkpoint with the lowest validation MSE (includes its config)
  last.pt        checkpoint from the final epoch
  results.json   test-set MSE, PSNR and SSIM of best.pt
  curves.png     loss curves
  psnr.png, ssim.png  validation PSNR / SSIM per epoch
  recon.png      test images (top rows) and their reconstructions (bottom rows)
"""

import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import tyro
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import make_splits, new_run_dir, plot_curves, psnr, ssim, to_unit  # noqa: E402
from dreamer import IMAGE_SIZE, AEConfig, build_ae, load_ae, save_ae  # noqa: E402

from zoo import data  # noqa: E402
from zoo.utils.device import get_device, seed_everything  # noqa: E402

METRIC_FIELDS = ["epoch", "lr", "train_mse", "val_mse", "val_mse_global", "val_psnr", "val_ssim", "steps_per_s", "epoch_time_s"]


@torch.no_grad()
def evaluate(model, dl, device) -> dict[str, float]:
    """MSE (per pixel), and PSNR / SSIM (mean over images) of clamped reconstructions."""
    model.eval()
    se = torch.zeros((), device=device)
    se_global = torch.zeros((), device=device)  # detail channels masked: how much the global half alone captures
    psnr_sum = torch.zeros((), device=device)
    ssim_sum = torch.zeros((), device=device)
    n_img = n_px = 0
    for x, _ in dl:
        x = x.to(device, non_blocking=True)
        # Finish with each output before the next model call: with --compile (CUDA graphs) a call overwrites the
        # previous call's output buffer.
        recon = model(x)
        se += F.mse_loss(recon, x, reduction="sum")
        recon = recon.clamp(-1, 1)
        psnr_sum += psnr(recon, x).sum()
        ssim_sum += ssim(recon, x).sum()
        se_global += F.mse_loss(model(x, 1.0), x, reduction="sum")
        n_img += len(x)
        n_px += x.numel()
    return {"mse": se.item() / n_px, "mse_global": se_global.item() / n_px, "psnr": psnr_sum.item() / n_img, "ssim": ssim_sum.item() / n_img}


@torch.no_grad()
def save_recon(model, ds, device, path: Path, n: int = 16) -> None:
    model.eval()
    x = torch.stack([ds[i][0] for i in range(n)]).to(device)
    save_image(to_unit(torch.cat([x, model(x).clamp(-1, 1)])), path, nrow=n, padding=1)


def main(cfg: AEConfig):
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    device = get_device()
    print(f"Using device: {device}")
    run_dir = new_run_dir("ae", cfg.to_dict(), device)

    train_ds, val_ds, test_ds = make_splits(cfg.dataset, cfg.val_size, cfg.seed)
    loader_kw = dict(
        num_workers=cfg.num_workers, persistent_workers=cfg.num_workers > 0, pin_memory=device.type == "cuda"
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(cfg.seed),
        **loader_kw,
    )
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, **loader_kw)
    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=loader_kw["pin_memory"])

    model = build_ae(cfg).to(device)
    c = getattr(data, cfg.dataset).image_shape[0]
    h = w = IMAGE_SIZE
    summary(model, input_size=(cfg.batch_size, c, h, w), device=device, depth=2)
    if cfg.compile:
        model.compile(mode="reduce-overhead", dynamic=False)  # in place, so state_dict keys and checkpoints are unchanged

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.epochs * len(train_dl))

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
            loss_sum = torch.zeros((), device=device)
            seen = 0
            for step, (x, _) in enumerate(train_dl, 1):
                x = x.to(device, non_blocking=True)
                loss = F.mse_loss(model(x, cfg.mask_prob), x)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()

                loss_sum += loss.detach() * len(x)
                seen += len(x)
                if step % cfg.log_every == 0:
                    mean_loss = loss_sum.item() / seen  # syncs the device, so the timing below is real
                    now = time.perf_counter()
                    steps_per_s_now = (step - n_log) / (now - t_log)
                    t_log, n_log = now, step
                    print(f"epoch {epoch}/{cfg.epochs} step {step}/{len(train_dl)}: train_mse={mean_loss:.5f} ({steps_per_s_now:.1f} steps/s)")

            loss_sum.item()  # sync the device so the timing below covers all training work
            steps_per_s = len(train_dl) / (time.perf_counter() - t0)
            val = evaluate(model, val_dl, device)
            val_mse = val["mse"]
            row = {
                "epoch": epoch,
                "lr": lr,
                "train_mse": loss_sum.item() / seen,
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
                f"epoch {epoch}: train_mse={row['train_mse']:.5f} val_mse={val_mse:.5f} val_mse_global={val['mse_global']:.5f} "
                f"val_psnr={val['psnr']:.2f}dB val_ssim={val['ssim']:.4f} ({row['steps_per_s']:.1f} steps/s, {row['epoch_time_s']:.1f}s)"
            )

            if val_mse < best_val:
                best_val, best_epoch = val_mse, epoch
                save_ae(run_dir / "best.pt", model, cfg, epoch, row)

    save_ae(run_dir / "last.pt", model, cfg, cfg.epochs, rows[-1])
    plot_curves(rows, ["train_mse", "val_mse", "val_mse_global"], run_dir / "curves.png", ylabel="MSE", log=True)
    plot_curves(rows, ["val_psnr"], run_dir / "psnr.png", ylabel="PSNR (dB)")
    plot_curves(rows, ["val_ssim"], run_dir / "ssim.png", ylabel="SSIM")

    best_model, _, _ = load_ae(run_dir / "best.pt")
    best_model = best_model.to(device)
    test = evaluate(best_model, test_dl, device)
    save_recon(best_model, test_ds, device, run_dir / "recon.png")
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
    main(tyro.cli(AEConfig, description=__doc__))
