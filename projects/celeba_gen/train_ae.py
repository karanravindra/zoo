"""Stage 1: train the autoencoder on CelebA-HQ at 256x256.

Run from anywhere: `uv run projects/celeba_gen/train_ae.py` (the first run builds the 256px cache, see celeba.py).

Each run gets its own directory, `<repo>/runs/celeba_gen/ae/<timestamp>/`, containing:
  config.json    the full config plus git commit and device
  metrics.csv    one row per epoch (train_mse and val_* are on decoded images in [-1, 1])
  best.pt        checkpoint with the lowest validation MSE (includes its config)
  last.pt        checkpoint from the final epoch
  results.json   test-set MSE, PSNR and SSIM of best.pt (test = the official CelebA-HQ val split)
  curves.png     MSE curves
  psnr.png, ssim.png  train and validation PSNR, validation SSIM per epoch
  recon.png      test images (top row) and their reconstructions (bottom row)
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
import tyro
from torch.utils.data import DataLoader, Dataset, random_split
from torchinfo import summary
from torchvision.utils import save_image

from autoencoder import AEConfig, build_ae, load_ae, save_ae
from celeba import SIZE, CelebAHQ256

from zoo.metrics import psnr, ssim, to_unit
from zoo.monitor import GPU_FIELDS, GpuMonitor
from zoo.paths import DATA_ROOT
from zoo.plot import plot_curves
from zoo.runs import MetricsLogger, new_run_dir
from zoo.train import STAT_FIELDS, ParamStats, Throughput
from zoo.utils.device import get_device, seed_everything

METRIC_FIELDS = ["epoch", "lr", "train_mse", "train_psnr", "val_mse", "val_psnr", "val_ssim", *STAT_FIELDS, *GPU_FIELDS, "steps_per_s", "epoch_time_s"]


def make_splits(val_size: int, seed: int) -> tuple[Dataset, Dataset, Dataset]:
    """(train, val, test): `val_size` images are carved out of the StarGAN train split; its val split is the test set."""
    full_train = CelebAHQ256(root=str(DATA_ROOT), train=True)
    test = CelebAHQ256(root=str(DATA_ROOT), train=False)
    train, val = random_split(full_train, [len(full_train) - val_size, val_size], generator=torch.Generator().manual_seed(seed))
    return train, val, test


@torch.no_grad()
def evaluate(model, dl, device) -> dict[str, float]:
    """MSE (per value), and PSNR / SSIM (mean over images) of reconstructions clamped to [-1, 1]."""
    model.eval()
    se = torch.zeros((), device=device)
    psnr_sum = torch.zeros((), device=device)
    ssim_sum = torch.zeros((), device=device)
    n_img = n_val = 0
    for x, _ in dl:
        x = x.to(device, non_blocking=True)
        # Finish with each output before the next model call: with --compile (CUDA graphs) a call overwrites the
        # previous call's output buffer.
        recon = model(x)
        se += F.mse_loss(recon, x, reduction="sum")
        recon = recon.clamp(-1, 1)
        psnr_sum += psnr(recon, x).sum()
        ssim_sum += ssim(recon, x).sum()
        n_img += len(x)
        n_val += x.numel()
    return {"mse": se.item() / n_val, "psnr": psnr_sum.item() / n_img, "ssim": ssim_sum.item() / n_img}


@torch.no_grad()
def save_recon(model, ds: Dataset, device, path: Path, n: int = 16) -> None:
    model.eval()
    x = torch.stack([ds[i][0] for i in range(n)]).to(device)
    save_image(to_unit(torch.cat([x, model(x).clamp(-1, 1)])), path, nrow=n, padding=1)


def main(cfg: AEConfig):
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    device = get_device()
    print(f"Using device: {device}")
    run_dir = new_run_dir("celeba_gen", cfg.to_dict(), device, dev=cfg.dev, stage="ae")

    train_ds, val_ds, test_ds = make_splits(cfg.val_size, cfg.seed)
    print(f"train {len(train_ds)}, val {len(val_ds)}, test {len(test_ds)} ({SIZE}x{SIZE})")
    loader_kw = dict(num_workers=cfg.num_workers, persistent_workers=cfg.num_workers > 0, pin_memory=device.type == "cuda")
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
    summary(model, input_size=(cfg.batch_size, 3, SIZE, SIZE), device=device, depth=2)
    if cfg.compile:
        model.compile(mode="reduce-overhead", dynamic=False)  # in place, so state_dict keys and checkpoints are unchanged

    steps_per_epoch = len(train_dl)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.epochs * steps_per_epoch)

    best_val, best_epoch = float("inf"), 0
    stats = ParamStats(model, every=100)  # gradient / weight statistics, see zoo.train.ParamStats
    gpu = GpuMonitor(device)
    global_step = 0
    with MetricsLogger(run_dir / "metrics.csv", METRIC_FIELDS) as logger:
        rows = logger.rows

        for epoch in range(1, cfg.epochs + 1):
            tp = Throughput()  # rolling steps/s window
            lr = sched.get_last_lr()[0]
            model.train()
            loss_sum = torch.zeros((), device=device)
            psnr_sum = torch.zeros((), device=device)
            gn_sum, gn_max, n_clip = (torch.zeros((), device=device) for _ in range(3))  # pre-clip global grad norm
            seen = 0
            for step, (x, _) in enumerate(train_dl, 1):
                x = x.to(device, non_blocking=True)
                if cfg.hflip_prob > 0:
                    flip = torch.rand(len(x), device=device) < cfg.hflip_prob
                    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
                recon = model(x)
                loss = F.mse_loss(recon, x)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip if cfg.grad_clip > 0 else float("inf"))
                gn_sum += gn
                gn_max = torch.maximum(gn_max, gn)
                n_clip += (gn > cfg.grad_clip) if cfg.grad_clip > 0 else 0
                global_step += 1
                stats.grads(global_step, gn, cfg.grad_clip)
                opt.step()
                stats.stepped(global_step)
                sched.step()

                loss_sum += loss.detach() * len(x)
                psnr_sum += psnr(recon.detach().clamp(-1, 1), x).sum()
                seen += len(x)
                if step % cfg.log_every == 0:
                    mean_loss = loss_sum.item() / seen  # syncs the device, so the timing below is real
                    steps_per_s_now = tp.tick(step)
                    print(f"epoch {epoch}/{cfg.epochs} step {step}/{steps_per_epoch}: train_mse={mean_loss:.5f} train_psnr={psnr_sum.item() / seen:.2f}dB grad_norm={gn_sum.item() / step:.3f} (max {gn_max.item():.2f}) ({steps_per_s_now:.1f} steps/s)")

            loss_sum.item()  # sync the device so the timing below covers all training work
            steps_per_s = tp.rate(steps_per_epoch)
            train_stats, gpu_stats = stats.read()[0], gpu.read()  # read before validation, so both cover training only
            val = evaluate(model, val_dl, device)
            row = {
                "epoch": epoch,
                "lr": lr,
                "train_mse": loss_sum.item() / seen,
                "train_psnr": psnr_sum.item() / seen,
                "val_mse": val["mse"],
                "val_psnr": val["psnr"],
                "val_ssim": val["ssim"],
                "grad_norm": gn_sum.item() / steps_per_epoch,
                "grad_norm_max": gn_max.item(),
                "clip_frac": n_clip.item() / steps_per_epoch,
                **train_stats,
                **gpu_stats,
                "steps_per_s": steps_per_s,
                "epoch_time_s": tp.elapsed(),
            }
            logger.log(row)
            print(
                f"epoch {epoch}: train_mse={row['train_mse']:.5f} train_psnr={row['train_psnr']:.2f}dB val_mse={val['mse']:.5f} "
                f"val_psnr={val['psnr']:.2f}dB val_ssim={val['ssim']:.4f} grad_norm={row['grad_norm']:.3f} (max {row['grad_norm_max']:.2f}, clipped {row['clip_frac']:.0%}) "
                f"({row['steps_per_s']:.1f} steps/s, {row['epoch_time_s']:.1f}s)"
            )

            if val["mse"] < best_val:
                best_val, best_epoch = val["mse"], epoch
                save_ae(run_dir / "best.pt", model, cfg, epoch, row)

    save_ae(run_dir / "last.pt", model, cfg, cfg.epochs, rows[-1])
    plot_curves(rows, ["train_mse", "val_mse"], run_dir / "curves.png", ylabel="MSE", log=True)
    plot_curves(rows, ["train_psnr", "val_psnr"], run_dir / "psnr.png", ylabel="PSNR (dB)")
    plot_curves(rows, ["val_ssim"], run_dir / "ssim.png", ylabel="SSIM")

    best_model, _, _ = load_ae(run_dir / "best.pt")
    best_model = best_model.to(device)
    test = evaluate(best_model, test_dl, device)
    save_recon(best_model, test_ds, device, run_dir / "recon.png")
    (run_dir / "results.json").write_text(
        json.dumps({"best_epoch": best_epoch, "best_val_mse": best_val, **{f"test_{k}": v for k, v in test.items()}}, indent=2)
    )
    print(f"best epoch {best_epoch}: val_mse={best_val:.5f} test_mse={test['mse']:.5f} test_psnr={test['psnr']:.2f}dB test_ssim={test['ssim']:.4f}")
    print(f"Outputs in {run_dir}")


if __name__ == "__main__":
    main(tyro.cli(AEConfig, description=__doc__))
