"""Stage 2: train a class-conditioned rectified flow in the frozen autoencoder's latent space.

Run from anywhere (after train_ae.py): `uv run projects/digitdreamer/train_flow.py`
Uses `FlowConfig.ae_ckpt` (default: the latent=1 AE run); if that is None, best.pt of the latest AE run.

The AE is deterministic, so every training image is encoded once up front; the flow then
trains on cached latents (per-channel standardized so they match the unit-Gaussian source).

Each run gets its own directory, `<repo>/runs/digitdreamer/flow/<timestamp>/`, containing:
  config.json    the full config plus git commit, device, and the AE checkpoint used
  metrics.csv    one row per epoch
  samples/       a class-by-sample grid after every epoch (rows = digits 0-9), from the EMA model
  best.pt        self-contained checkpoint (EMA flow + AE + latent stats) with the lowest validation loss
  last.pt        same, from the final epoch; load either with `load_dreamer`
  results.json   best epoch, validation and test loss of best.pt
  curves.png     loss curves
  samples.png    final sample grid
"""

import json
from copy import deepcopy
from pathlib import Path

import torch
import tyro
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision.utils import save_image

from common import make_splits
from dreamer import (
    LATENT_SIZE,
    Dreamer,
    FlowConfig,
    build_flow,
    load_ae,
    load_dreamer,
    rectified_flow_loss,
    save_flow,
)

from zoo import data
from zoo.metrics import to_unit
from zoo.monitor import GPU_FIELDS, GpuMonitor
from zoo.paths import REPO_ROOT
from zoo.plot import plot_curves
from zoo.runs import MetricsLogger, latest_checkpoint, new_run_dir
from zoo.train import STAT_FIELDS, ParamStats, Throughput, update_ema
from zoo.utils.device import get_device, seed_everything

METRIC_FIELDS = ["epoch", "lr", "train_loss", "val_loss", *STAT_FIELDS, *GPU_FIELDS, "steps_per_s", "epoch_time_s"]
SAMPLES_PER_CLASS = 10


@torch.no_grad()
def encode_split(ae, ds, device, num_workers: int, batch_size: int = 1024) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a whole dataset to (latents, labels), kept on the device."""
    ae.eval()
    zs, ys = [], []
    for x, y in DataLoader(ds, batch_size=batch_size, num_workers=num_workers):
        zs.append(ae.encode(x.to(device)))
        ys.append(y)
    return torch.cat(zs), torch.cat(ys).to(device)


@torch.no_grad()
def eval_loss(model, z, y, batch_size: int, seed: int = 0) -> float:
    """Flow loss on held-out latents with fixed noise and times, so it is comparable across epochs."""
    model.eval()
    g = torch.Generator(device=z.device).manual_seed(seed)
    total, n = 0.0, 0
    for i in range(0, len(z), batch_size):
        zb, yb = z[i : i + batch_size], y[i : i + batch_size]
        x0 = torch.randn(zb.shape, device=z.device, generator=g)
        t = torch.rand(len(zb), device=z.device, generator=g)
        total += rectified_flow_loss(model, zb, yb, x0, t).item() * len(zb)
        n += len(zb)
    return total / n


def save_samples(dreamer: Dreamer, num_classes: int, device, path: Path) -> None:
    """Rows are classes, columns are samples; the noise is fixed so epochs are comparable."""
    y = torch.arange(num_classes, device=device).repeat_interleave(SAMPLES_PER_CLASS)
    g = torch.Generator(device=device).manual_seed(1234)
    shape = (len(y), dreamer.ae_cfg.latent_channels, LATENT_SIZE, LATENT_SIZE)
    noise = torch.randn(shape, device=device, generator=g)
    imgs = dreamer.generate(y, noise=noise)
    save_image(to_unit(imgs), path, nrow=SAMPLES_PER_CLASS, padding=1)


def main(cfg: FlowConfig):
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    device = get_device()
    print(f"Using device: {device}")

    ae_path = REPO_ROOT / cfg.ae_ckpt if cfg.ae_ckpt else latest_checkpoint("digitdreamer", "best.pt", stage="ae")
    ae, ae_cfg, _ = load_ae(ae_path)
    ae = ae.to(device).eval().requires_grad_(False)
    print(f"Autoencoder: {ae_path}")
    run_dir = new_run_dir("digitdreamer", cfg.to_dict(), device, dev=cfg.dev, stage="flow", ae_ckpt=str(ae_path))
    (run_dir / "samples").mkdir()

    # Encode with the same train/val split the AE saw, so the flow never trains on AE-val images.
    train_ds, val_ds, test_ds = make_splits(ae_cfg.dataset, ae_cfg.val_size, ae_cfg.seed)
    z_train, y_train = encode_split(ae, train_ds, device, cfg.num_workers)
    z_val, y_val = encode_split(ae, val_ds, device, cfg.num_workers)
    z_test, y_test = encode_split(ae, test_ds, device, cfg.num_workers)
    mean = z_train.mean(dim=(0, 2, 3), keepdim=True)
    std = z_train.std(dim=(0, 2, 3), keepdim=True)
    z_train, z_val, z_test = (z_train - mean) / std, (z_val - mean) / std, (z_test - mean) / std
    print(f"Latents: train {tuple(z_train.shape)}, val {tuple(z_val.shape)}")

    num_classes = getattr(data, ae_cfg.dataset).num_classes
    model = build_flow(cfg, ae_cfg).to(device)
    summary(
        model,
        input_data=(z_train[: cfg.batch_size], torch.rand(cfg.batch_size, device=device), y_train[: cfg.batch_size]),
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
    # Dreamer wraps the EMA model, so previews and the final checkpoint use the EMA weights.
    dreamer = Dreamer(ema, ae, mean, std, cfg, ae_cfg).to(device)

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
            perm = torch.randperm(len(z_train), device=device, generator=gen)
            loss_sum = torch.zeros((), device=device)
            seen = 0
            for step in range(steps_per_epoch):
                idx = perm[step * cfg.batch_size : (step + 1) * cfg.batch_size]
                z, y = z_train[idx], y_train[idx]
                # Randomly drop the label to the null class so one model learns both p(z|y) and p(z).
                drop = torch.rand(len(y), device=device, generator=gen) < cfg.class_dropout
                y = torch.where(drop, torch.full_like(y, num_classes), y)

                loss = rectified_flow_loss(model, z, y, torch.randn(z.shape, device=device, generator=gen))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                global_step += 1
                stats.grads(global_step)
                opt.step()
                stats.stepped(global_step)
                sched.step()
                update_ema(ema, model, cfg.ema_decay)

                loss_sum += loss.detach() * len(y)
                seen += len(y)
                if (step + 1) % cfg.log_every == 0:
                    mean_loss = loss_sum.item() / seen  # syncs the device, so the timing below is real
                    steps_per_s_now = tp.tick(step + 1)
                    print(f"epoch {epoch}/{cfg.epochs} step {step + 1}/{steps_per_epoch}: train_loss={mean_loss:.4f} ({steps_per_s_now:.1f} steps/s)")

            loss_sum.item()  # sync the device so the timing below covers all training work
            steps_per_s = tp.rate(steps_per_epoch)
            train_stats, gpu_stats = stats.read()[0], gpu.read()  # read before validation, so both cover training only
            val_loss = eval_loss(ema, z_val, y_val, cfg.batch_size)
            row = {
                "epoch": epoch,
                "lr": lr,
                "train_loss": loss_sum.item() / seen,
                "val_loss": val_loss,
                **train_stats,
                **gpu_stats,
                "steps_per_s": steps_per_s,
                "epoch_time_s": tp.elapsed(),
            }
            logger.log(row)
            print(f"epoch {epoch}: train_loss={row['train_loss']:.4f} val_loss={val_loss:.4f} ({row['steps_per_s']:.1f} steps/s, {row['epoch_time_s']:.1f}s)")
            save_samples(dreamer, num_classes, device, run_dir / "samples" / f"epoch_{epoch:03d}.png")
            if val_loss < best_val:
                best_val, best_epoch = val_loss, epoch
                save_flow(run_dir / "best.pt", ema, cfg, ae, ae_cfg, mean, std, epoch, row)

    save_flow(run_dir / "last.pt", ema, cfg, ae, ae_cfg, mean, std, cfg.epochs, rows[-1])
    plot_curves(rows, ["train_loss", "val_loss"], run_dir / "curves.png", ylabel="velocity MSE")
    save_samples(dreamer, num_classes, device, run_dir / "samples.png")
    test_loss = eval_loss(load_dreamer(run_dir / "best.pt").flow.to(device), z_test, y_test, cfg.batch_size)
    (run_dir / "results.json").write_text(
        json.dumps({"best_epoch": best_epoch, "best_val_loss": best_val, "test_loss": test_loss}, indent=2)
    )
    print(f"best epoch {best_epoch}: val_loss={best_val:.4f} test_loss={test_loss:.4f}")
    print(f"Outputs in {run_dir}")


if __name__ == "__main__":
    main(tyro.cli(FlowConfig, description=__doc__))
