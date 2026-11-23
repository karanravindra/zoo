"""Train DigitNet (a small CNN) on MNIST.

Run from anywhere: `uv run projects/digitnet/train.py`

Each run gets its own directory, `<repo>/runs/digitnet/<timestamp>/`, containing:
  config.json    the full config plus git commit and device
  metrics.csv    one row per epoch (written as training goes)
  best.pt        checkpoint with the best validation accuracy (includes its config)
  last.pt        checkpoint from the final epoch
  results.json   final test-set score of best.pt
  curves.png     loss and accuracy curves
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
import tyro
from torch.utils.data import DataLoader, random_split
from torchinfo import summary
from tqdm import tqdm

from digitnet import Config, build_model, load_checkpoint, save_checkpoint

from zoo import data
from zoo.monitor import GPU_FIELDS, GpuMonitor
from zoo.paths import DATA_ROOT
from zoo.plot import plt
from zoo.runs import MetricsLogger, new_run_dir
from zoo.train import STAT_FIELDS, ParamStats, Throughput
from zoo.utils.device import get_device, seed_everything

METRIC_FIELDS = ["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", *STAT_FIELDS, *GPU_FIELDS, "steps_per_s", "epoch_time_s"]


@torch.no_grad()
def evaluate(model, dl, device) -> tuple[float, float]:
    model.eval()
    total_loss = torch.zeros((), device=device)
    correct = torch.zeros((), device=device, dtype=torch.long)
    n = 0
    for x, y in dl:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction="sum")
        correct += (logits.argmax(1) == y).sum()
        n += len(y)
    return total_loss.item() / n, correct.item() / n


def plot_curves(rows: list[dict], path: Path) -> None:
    epochs = [r["epoch"] for r in rows]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
    ax_loss.plot(epochs, [r["train_loss"] for r in rows], label="train")
    ax_loss.plot(epochs, [r["val_loss"] for r in rows], label="val")
    ax_loss.set(xlabel="epoch", ylabel="loss", title="Loss")
    ax_loss.legend()
    ax_acc.plot(epochs, [r["train_acc"] for r in rows], label="train")
    ax_acc.plot(epochs, [r["val_acc"] for r in rows], label="val")
    ax_acc.set(xlabel="epoch", ylabel="accuracy", title="Accuracy")
    ax_acc.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def make_loaders(cfg: Config, device: torch.device):
    ds_cls = getattr(data, cfg.dataset)
    full_train = ds_cls(root=str(DATA_ROOT), train=True)
    test_ds = ds_cls(root=str(DATA_ROOT), train=False)
    train_ds, val_ds = random_split(
        full_train,
        [len(full_train) - cfg.val_size, cfg.val_size],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    loader_kw = dict(
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        pin_memory=device.type == "cuda",
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
    # Only used once at the end, so no persistent workers.
    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=loader_kw["pin_memory"])
    return train_dl, val_dl, test_dl


def main(cfg: Config):
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    device = get_device()
    print(f"Using device: {device}")

    run_dir = new_run_dir("digitnet", cfg.to_dict(), device, dev=cfg.dev)

    train_dl, val_dl, test_dl = make_loaders(cfg, device)

    model = build_model(cfg).to(device)
    c, h, w = getattr(data, cfg.dataset).image_shape
    summary(model, input_size=(cfg.batch_size, c, h, w), device=device, depth=2)
    if cfg.compile:
        model.compile(mode="reduce-overhead", dynamic=False)  # in place, so state_dict keys and checkpoints are unchanged

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.epochs * len(train_dl))

    best_val_acc, best_epoch = -1.0, 0
    stats = ParamStats(model, every=100)  # gradient / weight statistics, see zoo.train.ParamStats
    gpu = GpuMonitor(device)
    global_step = 0
    with MetricsLogger(run_dir / "metrics.csv", METRIC_FIELDS) as logger:
        rows = logger.rows

        for epoch in range(1, cfg.epochs + 1):
            tp = Throughput()  # rolling steps/s window
            lr = sched.get_last_lr()[0]
            model.train()
            # Accumulate on device; only sync with .item() every `log_every` steps.
            loss_sum = torch.zeros((), device=device)
            correct = torch.zeros((), device=device, dtype=torch.long)
            seen = 0
            pbar = tqdm(train_dl, desc=f"epoch {epoch}/{cfg.epochs}")
            for step, (x, y) in enumerate(pbar, 1):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                global_step += 1
                stats.grads(global_step)
                opt.step()
                stats.stepped(global_step)
                sched.step()

                loss_sum += loss.detach() * len(y)
                correct += (logits.detach().argmax(1) == y).sum()
                seen += len(y)
                if step % cfg.log_every == 0:
                    mean_loss = loss_sum.item() / seen  # syncs the device, so the timing below is real
                    steps_per_s_now = tp.tick(step)
                    pbar.set_postfix(loss=f"{mean_loss:.4f}", steps_per_s=f"{steps_per_s_now:.1f}")

            loss_sum.item()  # sync the device so the timing below covers all training work
            steps_per_s = tp.rate(len(train_dl))
            train_stats, gpu_stats = stats.read()[0], gpu.read()  # read before validation, so both cover training only
            val_loss, val_acc = evaluate(model, val_dl, device)
            row = {
                "epoch": epoch,
                "lr": lr,
                "train_loss": loss_sum.item() / seen,
                "train_acc": correct.item() / seen,
                "val_loss": val_loss,
                "val_acc": val_acc,
                **train_stats,
                **gpu_stats,
                "steps_per_s": steps_per_s,
                "epoch_time_s": tp.elapsed(),
            }
            logger.log(row)
            print(
                f"epoch {epoch}: train_loss={row['train_loss']:.4f} train_acc={row['train_acc']:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({row['steps_per_s']:.1f} steps/s, {row['epoch_time_s']:.1f}s)"
            )

            if val_acc > best_val_acc:
                best_val_acc, best_epoch = val_acc, epoch
                save_checkpoint(run_dir / "best.pt", model, cfg, epoch, row)

    save_checkpoint(run_dir / "last.pt", model, cfg, cfg.epochs, rows[-1])
    plot_curves(rows, run_dir / "curves.png")

    # Score the test set exactly once, with the checkpoint chosen on validation.
    best_model, _, _ = load_checkpoint(run_dir / "best.pt")
    test_loss, test_acc = evaluate(best_model.to(device), test_dl, device)
    results = {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"best epoch {best_epoch}: val_acc={best_val_acc:.4f} test_acc={test_acc:.4f}")
    print(f"Outputs in {run_dir}")


if __name__ == "__main__":
    main(tyro.cli(Config, description=__doc__))
