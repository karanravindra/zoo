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

import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchinfo import summary
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from digitnet import Config, build_model, load_checkpoint, save_checkpoint  # noqa: E402
from zoo import data  # noqa: E402
from zoo.utils.device import get_device, seed_everything  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
RUNS_ROOT = REPO_ROOT / "runs" / "digitnet"

METRIC_FIELDS = ["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "steps_per_s", "epoch_time_s"]


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


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True)
        return out + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return None


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


def main(cfg: Config = Config()):
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    device = get_device()
    print(f"Using device: {device}")

    run_dir = RUNS_ROOT / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(
        json.dumps({"config": cfg.to_dict(), "git_commit": git_commit(), "device": str(device)}, indent=2)
    )
    print(f"Run directory: {run_dir}")

    train_dl, val_dl, test_dl = make_loaders(cfg, device)

    model = build_model(cfg).to(device)
    c, h, w = getattr(data, cfg.dataset).image_shape
    summary(model, input_size=(cfg.batch_size, c, h, w), device=device, depth=2)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.epochs * len(train_dl))

    rows: list[dict] = []
    best_val_acc, best_epoch = -1.0, 0
    with open(run_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        writer.writeheader()

        for epoch in range(1, cfg.epochs + 1):
            t0 = time.perf_counter()
            t_log, n_log = t0, 0  # window for the rolling steps/s
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
                opt.step()
                sched.step()

                loss_sum += loss.detach() * len(y)
                correct += (logits.detach().argmax(1) == y).sum()
                seen += len(y)
                if step % cfg.log_every == 0:
                    mean_loss = loss_sum.item() / seen  # syncs the device, so the timing below is real
                    now = time.perf_counter()
                    steps_per_s_now = (step - n_log) / (now - t_log)
                    t_log, n_log = now, step
                    pbar.set_postfix(loss=f"{mean_loss:.4f}", steps_per_s=f"{steps_per_s_now:.1f}")

            loss_sum.item()  # sync the device so the timing below covers all training work
            steps_per_s = len(train_dl) / (time.perf_counter() - t0)
            val_loss, val_acc = evaluate(model, val_dl, device)
            row = {
                "epoch": epoch,
                "lr": lr,
                "train_loss": loss_sum.item() / seen,
                "train_acc": correct.item() / seen,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "steps_per_s": steps_per_s,
                "epoch_time_s": time.perf_counter() - t0,
            }
            rows.append(row)
            writer.writerow(row)
            f.flush()
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
    main()
