"""Train DigitNet (a small CNN) on MNIST.

Run from anywhere: `uv run projects/digitnet/train.py`
Outputs (checkpoint + training curves) go to `<repo>/runs/digitnet/`.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

from model import DigitNet
from zoo.data import MNIST, QMNIST  # noqa: F401
from zoo.utils.device import get_device, seed_everything

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
RUN_DIR = REPO_ROOT / "runs" / "digitnet"

DATASET = MNIST
BATCH_SIZE = 512
EPOCHS = 10
LR = 4e-3
WEIGHT_DECAY = 5e-4
WIDTH = 32


@torch.no_grad()
def evaluate(model, dl, device) -> tuple[float, float]:
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


def plot_history(history: dict, path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
    ax_loss.plot(epochs, history["train_loss"], label="train")
    ax_loss.plot(epochs, history["test_loss"], label="test")
    ax_loss.set(xlabel="epoch", ylabel="loss", title="Loss")
    ax_loss.legend()
    ax_acc.plot(epochs, history["test_acc"])
    ax_acc.set(xlabel="epoch", ylabel="accuracy", title="Test accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    seed_everything(0)
    device = get_device()
    print(f"Using device: {device}")
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    train_ds = DATASET(root=str(DATA_ROOT), train=True)
    test_ds = DATASET(root=str(DATA_ROOT), train=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

    c, h, w = DATASET.image_shape
    model = DigitNet(in_channels=c, num_classes=DATASET.num_classes, width=WIDTH).to(device)
    summary(model, input_size=(BATCH_SIZE, c, h, w), device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * len(train_dl))

    history = {"train_loss": [], "test_loss": [], "test_acc": []}
    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        running, steps = 0.0, 0
        pbar = tqdm(train_dl, desc=f"epoch {epoch + 1}/{EPOCHS}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item()
            steps += 1
            pbar.set_postfix(loss=f"{running / steps:.4f}")

        test_loss, test_acc = evaluate(model, test_dl, device)
        history["train_loss"].append(running / steps)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        print(f"epoch {epoch + 1}: train_loss={running / steps:.4f} test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), RUN_DIR / "best.pt")
        plot_history(history, RUN_DIR / "curves.png")

    print(f"best test_acc={best_acc:.4f} — checkpoint and curves in {RUN_DIR}")


if __name__ == "__main__":
    main()
