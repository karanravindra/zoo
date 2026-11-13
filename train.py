import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

from zoo.data import MNIST, QMNIST  # noqa: F401
from zoo.utils.device import get_device, seed_everything

DATASET = QMNIST
BATCH_SIZE = 256
EPOCHS = 10
LR = 1e-3
EVAL_EVERY = 10


def main():
    seed_everything(0)
    device = get_device()
    print(f"Using device: {device}")

    train_ds = DATASET(train=True)
    test_ds = DATASET(train=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

    c, h, w = DATASET.image_shape
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(c * h * w, 16),
        nn.GELU(),
        nn.Linear(16, DATASET.num_classes),
    ).to(device)
    summary(model, input_size=(BATCH_SIZE, c, h, w), device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        for x, y in tqdm(train_dl, desc=f"epoch {epoch + 1}/{EPOCHS}"):
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        if (epoch + 1) % EVAL_EVERY != 0 and epoch + 1 != EPOCHS:
            continue
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_dl:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
        print(f"epoch {epoch + 1}: loss={loss.item():.4f} test_acc={correct / len(test_ds):.4f}")


if __name__ == "__main__":
    main()
