"""Preview the same random crop and flip used by CIFAR-100 training."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision import datasets

from zoo.data.cifar100 import AUGMENT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/augmentations.png")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    dataset = datasets.CIFAR100(args.data_dir, train=True, download=True)
    fig, axes = plt.subplots(5, 6, figsize=(11, 10))
    for row in range(5):
        index = dataset.targets.index(row)
        image, label = dataset[index]
        for col in range(6):
            axes[row, col].imshow(image if col == 0 else AUGMENT(image), interpolation="nearest")
            axes[row, col].set_title(dataset.classes[label] if col == 0 else f"Augment {col}", fontsize=9)
            axes[row, col].axis("off")
    fig.suptitle("CIFAR-100 · random crop (padding 4) + horizontal flip")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
