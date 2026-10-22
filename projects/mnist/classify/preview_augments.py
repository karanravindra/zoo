from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms


def main():
    torch.manual_seed(42)
    dataset = datasets.MNIST("data", train=True, download=True)
    resize = transforms.Resize(
        (32, 32), interpolation=transforms.InterpolationMode.BILINEAR,
    )
    augment = transforms.RandomAffine(
        degrees=30, shear=10, interpolation=transforms.InterpolationMode.BILINEAR,
    )
    fig, axes = plt.subplots(5, 6, figsize=(10, 9))
    for row, digit in enumerate((0, 1, 2, 5, 8)):
        index = (dataset.targets == digit).nonzero()[0].item()
        image = resize(dataset[index][0])
        for col in range(6):
            ax = axes[row, col]
            ax.imshow(image if col == 0 else augment(image), cmap="gray",
                      vmin=0, vmax=255, interpolation="nearest")
            ax.axis("off")
            if row == 0:
                ax.set_title("Original" if col == 0 else f"Augment {col}")
    fig.suptitle("32×32 MNIST · random rotation ±15° + shear ±10° · bilinear")
    fig.tight_layout()
    output = Path(__file__).parent / "outputs" / "augmentations.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
