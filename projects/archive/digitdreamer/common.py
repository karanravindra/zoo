"""Shared bits for the DigitDreamer training scripts: the MNIST transform and train/val/test splits."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, random_split
from torchvision.transforms import v2

from zoo import data
from zoo.paths import DATA_ROOT

# MNIST is 28x28; pad 2 px of background (0) on each side to 32x32, then uint8 [0, 255] -> float [-1, 1]
TO_PM1 = v2.Compose([v2.Pad(2, fill=0), v2.ToDtype(torch.float32, scale=True), v2.Normalize((0.5,), (0.5,))])


def make_splits(dataset: str, val_size: int, seed: int) -> tuple[Dataset, Dataset, Dataset]:
    """(train, val, test), with val carved deterministically out of the train split."""
    ds_cls = getattr(data, dataset)
    full_train = ds_cls(root=str(DATA_ROOT), train=True, transform=TO_PM1)
    test = ds_cls(root=str(DATA_ROOT), train=False, transform=TO_PM1)
    train, val = random_split(
        full_train, [len(full_train) - val_size, val_size], generator=torch.Generator().manual_seed(seed)
    )
    return train, val, test
