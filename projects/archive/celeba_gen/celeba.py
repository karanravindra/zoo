"""CelebA-HQ at 256x256, cached as uint8 on disk.

`CelebAHQ256` is an `ImageDataset`: the first construction resizes every image to 256x256 (bicubic, antialiased) and
writes `<root>/cache/CelebAHQ256/{train,test}/`, so later runs read memory-mapped arrays instead of decoding the 1024px
originals again. The size is part of the class name because the cache key is the class name: a different resolution
needs a different class, never a different value passed to the same cache.

Splits follow `zoo.data.stargan`: `train=True` is the StarGAN train split (28,000 images), `train=False` is its val
split (2,000 images). Training scripts carve their own validation set out of the train split.

`load_split` and `Batches` replace a `DataLoader` for training: a split is read once into one uint8 tensor on the
training device, and a batch is an index plus a cast to [-1, 1] on that device. There are no worker processes, no
per-sample transform and no host-to-device copy per step. The training split takes 5.3 GB of device memory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split
from torchvision import datasets
from torchvision.transforms.v2.functional import pil_to_tensor

from zoo.data import stargan
from zoo.data.image import ImageDataset
from zoo.paths import DATA_ROOT

NAME = "celeba_hq"
SIZE = 256


class CelebAHQ256(ImageDataset):
    num_classes = 2  # the folders under each split (female / male); the autoencoders ignore labels
    image_shape = (3, SIZE, SIZE)
    mean = (0.5, 0.5, 0.5)  # with std 0.5 the default transform maps uint8 [0, 255] to [-1, 1]
    std = (0.5, 0.5, 0.5)

    @classmethod
    def preprocess(cls, img: Image.Image):
        # Pillow's resize antialiases when downsampling, so this is a proper bicubic downscale from 1024px.
        img = img.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BICUBIC)
        return pil_to_tensor(img)

    @classmethod
    def _load(cls, root, train, download):
        root = Path(root)
        base = stargan.download(NAME, root) if download else root / NAME
        split = stargan.splits(NAME)[0 if train else 1]
        return datasets.ImageFolder(str(base / split))


def make_splits(val_size: int, seed: int) -> tuple[Dataset, Dataset, Dataset]:
    """(train, val, test): `val_size` images are carved out of the StarGAN train split; its val split is the test set."""
    full_train = CelebAHQ256(root=str(DATA_ROOT), train=True)
    test = CelebAHQ256(root=str(DATA_ROOT), train=False)
    train, val = random_split(full_train, [len(full_train) - val_size, val_size], generator=torch.Generator().manual_seed(seed))
    return train, val, test


def load_split(ds: Dataset, device: torch.device | str, chunk: int = 1024) -> torch.Tensor:
    """The images of a split from `make_splits` as one uint8 tensor (N, 3, 256, 256) on `device`, in the split's order
    (5.3 GB for the training split). Read from the memory-mapped cache in chunks, so the host never holds a second
    copy."""
    base, indices = (ds.dataset, ds.indices) if isinstance(ds, Subset) else (ds, range(len(ds)))
    src = np.load(base.cache_dir / "images.npy", mmap_mode="r")
    idx = np.asarray(indices)
    out = torch.empty((len(idx), *src.shape[1:]), dtype=torch.uint8, device=device)
    for i in range(0, len(idx), chunk):
        out[i : i + chunk] = torch.from_numpy(np.asarray(src[idx[i : i + chunk]])).to(device)
    return out


class Batches:
    """Iterate a uint8 image tensor from `load_split` in batches of float32 images normalized to [-1, 1], on the
    tensor's device (the same values as `CelebAHQ256`'s default transform). A batch is a gather on the device.
    `len()` is the number of batches per epoch, and with `shuffle` every pass is a new permutation from a generator
    seeded with `seed`."""

    def __init__(
        self,
        images: torch.Tensor,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: int = 0,
        mean: tuple[float, ...] = CelebAHQ256.mean,
        std: tuple[float, ...] = CelebAHQ256.std,
    ) -> None:
        self.images, self.batch_size, self.shuffle, self.drop_last = images, batch_size, shuffle, drop_last
        self.generator = torch.Generator().manual_seed(seed)
        self.mean = torch.tensor(mean, device=images.device).view(1, -1, 1, 1)
        self.std = torch.tensor(std, device=images.device).view(1, -1, 1, 1)

    def __len__(self) -> int:
        n = len(self.images)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def __iter__(self):
        n, bs = len(self.images), self.batch_size
        order = torch.randperm(n, generator=self.generator) if self.shuffle else torch.arange(n)
        order = order.to(self.images.device)  # one copy per epoch: a per-step host-to-device copy would sync the host
        for i in range(len(self)):
            x = self.images[order[i * bs : (i + 1) * bs]]
            yield (x.float().div_(255) - self.mean) / self.std
