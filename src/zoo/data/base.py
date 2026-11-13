"""Abstract base class for image classification datasets, with optional on-disk tensor caches."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, ClassVar

import numpy as np
import torch
from numpy.lib.format import open_memmap
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from torchvision.transforms.v2.functional import pil_to_tensor
from tqdm import tqdm


class ImageDataset(Dataset, ABC):
    """An image classification dataset with known shape and normalization stats.

    Subclasses set the class-level metadata and implement `_load`, which returns an
    indexable source yielding `(PIL image, label)` pairs.

    Every image goes through `preprocess` (PIL -> uint8 CHW tensor of `image_shape`),
    then `transform`. With `cache=True` the preprocessed images are written once to
    a memory-mapped `.npy` file under `<root>/cache/<ClassName>/<split>/` and read
    from there afterwards, so the source dataset is never touched again and the
    dataset never has to fit in RAM.

    Note: transforms receive uint8 CHW tensors, not PIL images.
    """

    num_classes: ClassVar[int]
    image_shape: ClassVar[tuple[int, int, int]]  # (C, H, W)
    mean: ClassVar[tuple[float, ...]]
    std: ClassVar[tuple[float, ...]]

    def __init__(
        self,
        root: str = "data",
        train: bool = True,
        transform: Callable | None = None,
        download: bool = True,
        cache: bool = True,
        num_workers: int | None = None,
    ) -> None:
        self.train = train
        self.transform = transform if transform is not None else self.default_transform()
        self._source = None
        self._images = self._labels = None  # memmaps, opened lazily per process

        if cache:
            self.cache_dir = self.cache_path(root, train)
            if not (self.cache_dir / "meta.json").exists():
                self.build_cache(root, train, download, num_workers)
            meta = json.loads((self.cache_dir / "meta.json").read_text())
            self._len = meta["length"]
        else:
            self.cache_dir = None
            self._source = self._load(root, train, download)
            self._len = len(self._source)

    # --- subclass hooks -------------------------------------------------------

    @classmethod
    @abstractmethod
    def _load(cls, root: str, train: bool, download: bool) -> Dataset:
        """Return the raw dataset (items are `(PIL image, label)`)."""

    @classmethod
    def preprocess(cls, img: Image.Image) -> torch.Tensor:
        """Deterministic PIL -> uint8 CHW tensor of `image_shape`. This is what gets cached.

        Override (e.g. to resize/crop) for datasets with variable-size images.
        """
        c, h, w = cls.image_shape
        img = img.convert("RGB" if c == 3 else "L")
        x = pil_to_tensor(img)
        if tuple(x.shape) != cls.image_shape:
            raise ValueError(
                f"{cls.__name__}: got image of shape {tuple(x.shape)}, expected {cls.image_shape}. "
                "Override `preprocess` to resize."
            )
        return x

    @classmethod
    def default_transform(cls) -> Callable:
        return v2.Compose([v2.ToDtype(torch.float32, scale=True), v2.Normalize(cls.mean, cls.std)])

    # --- caching --------------------------------------------------------------

    @classmethod
    def cache_path(cls, root: str, train: bool) -> Path:
        return Path(root) / "cache" / cls.__name__ / ("train" if train else "test")

    @classmethod
    def build_cache(
        cls,
        root: str = "data",
        train: bool = True,
        download: bool = True,
        num_workers: int | None = None,
        chunk_size: int = 1024,
    ) -> Path:
        """Preprocess the whole split into memory-mapped `.npy` files.

        Streams in chunks (decoding in parallel workers), so memory use stays
        bounded regardless of dataset size. Files are written under temp names and
        `meta.json` is written last, so an interrupted build is never mistaken for a
        complete one.
        """
        out = cls.cache_path(root, train)
        out.mkdir(parents=True, exist_ok=True)
        source = cls._load(root, train, download)
        n = len(source)

        if num_workers is None:
            num_workers = min(8, os.cpu_count() or 1)
        loader = DataLoader(
            _Preprocessed(cls, source),
            batch_size=chunk_size,
            num_workers=num_workers,
            persistent_workers=False,
        )

        img_tmp, lbl_tmp = out / "images.tmp.npy", out / "labels.tmp.npy"
        images = open_memmap(img_tmp, mode="w+", dtype=np.uint8, shape=(n, *cls.image_shape))
        labels = open_memmap(lbl_tmp, mode="w+", dtype=np.int64, shape=(n,))

        i = 0
        split = "train" if train else "test"
        for x, y in tqdm(loader, desc=f"caching {cls.__name__}/{split}", unit_scale=chunk_size, unit="img"):
            images[i : i + len(x)] = x.numpy()
            labels[i : i + len(y)] = y.numpy()
            i += len(x)
        assert i == n, f"expected {n} items, wrote {i}"

        images.flush()
        labels.flush()
        del images, labels
        os.replace(img_tmp, out / "images.npy")
        os.replace(lbl_tmp, out / "labels.npy")
        (out / "meta.json").write_text(json.dumps({"length": n, "image_shape": list(cls.image_shape)}))
        return out

    def _open_cache(self) -> None:
        self._images = np.load(self.cache_dir / "images.npy", mmap_mode="r")
        self._labels = np.load(self.cache_dir / "labels.npy", mmap_mode="r")

    def __getstate__(self):
        # Don't pickle memmaps into DataLoader workers (that would copy the data);
        # each worker reopens them lazily.
        state = self.__dict__.copy()
        state["_images"] = state["_labels"] = None
        return state

    # --- Dataset API ----------------------------------------------------------

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        if self.cache_dir is not None:
            if self._images is None:
                self._open_cache()
            img = torch.from_numpy(np.array(self._images[idx]))
            label = int(self._labels[idx])
        else:
            raw, label = self._source[idx]
            img, label = self.preprocess(raw), int(label)
        return self.transform(img), label


class _Preprocessed(Dataset):
    """Wraps a raw source so `build_cache` can decode in DataLoader workers."""

    def __init__(self, cls: type[ImageDataset], source: Dataset) -> None:
        self.cls, self.source = cls, source

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img, label = self.source[idx]
        return self.cls.preprocess(img), int(label)
