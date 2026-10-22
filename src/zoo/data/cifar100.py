"""CIFAR-100 preprocessing, stratified splits, and storage backends."""

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision
from PIL import __version__ as pillow_version
from torch.utils.data import Subset
from torchvision import datasets, transforms

from .cache import TensorChunks
from .loaders import LoaderConfig, add_loader_arguments
from .transforms import TransformDataset

MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)
NORMALIZE = transforms.Normalize(MEAN, STD)
AUGMENT = transforms.Compose([
    transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
])
EVALUATE = transforms.Compose([transforms.ToTensor(), NORMALIZE])
TRAIN = transforms.Compose([AUGMENT, transforms.ToTensor(), NORMALIZE])


@dataclass
class DataConfig(LoaderConfig):
    data_dir: Path = Path("data")
    cache_dir: Path = Path("data/cifar100-tensor-cache")
    backend: str = "pil"
    chunk_size: int = 8192
    max_open_chunks: int = 16

    def __post_init__(self):
        super().__post_init__()
        if self.backend not in BACKENDS:
            raise ValueError(f"Unknown backend: {self.backend}")
        if min(self.chunk_size, self.max_open_chunks) < 1:
            raise ValueError("chunk-size and max-open-chunks must be positive")


def pil_backend(source, training, config):
    return TransformDataset(source, TRAIN if training else EVALUATE)


def tensor_cache_backend(source, training, config):
    digest = hashlib.sha256(source.data.tobytes())
    digest.update(torch.tensor(source.targets, dtype=torch.int64).numpy().tobytes())
    metadata = {"dataset": "CIFAR100", "source_sha256": digest.hexdigest(),
                "classes": source.classes, "mean": list(MEAN), "std": list(STD),
                "preprocessing": "rgb-uint8-v1" if training else "normalized-f32-v1",
                "torchvision": torchvision.__version__, "pillow": pillow_version}
    stored = TensorChunks.prepare(
        config.cache_dir,
        TransformDataset(source, transforms.PILToTensor() if training else EVALUATE),
        metadata, chunk_size=config.chunk_size, max_open_chunks=config.max_open_chunks,
    )
    if training:
        return TransformDataset(stored, transforms.Compose([transforms.ToPILImage(), TRAIN]))
    return stored


BACKENDS = {"pil": pil_backend, "tensor-cache": tensor_cache_backend}


def add_data_arguments(parser):
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cifar100-tensor-cache"))
    parser.add_argument("--backend", choices=BACKENDS, default="pil")
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--max-open-chunks", type=int, default=16)
    add_loader_arguments(parser)


def config_from_args(args, **overrides):
    values = {field: getattr(args, field) for field in DataConfig.__dataclass_fields__}
    return DataConfig(**(values | overrides))


def make_dataset(config, split):
    if split not in ("train", "val", "test"):
        raise ValueError(f"Unknown split: {split}")
    source = datasets.CIFAR100(config.data_dir, train=split != "test", download=True)
    result = BACKENDS[config.backend](source, split == "train", config)
    result.classes = list(source.classes)
    return result


def stratified_indices(targets, seed, validation_per_class=50):
    generator = torch.Generator().manual_seed(seed)
    targets = torch.as_tensor(targets)
    train, val = [], []
    for label in targets.unique(sorted=True):
        indices = (targets == label).nonzero().flatten()
        if len(indices) <= validation_per_class or validation_per_class < 1:
            raise ValueError("Each class needs more samples than validation-per-class")
        indices = indices[torch.randperm(len(indices), generator=generator)]
        val.extend(indices[:validation_per_class].tolist())
        train.extend(indices[validation_per_class:].tolist())
    return train, val


def make_splits(config, seed):
    source = datasets.CIFAR100(config.data_dir, train=True, download=True)
    train_indices, val_indices = stratified_indices(source.targets, seed)
    train = BACKENDS[config.backend](source, True, config)
    val = BACKENDS[config.backend](source, False, config)
    return Subset(train, train_indices), Subset(val, val_indices), make_dataset(config, "test")


def main():
    parser = argparse.ArgumentParser(description="Precompute CIFAR-100 tensor chunks.")
    add_data_arguments(parser)
    parser.set_defaults(backend="tensor-cache")
    args = parser.parse_args()
    try:
        config = config_from_args(args)
    except ValueError as error:
        parser.error(str(error))
    for split in ("train", "val", "test"):
        dataset = make_dataset(config, split)
        stored = dataset.dataset if isinstance(dataset, TransformDataset) else dataset
        print(f"{split}: {len(dataset)} samples, {getattr(stored, 'path', config.backend)}")


if __name__ == "__main__":
    main()
