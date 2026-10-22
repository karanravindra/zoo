"""MNIST preprocessing, storage strategies, and loader configuration."""

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

RESIZE = transforms.Resize(
    (32, 32), interpolation=transforms.InterpolationMode.BILINEAR
)
NORMALIZE = transforms.Normalize((0.1307,), (0.3081,))
AFFINE = transforms.RandomAffine(
    degrees=30,
    shear=10,
    interpolation=transforms.InterpolationMode.BILINEAR,
)
EVALUATE = transforms.Compose([RESIZE, transforms.ToTensor(), NORMALIZE])
TRAIN = transforms.Compose([RESIZE, AFFINE, transforms.ToTensor(), NORMALIZE])


def source_digest(dataset):
    digest = hashlib.sha256()
    for tensor in (dataset.data, dataset.targets):
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def pil_backend(source, training, config):
    return TransformDataset(source, TRAIN if training else EVALUATE)


def tensor_cache_backend(source, training, config):
    # Preserve PIL affine interpolation and integer rounding used by training.
    # Evaluation can cache the entire deterministic pipeline as float32.
    preprocess = (
        transforms.Compose([RESIZE, transforms.PILToTensor()]) if training else EVALUATE
    )
    metadata = {
        "dataset": "MNIST",
        "source_sha256": source_digest(source),
        "preprocessing": "resize32-uint8-v1"
        if training
        else "resize32-normalized-f32-v1",
        "mean": 0.1307,
        "std": 0.3081,
        "interpolation": "PIL-bilinear",
        "torchvision": torchvision.__version__,
        "pillow": pillow_version,
    }
    stored = TensorChunks.prepare(
        config.cache_dir,
        TransformDataset(source, preprocess),
        metadata,
        chunk_size=config.chunk_size,
        max_open_chunks=config.max_open_chunks,
    )
    if training:
        return TransformDataset(
            stored,
            transforms.Compose(
                [
                    transforms.ToPILImage(),
                    AFFINE,
                    transforms.ToTensor(),
                    NORMALIZE,
                ]
            ),
        )
    return stored


# New storage strategies plug into the same training and benchmark paths.
BACKENDS = {"pil": pil_backend, "tensor-cache": tensor_cache_backend}


@dataclass
class DataConfig(LoaderConfig):
    data_dir: Path = Path("data")
    cache_dir: Path = Path("data/mnist-tensor-cache")
    backend: str = "pil"
    chunk_size: int = 8192
    max_open_chunks: int = 16

    def __post_init__(self):
        super().__post_init__()
        if self.backend not in BACKENDS:
            raise ValueError(f"Unknown backend: {self.backend}")
        if min(self.chunk_size, self.max_open_chunks) < 1:
            raise ValueError("chunk-size and max-open-chunks must be positive")


def add_data_arguments(parser):
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("data/mnist-tensor-cache")
    )
    parser.add_argument("--backend", choices=BACKENDS, default="pil")
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--max-open-chunks", type=int, default=16)
    add_loader_arguments(parser)


def config_from_args(args, **overrides):
    values = {field: getattr(args, field) for field in DataConfig.__dataclass_fields__}
    values.update(overrides)
    return DataConfig(**values)


def make_dataset(config, split):
    if split not in ("train", "val", "test"):
        raise ValueError(f"Unknown split: {split}")
    source = datasets.MNIST(config.data_dir, train=split != "test", download=True)
    return BACKENDS[config.backend](source, split == "train", config)


def make_splits(config, seed):
    indices = torch.randperm(
        60000, generator=torch.Generator().manual_seed(seed)
    ).tolist()
    return (
        Subset(make_dataset(config, "train"), indices[:-512]),
        Subset(make_dataset(config, "val"), indices[-512:]),
        make_dataset(config, "test"),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Precompute deterministic MNIST tensor chunks."
    )
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
        print(
            f"{split}: {len(dataset)} samples, {getattr(stored, 'path', config.backend)}"
        )


if __name__ == "__main__":
    main()
