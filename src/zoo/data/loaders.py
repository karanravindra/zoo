from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader


@dataclass
class LoaderConfig:
    num_workers: int = 0
    pin_memory: bool = False
    prefetch_factor: int = 2
    persistent_workers: bool = False

    def __post_init__(self):
        if self.num_workers < 0 or self.prefetch_factor < 1:
            raise ValueError(
                "num-workers must be nonnegative and prefetch-factor positive"
            )
        if self.persistent_workers and not self.num_workers:
            raise ValueError("persistent-workers requires num-workers > 0")


def add_loader_arguments(parser):
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true")


def make_loader(dataset, config, batch_size, shuffle=False, seed=42):
    kwargs = {}
    if config.num_workers:
        kwargs.update(
            prefetch_factor=config.prefetch_factor,
            persistent_workers=config.persistent_workers,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        generator=torch.Generator().manual_seed(seed),
        **kwargs,
    )
