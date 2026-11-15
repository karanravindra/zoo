"""Precompute tensor caches: `python -m zoo.data.precompute MNIST QMNIST`."""

from __future__ import annotations

import argparse

import zoo.data
from zoo.data.image import ImageDataset

DATASETS: dict[str, type[ImageDataset]] = {
    name: obj
    for name in zoo.data.__all__
    if isinstance(obj := getattr(zoo.data, name), type) and issubclass(obj, ImageDataset) and obj is not ImageDataset
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("datasets", nargs="+", choices=sorted(DATASETS))
    p.add_argument("--root", default="data")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--chunk-size", type=int, default=1024)
    args = p.parse_args()

    for name in args.datasets:
        for train in (True, False):
            out = DATASETS[name].build_cache(
                args.root, train, num_workers=args.num_workers, chunk_size=args.chunk_size
            )
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
