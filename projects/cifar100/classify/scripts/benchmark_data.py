"""Compare CIFAR-100 loader throughput without model execution."""

import argparse
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torchvision

from zoo.data import make_loader
from zoo.data.benchmark import benchmark_loader
from zoo.data.cifar100 import (
    BACKENDS,
    add_data_arguments,
    config_from_args,
    make_splits,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--backends", choices=BACKENDS, nargs="+", default=list(BACKENDS))
    parser.add_argument("--workers", type=int, nargs="+")
    parser.add_argument("--splits", choices=("train", "val", "test"), nargs="+", default=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batches", type=int, default=100, help="0 means a full pass")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/dataloader_benchmark.json")
    args = parser.parse_args()
    if min(args.batch_size, args.repeats, args.threads) < 1 or args.batches < 0:
        parser.error("batch-size, repeats, threads must be positive; batches nonnegative")
    try:
        configs = [config_from_args(args, backend=backend, num_workers=workers)
                   for backend in args.backends for workers in (args.workers or [args.num_workers])]
    except ValueError as error:
        parser.error(str(error))
    torch.set_num_threads(args.threads)
    report = {"environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "torch": torch.__version__, "torchvision": torchvision.__version__, "threads": args.threads},
              "batch_size": args.batch_size, "batches": args.batches,
              "repeats": args.repeats, "seed": args.seed, "results": []}
    for config in configs:
        torch.manual_seed(args.seed)
        start = time.perf_counter()
        datasets = dict(zip(("train", "val", "test"), make_splits(config, args.seed)))
        setup_seconds = time.perf_counter() - start
        for split in args.splits:
            loader = make_loader(datasets[split], config, args.batch_size, shuffle=split == "train", seed=args.seed)
            measurements = benchmark_loader(loader, args.batches, args.repeats)
            report["results"].append(dict(backend=config.backend, split=split,
                                          config={key: str(value) if isinstance(value, Path) else value
                                                  for key, value in asdict(config).items()},
                                          setup_seconds=setup_seconds, **measurements))
            print(f"{config.backend:12s} {split:5s} workers={config.num_workers}: "
                  f"{measurements['median_samples_per_second']:,.0f} samples/s; setup={setup_seconds:.2f}s", flush=True)
            del loader
        del datasets
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
