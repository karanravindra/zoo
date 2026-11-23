"""DigitNet: a small CNN on MNIST.

  uv run projects/digitnet/cli.py prepare           download and cache the dataset
  uv run projects/digitnet/cli.py train [flags]     train and score the test set once

`<command> --help` lists a command's flags. A training run is a W&B run in the `digitnet` project, with local files in
`<runs>/digitnet/<run id>/`:
  checkpoints/   the best by val/acc, plus last.ckpt
  config.json    the full config
  csv/           metrics.csv, a local copy of the logged metrics
  wandb/         W&B's own files (`wandb sync` uploads an offline run)
"""

from dataclasses import asdict
from typing import Annotated

import tyro
from torchinfo import summary

from config import Config, PrepareConfig
from data import DigitData
from module import LitDigitNet

from zoo import data
from zoo.paths import DATA_ROOT
from zoo.training import make_trainer
from zoo.utils.device import seed_everything

PROJECT = "digitnet"


def prepare(cfg: PrepareConfig) -> None:
    ds_cls = getattr(data, cfg.dataset)
    for train in (True, False):
        ds_cls(root=str(DATA_ROOT), train=train)
    print(f"{cfg.dataset} is cached")


def train(cfg: Config) -> None:
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    module = LitDigitNet(asdict(cfg))
    c, h, w = getattr(data, cfg.dataset).image_shape
    summary(module.net, input_size=(cfg.batch_size, c, h, w), device="cpu", depth=2)

    trainer = make_trainer(cfg, PROJECT, monitor="val/acc", mode="max")
    datamodule = DigitData(cfg)
    trainer.fit(module, datamodule)
    # Score the test set exactly once, with the checkpoint chosen on validation (a dev run keeps none: final weights).
    trainer.test(module, datamodule, ckpt_path=None if cfg.dev else "best")


Command = Annotated[PrepareConfig, tyro.conf.subcommand("prepare")] | Annotated[Config, tyro.conf.subcommand("train")]


def main(cmd: Command) -> None:
    match cmd:
        case PrepareConfig():
            prepare(cmd)
        case Config():
            train(cmd)


if __name__ == "__main__":
    main(tyro.cli(Command, description=__doc__))
