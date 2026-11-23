"""Train DigitNet (a small CNN) on MNIST.

Run from anywhere: `uv run projects/digitnet/train.py`

Each run is a W&B run in the `digitnet` project, with local files in `<runs>/digitnet/<run id>/`:
  checkpoints/   the best by val/acc, plus last.ckpt
  config.json    the full config
  csv/           metrics.csv, a local copy of the logged metrics
  wandb/         W&B's own files (`wandb sync` uploads an offline run)
"""

from dataclasses import asdict

import tyro
from torchinfo import summary

from config import Config
from data import DigitData
from module import LitDigitNet

from zoo import data
from zoo.training import make_trainer
from zoo.utils.device import seed_everything

PROJECT = "digitnet"


def main(cfg: Config):
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    module = LitDigitNet(asdict(cfg))
    c, h, w = getattr(data, cfg.dataset).image_shape
    summary(module.net, input_size=(cfg.batch_size, c, h, w), device="cpu", depth=2)

    trainer = make_trainer(cfg, PROJECT, monitor="val/acc", mode="max")
    datamodule = DigitData(cfg)
    trainer.fit(module, datamodule)
    # Score the test set exactly once, with the checkpoint chosen on validation (a dev run keeps none: final weights).
    trainer.test(module, datamodule, ckpt_path=None if cfg.dev else "best")


if __name__ == "__main__":
    main(tyro.cli(Config, description=__doc__))
