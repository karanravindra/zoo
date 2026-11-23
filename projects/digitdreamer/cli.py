"""DigitDreamer: class-conditioned digit generation in two stages, an autoencoder and a rectified flow in its latent space.

  uv run projects/digitdreamer/cli.py prepare                          download and cache MNIST
  uv run projects/digitdreamer/cli.py ae [flags]                       stage 1: train the autoencoder
  uv run projects/digitdreamer/cli.py flow --ae-ckpt <path> [flags]    stage 2: train the flow on its latents
  uv run projects/digitdreamer/cli.py sample <flow checkpoint>         generate a grid of digits

`<command> --help` lists a command's flags. Training runs are W&B runs in the `digitdreamer` project (group `ae` or `flow`),
with local files in `<runs>/digitdreamer/<run id>/`: `checkpoints/` (the best, plus last.ckpt), `config.json`, `csv/metrics.csv`.
"""

from dataclasses import asdict
from typing import Annotated

import torch
import tyro
from torchinfo import summary
from torchvision.utils import save_image

from ae import IMAGE_SIZE, LATENT_SIZE, LitAutoencoder
from config import AEConfig, FlowConfig, PrepareConfig, SampleConfig
from data import ImageData, LatentData, prepare_data
from flow import LitFlow

from zoo import data
from zoo.metrics import to_unit
from zoo.training import load_module, make_trainer
from zoo.utils.device import get_device, seed_everything

PROJECT = "digitdreamer"


def prepare(cfg: PrepareConfig) -> None:
    prepare_data(cfg.dataset)
    print(f"{cfg.dataset} is cached")


def train_ae(cfg: AEConfig) -> None:
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    module = LitAutoencoder(asdict(cfg))
    c = getattr(data, cfg.dataset).image_shape[0]
    summary(module.net, input_size=(cfg.batch_size, c, IMAGE_SIZE, IMAGE_SIZE), device="cpu", depth=2)

    trainer = make_trainer(cfg, PROJECT, stage="ae", monitor="val/mse", mode="min")
    datamodule = ImageData(cfg)
    trainer.fit(module, datamodule)
    # Score the test set once, with the checkpoint chosen on validation (a dev run keeps none: final weights).
    trainer.test(module, datamodule, ckpt_path=None if cfg.dev else "best")


def train_flow(cfg: FlowConfig) -> None:
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    ae = load_module(LitAutoencoder, cfg.ae_ckpt)  # with the AE's EMA weights if it was trained with them
    print(f"Autoencoder: {cfg.ae_ckpt}")
    module = LitFlow(asdict(cfg), asdict(ae.cfg))
    module.ae.load_state_dict(ae.net.state_dict())
    b, c = cfg.batch_size, ae.cfg.latent_channels
    inputs = (torch.randn(b, c, LATENT_SIZE, LATENT_SIZE), torch.rand(b), torch.zeros(b, dtype=torch.long))
    summary(module.flow, input_data=inputs, device="cpu", depth=2)

    trainer = make_trainer(cfg, PROJECT, stage="flow", monitor="val/loss", mode="min")
    datamodule = LatentData(cfg, ae.net, ae.cfg)
    trainer.fit(module, datamodule)
    trainer.test(module, datamodule, ckpt_path=None if cfg.dev else "best")


def sample(cfg: SampleConfig) -> None:
    device = get_device()
    dreamer = load_module(LitFlow, cfg.ckpt).to(device)
    y = torch.tensor(cfg.digits, device=device).repeat_interleave(cfg.per_digit)
    g = torch.Generator(device=device).manual_seed(cfg.seed)
    shape = (len(y), dreamer.ae_cfg.latent_channels, LATENT_SIZE, LATENT_SIZE)
    images = dreamer.generate(y, cfg.steps, cfg.cfg, noise=torch.randn(shape, device=device, generator=g))
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    save_image(to_unit(images), cfg.out, nrow=cfg.per_digit, padding=1)
    print(f"Wrote {cfg.out} (from {cfg.ckpt})")


Command = (
    Annotated[PrepareConfig, tyro.conf.subcommand("prepare")]
    | Annotated[AEConfig, tyro.conf.subcommand("ae")]
    | Annotated[FlowConfig, tyro.conf.subcommand("flow")]
    | Annotated[SampleConfig, tyro.conf.subcommand("sample")]
)


def main(cmd: Command) -> None:
    match cmd:
        case PrepareConfig():
            prepare(cmd)
        case AEConfig():
            train_ae(cmd)
        case FlowConfig():
            train_flow(cmd)
        case SampleConfig():
            sample(cmd)


if __name__ == "__main__":
    main(tyro.cli(Command, description=__doc__))
