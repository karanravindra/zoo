"""fastdocling: draft models for faster granite-docling decoding (speculative decoding), mainly on Apple silicon.

  uv run projects/fastdocling/cli.py prepare [flags]                  self-distill the target's DocTags over page images (vLLM, GPU)
  uv run projects/fastdocling/cli.py train --drafter mtp [flags]      train a drafter on the frozen target's hidden states
  uv run projects/fastdocling/cli.py bench <ckpt> [<ckpt> ...]        speed-up and exactness of each drafter, in and out of domain
  uv run projects/fastdocling/cli.py vllm-bench                       the target's throughput in vLLM, plain and with its n-gram drafter

`<command> --help` lists a command's flags. Runs are W&B runs in the `fastdocling` project (group = the drafter name), with local
files in `<runs>/fastdocling/<run id>/`.
"""

import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import tyro

from config import PROMPT, BenchConfig, PrepareConfig, TrainConfig, VllmBenchConfig
from data import DistillData
from module import LitDrafter
from pages import distill_shards

from zoo.artifacts import current_build, publish_build, staged_build
from zoo.training import make_trainer
from zoo.utils.device import seed_everything

PROJECT = "fastdocling"


def prepare(cfg: PrepareConfig) -> None:
    # vLLM pins an older torch than the project, so generation runs in its own uv env (the script's inline dependencies).
    worker = Path(__file__).with_name("generate.py")
    if cfg.n_pages < 1:
        raise ValueError("n_pages must be positive")
    with staged_build(cfg.out) as stage:
        args = json.dumps(asdict(cfg) | {"out": stage, "prompt": PROMPT}, default=str)
        subprocess.run(["uv", "run", str(worker), args], check=True)
        distill_shards(str(stage), target=cfg.target)
        publish_build(cfg.out, stage)
    print(f"Wrote {cfg.out}")


def train(cfg: TrainConfig) -> None:
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    module = LitDrafter(asdict(cfg))
    n = sum(p.numel() for p in module.drafter.parameters())
    print(f"Drafter {cfg.drafter}: {n / 1e6:.1f}M parameters, k={cfg.k}")
    trainer = make_trainer(cfg, PROJECT, stage=cfg.drafter, monitor="val/accept_len", mode="max")
    datamodule = DistillData(cfg)
    trainer.fit(module, datamodule)


def bench(cfg: BenchConfig) -> None:
    from bench import run_bench  # imports the HF stack only when benchmarking

    run_bench(cfg)


def vllm_bench(cfg: VllmBenchConfig) -> None:
    # Like `prepare`: vLLM runs in its own uv env, one engine per setting, and the per-setting results are merged here.
    worker = Path(__file__).with_name("vllm_bench.py")
    records = []
    data = current_build(Path(cfg.data))
    for spec in cfg.specs:
        part = cfg.out.with_name(f"vllm_bench_{spec}.json")
        args = json.dumps(asdict(cfg) | {"spec": spec, "out": str(part), "data": data}, default=str)
        cfg.out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["uv", "run", str(worker), args], check=True)
        records += json.loads(part.read_text())
    cfg.out.write_text(json.dumps(records, indent=2))
    print(f"Wrote {cfg.out}")


Command = (
    Annotated[PrepareConfig, tyro.conf.subcommand("prepare")]
    | Annotated[TrainConfig, tyro.conf.subcommand("train")]
    | Annotated[BenchConfig, tyro.conf.subcommand("bench")]
    | Annotated[VllmBenchConfig, tyro.conf.subcommand("vllm-bench")]
)


def main(cmd: Command) -> None:
    match cmd:
        case PrepareConfig():
            prepare(cmd)
        case TrainConfig():
            train(cmd)
        case BenchConfig():
            bench(cmd)
        case VllmBenchConfig():
            vllm_bench(cmd)


if __name__ == "__main__":
    main(tyro.cli(Command, description=__doc__))
