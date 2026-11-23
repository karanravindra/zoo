"""Build the Lightning `Trainer` (W&B + CSV logging, checkpointing) from a `RunConfig`."""

from __future__ import annotations

import atexit
import json
import shutil
import tempfile
import uuid
from dataclasses import asdict
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from zoo.paths import runs_root
from zoo.training.callbacks import EMA, ParamStatsMonitor, ThroughputMonitor
from zoo.training.config import RunConfig


def compile_(model: torch.nn.Module) -> torch.nn.Module:
    """Compile in place (not `torch.compile(model)`, which adds `_orig_mod.` to checkpoint keys).

    Call it on the inner network (e.g. `self.net`) in the LightningModule's `setup`, after any weight-dependent copies.
    `dynamic=False`: with dynamic shapes, Python floats become float64 graph inputs, which crash Inductor on MPS and
    slow it down even when they don't; varying shapes just trigger recompiles instead."""
    model.compile(mode="reduce-overhead", dynamic=False)
    return model


def make_trainer(
    cfg: RunConfig,
    project: str,
    *,
    monitor: str | None = None,
    mode: str = "min",
    callbacks: list[Callback] | None = None,
    stage: str | None = None,
    **trainer_kwargs,
) -> L.Trainer:
    """A `Trainer` whose run lives in `runs/<project>/<run id>/` (the W&B run id).

    Layout: `checkpoints/` (best by `monitor` if given, plus `last.ckpt`), `config.json`, `csv/metrics.csv` and
    W&B's own `wandb/` files. The W&B project is `project`, the group is `cfg.group` (or `stage`).
    With `cfg.dev` nothing is logged or kept: no W&B, no checkpoints, outputs in a temp dir deleted on exit.
    `cfg.ema > 0` adds the `EMA` callback. Extra keyword arguments go to `Trainer` (e.g. `val_check_interval`, `devices`, `strategy`)."""
    callbacks = list(callbacks or [])
    if cfg.ema > 0 and not any(isinstance(c, EMA) for c in callbacks):
        callbacks.append(EMA(cfg.ema))
    callbacks += [ThroughputMonitor(cfg.log_every), ParamStatsMonitor(cfg.log_every)]

    if cfg.dev:
        run_dir = Path(tempfile.mkdtemp(prefix=f"zoo-dev-{project}-"))
        atexit.register(shutil.rmtree, run_dir, ignore_errors=True)
        print(f"Dev run: nothing is logged or kept (scratch outputs in {run_dir}, deleted on exit)")
        loggers, checkpointing = False, False  # no logger, so no LearningRateMonitor either (it refuses to run without one)
    else:
        run_id = uuid.uuid4().hex[:8]  # the same shape as a W&B run id
        run_dir = runs_root(project) / run_id
        run_dir.mkdir(parents=True)
        wandb_logger = WandbLogger(
            project=project,
            id=run_id,
            name=cfg.name,
            group=cfg.group or stage,
            tags=list(cfg.tags),
            save_dir=run_dir,
            mode=cfg.wandb,
            log_model=False,
        )
        wandb_logger.experiment.config.update(asdict(cfg), allow_val_change=True)
        loggers = [wandb_logger]
        if cfg.csv:
            loggers.append(CSVLogger(save_dir=run_dir, name="csv", version=""))
        (run_dir / "config.json").write_text(json.dumps({"project": project, "stage": stage, "config": asdict(cfg)}, indent=2))
        checkpointing = True
        callbacks.append(LearningRateMonitor(logging_interval="step"))
        callbacks.append(
            ModelCheckpoint(dirpath=run_dir / "checkpoints", monitor=monitor, mode=mode, save_last=True, save_top_k=1 if monitor else 0)
        )
        print(f"Run {run_id}: {run_dir}")

    return L.Trainer(
        default_root_dir=run_dir,
        max_epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        precision=cfg.precision,
        gradient_clip_val=cfg.grad_clip or None,
        log_every_n_steps=cfg.log_every,
        deterministic=cfg.deterministic or None,
        logger=loggers,
        enable_checkpointing=checkpointing,
        callbacks=callbacks,
        **trainer_kwargs,
    )
