# zoo

Personal ML experiments. Shared library code lives in `src/zoo/`; each experiment is a project under `projects/<name>/`.

## Running code

- Never run training, embedding, download, or other pipeline scripts unless the user says it's okay. That okay lasts for the rest of the session. Without it, write the code and hand it off with the exact command for the user to run.
- Cheap checks don't need permission: imports, `--help`, a tiny CPU forward pass.
- Use uv for everything: `uv run ...` and `uv add ...`. Never use pip.

## Layout

- Training uses [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) and experiments are tracked in [Weights & Biases](https://docs.wandb.ai/). Every new project is built on them from the start.
- `projects/archive/` holds the old pre-Lightning projects, which use `zoo.runs`, `zoo.monitor`, hand-written loops and `metrics.csv`. Port a project out of it into `projects/<name>/` and delete the archived copy as you go. Don't add to the archive, and don't build on its patterns.
- Code reused across projects goes in `src/zoo/`. Start a project self-contained and promote code into `zoo` when a second project needs it.
  - `zoo.paths`: `REPO_ROOT`, `DATA_ROOT`, `RUNS_ROOT`, `runs_root(project)`. `ZOO_DATA_ROOT` / `ZOO_RUNS_ROOT` override the defaults (a Lightning Studio points them at its persistent storage).
  - `zoo.training`: `RunConfig` (the shared settings every project's `Config` extends), `make_trainer(cfg, project, monitor=, mode=, callbacks=, stage=)` (W&B + CSV loggers, checkpoints, LR/throughput/param-stat logging), `compile_(model)`, `log_images(trainer, key, images)` (a grid to W&B), `load_module(cls, path)` (a checkpoint back into its module, with the EMA weights applied), `make_scheduler(optimizer, cfg, total_steps)` (use it in `configure_optimizers`; the optimizer's `lr` is the peak), and the `EMA`, `ThroughputMonitor`, `ParamStatsMonitor` callbacks.
  - `zoo.train`: `update_ema`, `Throughput`, `ParamStats`. The pieces behind the callbacks above.
  - `zoo.plot`: `plot_curves(rows, series, path, x="epoch")`; importing it selects the Agg backend (`from zoo.plot import plt`).
  - `zoo.metrics`: `to_unit`, `psnr`, `ssim` (images in [-1, 1]; pass `size=7` for 16x16), bits-per-byte for text.
  - `zoo.data`: `ImageDataset`/`MNIST`/`QMNIST` (`data/image.py`), token corpora on disk as shards (`data/text.py`: `write_token_corpus` writes `shard_NNNN.npy` files, `Shards` / `TokenDataset` read them as one array; every project's prepare step writes through it).
  - `zoo.utils`: device and seeding.
  - `zoo.runs` and `zoo.monitor` are legacy, kept only until the archive is gone. Don't use them in new code.
- A project is a folder `projects/<name>/` with a `README.md`, otherwise self-contained:
  - `config.py`: `Config(RunConfig)` with the project's fields.
  - `module.py`: the model and its `LightningModule`, together. A multi-stage project has one such file per stage, named for it (`ae.py`, `flow.py`), each holding that stage's network, loss or sampler, and module.
  - `data.py`: `LightningDataModule` or dataloaders.
  - `cli.py`: the one entry point, with a tyro subcommand per step (`prepare`, `train`, ...), each a short function over its own config dataclass: `main(tyro.cli(Command))`. A multi-stage project has one training command per stage (`ae`, `flow`), which passes `stage=` to `make_trainer`, plus any `sample`/eval command. A later stage takes the earlier stage's checkpoint path as a required flag (`--ae-ckpt`), and its module bundles what it needs, so one checkpoint is self-contained. `projects/digitdreamer/` is the multi-stage reference.
  - `sweeps/*.yaml`: committed W&B sweep configs (`program` is the `cli.py` path, `command` uses `uv run` and names the subcommand before `${args}`).
  - `projects/digitnet/` is the single-stage reference.
- Scripts import sibling modules directly (`from config import Config`); Python already puts the script's directory on `sys.path`, so no `sys.path` hacks.
- Dependencies used by one project only live in a uv dependency group named after it (`uv run --group ledger ...`).
- Keep the README in sync when scripts, outputs, or configs change.

## Script config

- Every runnable script builds its settings from a dataclass and parses them with [tyro](https://brentyi.github.io/tyro/), so the dataclass fields define the CLI flags. Don't use argparse.
- Pattern: `main(tyro.cli(Command))`, where `Command` is a union of `Annotated[Config, tyro.conf.subcommand("name")]` (see `projects/digitnet/cli.py`). Defaults live in the dataclass, and a field's trailing comment is its `--help` text. Don't put a comment on the line above a field: tyro reads it as that field's help.
- Training configs extend `zoo.training.RunConfig`, which provides `epochs`, `max_steps`, `seed`, `deterministic`, `precision`, `grad_clip`, `lr_schedule` (`constant` = no decay, `cosine`, `linear`, `onecycle`), `warmup`, `min_lr_ratio`, `ema`, `compile`, `num_workers`, `log_every`, `wandb`, `name`, `group`, `tags`, `csv` and `dev`. Don't redefine them, other than to change a default.
- The config is the single source of truth. `make_trainer` logs it to W&B and writes `config.json`, and the module takes it as a plain dict (`save_hyperparameters`), so `load_from_checkpoint` rebuilds the model. Keep configs plain, serializable dataclasses.
- `compile: bool = False`. When it's set, call `compile_(self.net)` (in place: `model.compile(mode="reduce-overhead", dynamic=False)`, not `torch.compile(model)`, which adds `_orig_mod.` to the checkpoint keys) in the module's `setup` for `stage == "fit"`, on the inner network and not the LightningModule. Keep `dynamic=False`: with dynamic shapes, Python floats become float64 graph inputs, which crash Inductor on MPS and slow it down even when they don't. Varying shapes and float values just trigger recompiles instead.
- `dev: bool = False` makes a throwaway run: `make_trainer` skips W&B and checkpoints and uses a temp directory deleted on exit.
- `cfg.ema > 0` makes `make_trainer` add `zoo.training.EMA` (don't deepcopy the model for averaging). It updates once per optimizer step (not per accumulated micro-batch) and warms up: the decay is `min(cfg.ema, (1 + n) / (10 + n))` after `n` updates, so the window is about a ninth of the steps so far until it reaches the cap (a run's first k steps then do not depend on its planned length, and a very high `ema` such as 0.99999 only caps the window). Validation and test use the average, so the best checkpoint is chosen on it; checkpoints hold the live weights, and the average is saved under `callbacks["EMA"]` (`EMA.load_from_checkpoint(path).load_into(model)` applies it).

## Run outputs

- Metrics, config, git state, system stats (GPU utilisation, memory, power, temperature) and console output live in W&B. The W&B project is the project folder's name; use `group` for the stage or sweep, `name` and `tags` to tell runs apart.
- Local files go to `<RUNS_ROOT>/<project>/<run id>/` (the run id is the W&B id; gitignored): `checkpoints/` (best by the monitored metric plus `last.ckpt`), `config.json`, `csv/metrics.csv` (a local backup of the logged metrics) and `wandb/`.
- Log metrics as `train/…`, `val/…`, `test/…`, with `lr-…`, `perf/…` and `stats/…` coming from the callbacks. Score the test set once, with the checkpoint chosen on validation (`trainer.test(..., ckpt_path="best")`).
- Log images and other media to W&B (`wandb.Image`), not to PNG files in the run directory. Plots saved as files are PNGs; never call `plt.show()`.
- Credentials come from the environment (`WANDB_API_KEY`, `WANDB_ENTITY`, and `WANDB_MODE=offline` when unreachable; upload later with `wandb sync <dir>`). Never write a key into the repo. `wandb login` is interactive, so the user runs it.
- `runs/` is a symlink to `/mnt/ai/zoo/runs` (the big disk) and `data/` (`DATA_ROOT`) to `/mnt/ai/zoo/data`; the local disk is small. HF downloads already go to `/mnt/ai/data/hf` (`HF_HOME`). Cheap CPU checks that make a run directory point `ZOO_RUNS_ROOT` at the scratchpad.

## Git

- Use atomic conventional commits, with the project as the scope when there is one: `feat(pixel_art): ...`, `docs(digitnet): ...`, `build: ...`.
- Commit only when asked.
- Never create a new branch. Always work on `main`.
- Commits are SSH-signed. Don't use history-rewriting tools that strip signatures (e.g. `git filter-branch`); if you rewrite history, re-sign it.
