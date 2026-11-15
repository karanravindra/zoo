# zoo

Personal ML experiments. Shared library code lives in `src/zoo/`; each experiment is a project under `projects/<name>/`.

## Running code

- Never run training, embedding, download, or other pipeline scripts unless the user says it's okay. That okay lasts for the rest of the session. Without it, write the code and hand it off with the exact command for the user to run.
- Cheap checks don't need permission: imports, `--help`, a tiny CPU forward pass.
- Use uv for everything: `uv run ...` and `uv add ...`. Never use pip.

## Layout

- Code reused across projects goes in `src/zoo/`. Utilities only; training loops, scripts and models stay in the project.
  - `zoo.paths`: `REPO_ROOT`, `DATA_ROOT`, `runs_root(project)`.
  - `zoo.runs`: `git_commit`, `new_run_dir(project, config, device, stage=None, dev=False, name=None, **extra)`, `dev_run_dir(project)`, `latest_checkpoint(project, name, stage=None)`, `MetricsLogger` (metrics.csv); `load_run`, `read_metrics`, `diff_labels`, `sort_runs` for a project's compare script.
  - `zoo.plot`: `plot_curves(rows, series, path, x="epoch")`; importing it selects the Agg backend (`from zoo.plot import plt`).
  - `zoo.metrics`: `to_unit`, `psnr`, `ssim` (images in [-1, 1]; pass `size=7` for 16x16), bits-per-byte for text.
  - `zoo.train`: `update_ema`, `Throughput` (steps/s timing), `ParamStats` (gradient norm, clipping, weight norm, update ratio, gradient direction; `STAT_FIELDS`).
  - `zoo.monitor`: `GpuMonitor` (GPU utilisation, memory, power, temperature; `GPU_FIELDS`).
  - `zoo.data`: `ImageDataset`/`MNIST`/`QMNIST` (`data/image.py`), token corpora on disk (`data/text.py`).
  - `zoo.utils`: device and seeding.
- Scripts import sibling modules directly (`from common import ...`); Python already puts the script's directory on `sys.path`, so no `sys.path` hacks.
- Dependencies used by one project only live in a uv dependency group named after it (`uv run --group ledger ...`).
- Each project lives in `projects/<name>/` with a `README.md` and is otherwise self-contained. Keep the README in sync when scripts, outputs, or configs change.

## Script config

- Every runnable script builds its settings from a dataclass and parses them with [tyro](https://brentyi.github.io/tyro/), so the dataclass fields define the CLI flags. Don't use argparse.
- Pattern: `main(tyro.cli(Config))`. Defaults live in the dataclass, and a field's trailing comment is its `--help` text.
- Every script that runs a model has a `compile: bool = False` field. When it's set, call `model.compile(mode="reduce-overhead", dynamic=False)` in place (not `torch.compile(model)`, which adds `_orig_mod.` to the checkpoint keys), after `torchinfo.summary` and after any EMA `deepcopy`. Keep `dynamic=False`: with dynamic shapes, Python floats become float64 graph inputs, which crash Inductor on MPS and slow it down even when they don't. Varying shapes and float values just trigger recompiles instead.
- Every training script has a `dev: bool = False` field for throwaway runs: pass `dev=cfg.dev` to `zoo.runs.new_run_dir` (or use `zoo.runs.dev_run_dir`), which swaps the run directory for a temporary one deleted on exit, so nothing is logged to `runs/`.
- Training configs are saved into checkpoints, so they should stay plain, serializable dataclasses.

## Run outputs

- Each run writes to `runs/<project>/<timestamp>/` (gitignored; a script can pass `name=` to `new_run_dir` for a descriptive directory name instead, as tinystories does): `config.json` (config plus git commit and device), `metrics.csv`, `best.pt` / `last.pt` checkpoints, and `results.json`.
- `runs/` is a symlink to `/mnt/ai/zoo/runs` (the big disk); use it through the symlink.
- Training scripts log `STAT_FIELDS` (from `ParamStats`) and `GPU_FIELDS` (from `GpuMonitor`) in `metrics.csv`. A column that is only measured on some rows is left blank on the others; `zoo.runs.read_metrics` reads blanks as nan and `plot_curves` skips them.
- Save plots as PNG files. Never call `plt.show()`.

## Git

- Use atomic conventional commits, with the project as the scope when there is one: `feat(pixel_art): ...`, `docs(digitnet): ...`, `build: ...`.
- Commit only when asked.
- Commits are SSH-signed. Don't use history-rewriting tools that strip signatures (e.g. `git filter-branch`); if you rewrite history, re-sign it.
