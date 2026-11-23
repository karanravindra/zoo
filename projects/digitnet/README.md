# DigitNet

A small CNN (two conv blocks, global pooling, linear head) that classifies 28x28 digits. Trains on MNIST
by default; any dataset class in `zoo.data` with `image_shape` and `num_classes` works via `Config.dataset`.
The reference project for the Lightning + W&B layout described in `CLAUDE.md`.

| | |
| --- | --- |
| W&B project | `digitnet` |
| Data | MNIST via `zoo.data` (cached under `DATA_ROOT`) |
| Local run files | `<RUNS_ROOT>/digitnet/<run id>/` |
| Extra dependencies | none |

```
uv run projects/digitnet/train.py
uv run projects/digitnet/train.py --epochs 20 --lr 1e-3 --name lr1e-3   # every Config field is a flag; see --help
uv run projects/digitnet/train.py --lr-schedule constant --ema 0.999     # no lr decay, EMA weights for val/test
uv run projects/digitnet/train.py --wandb offline                       # no network; upload later with `wandb sync`
uv run projects/digitnet/train.py --dev --epochs 1                      # throwaway: no W&B, no checkpoints
```

Files:
- `config.py`: `Config`, the project's fields on top of `zoo.training.RunConfig`.
- `module.py`: `DigitNet` and `LitDigitNet` (`LightningModule`: loss/accuracy per split, AdamW with `--lr-schedule`, one-cycle by default).
- `data.py`: `DigitData` (`LightningDataModule`); the validation set is carved deterministically from the train split
  (`val_size`, `seed`) and only used to pick the checkpoint.
- `train.py`: the training script. Scores the test set once, with the best checkpoint.
- `sweeps/lr.yaml`: a W&B sweep over the learning rate (the commands to start it are in the file).

## What a run produces

In W&B: the config, git state, `train/`, `val/` and `test/` loss and accuracy, `lr-AdamW`, `perf/steps_per_s`,
`stats/` (gradient norm, plus weight norm, update ratio and gradient direction every 100 steps), and system metrics.

Locally, in `<RUNS_ROOT>/digitnet/<run id>/`:

| File | Contents |
| --- | --- |
| `checkpoints/` | the best checkpoint by `val/acc`, and `last.ckpt` |
| `config.json` | the full config |
| `csv/metrics.csv` | a local copy of the logged metrics |
| `wandb/` | W&B's own files |

Load a trained model with `LitDigitNet.load_from_checkpoint(path)`; the config is stored in the checkpoint.

## Credentials

`WANDB_API_KEY` (or `uv run wandb login`, run by you) and optionally `WANDB_ENTITY` come from the environment.
