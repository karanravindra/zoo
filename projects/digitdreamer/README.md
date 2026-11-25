# DigitDreamer

Class-conditioned digit generation in two stages: a conv autoencoder squeezes 1x32x32 MNIST digits
(28x28 padded by 2) into a `latent_channels`x4x4 latent, then a rectified flow learns to turn Gaussian
noise into those latents, conditioned on the digit (with classifier-free guidance).

| | |
| --- | --- |
| W&B project | `digitdreamer`, group `ae` or `flow` |
| Data | MNIST or QMNIST (`--dataset` on `prepare` and `ae`) via `zoo.data` (cached under `DATA_ROOT`) |
| Local run files | `<RUNS_ROOT>/digitdreamer/<run id>/` |
| Extra dependencies | none |

```
uv run projects/digitdreamer/cli.py prepare                          # download and cache MNIST (`--dataset QMNIST` for QMNIST)
uv run projects/digitdreamer/cli.py ae                               # stage 1
uv run projects/digitdreamer/cli.py flow --ae-ckpt <AE best checkpoint>  # stage 2
uv run projects/digitdreamer/cli.py sample <flow checkpoint>         # a grid of digits
```

`cli.py <command> --help` lists a command's flags. The training commands take every config field as a flag; add
`--wandb offline` for no network, `--dev` for a throwaway run with no logging or checkpoints, `--ema 0.999` for averaged
weights, and `--lr-schedule constant` for no decay.

| Command | What it does |
| --- | --- |
| `prepare` | Downloads and caches MNIST. |
| `ae` | Stage 1. Trains the autoencoder. Logs `train/mse`, `val/` and `test/` MSE, PSNR, SSIM and `mse_global` (the MSE with the detail half of the latent masked), and a reconstruction grid every epoch. Checkpoints the best by `val/mse`. |
| `flow` | Stage 2. Trains the flow on the frozen AE's latents (`--ae-ckpt` is required). Logs `train/loss`, `val/loss`, `test/loss` and a class-by-sample grid every epoch (rows = digits 0-9). Checkpoints the best by `val/loss`. |
| `sample` | Generates digit grids from a flow checkpoint. |

Files:
- `cli.py`: the one entry point; each command is a short function.
- `config.py`: every command's settings (`AEConfig` and `FlowConfig` on top of `zoo.training.RunConfig`, plus `SampleConfig`, `PrepareConfig`).
- `ae.py`, `flow.py`: one file per stage, each with the network, its loss/sampler and its `LightningModule`.
- `data.py`: `ImageData` (padded MNIST in [-1, 1]) and `LatentData` (the AE's latents, encoded once).
- `sweeps/ae_latent.yaml`: a W&B sweep over the latent size and mask probability (the commands to start it are in the file).

Notes:
- A flow checkpoint is self-contained: the module owns the AE and the latent statistics, so
  `load_module(LitFlow, path)` is all sampling needs. It applies the EMA weights when the run had `--ema`, which is the default for the flow.
- The flow's splits are the AE's own, so it never trains on an image the AE validated on.
- Images live in [-1, 1]; flow time runs from t=0 (noise) to t=1 (data).
- Comparing AE runs (the old `compare_ae.py`) is done in W&B: group by `ae`, and chart `val/mse` against the config columns.
