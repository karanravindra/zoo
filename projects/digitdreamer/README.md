# DigitDreamer

Class-conditioned digit generation in two stages: a conv autoencoder squeezes 1x32x32 MNIST digits
(28x28 padded by 2) into a `latent_channels`x4x4 latent, then a rectified flow learns to turn Gaussian
noise into those latents, conditioned on the digit (with classifier-free guidance).

Run from anywhere with `uv run`. Every run writes to `<repo>/runs/digitdreamer/<stage>/<run>/`.

| Script | What it does |
| --- | --- |
| `train_ae.py` | Stage 1. Trains the autoencoder. Writes `best.pt`, `last.pt`, `results.json` (test MSE/PSNR/SSIM), curves and `recon.png`. |
| `train_flow.py` | Stage 2. Trains the flow on frozen-AE latents. Uses `FlowConfig.ae_ckpt`, or the latest AE run's `best.pt` if it is `None`. Writes `best.pt`, `last.pt`, `results.json` (val/test loss) and per-epoch sample grids. |
| `sample.py` | Generates digit grids from a flow checkpoint (default: latest flow run's `last.pt`). |
| `compare_ae.py` | Plots AE runs side by side, labelled by the config values that differ. |

Modules: `dreamer.py` (configs, models, losses, sampling, checkpoints), `common.py` (MNIST transform, data splits); run dirs, metrics.csv, plots, PSNR/SSIM and EMA come from `zoo.runs`, `zoo.plot`, `zoo.metrics`, `zoo.train`.

Notes:
- Flow checkpoints are self-contained: they bundle the AE and latent statistics, so `load_dreamer(path)` is all sampling needs.
- "Latest" checkpoint means most recently written, so run directories may be renamed by hand (e.g. `latent4-mask0.5`).
- Images live in [-1, 1]; flow time runs from t=0 (noise) to t=1 (data).

## Dev runs

Pass `train_ae.py` / `train_flow.py` `--dev` to try a change without logging anything: the usual outputs go to a temporary directory that is deleted
when the script exits, so the run never appears under `runs/`.

## Training diagnostics in metrics.csv

Besides the columns above, every row has gradient and GPU columns (blank where a value was not measured):

- `grad_norm`, `grad_norm_max`: mean and largest global gradient norm (before clipping) since the previous row;
  `clip_frac`: fraction of steps clipped, when the script clips.
- `weight_norm`, `update_ratio` (size of one optimizer step relative to the weights), `grad_cos` (cosine between
  consecutive steps' gradients: near 1 steady, near 0 noisy, negative oscillating), averaged over every 100th step.
- `gpu_util` (mean %), `gpu_mem_gb` (peak device memory), `gpu_alloc_gb` (peak memory torch allocated),
  `gpu_reserved_gb`, `gpu_power_w` (mean), `gpu_temp_c` (peak), sampled in the background while training.
