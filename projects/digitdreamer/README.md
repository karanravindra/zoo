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

Modules: `dreamer.py` (configs, models, losses, sampling, checkpoints), `common.py` (paths, data splits, metrics, run dirs, plots).

Notes:
- Flow checkpoints are self-contained: they bundle the AE and latent statistics, so `load_dreamer(path)` is all sampling needs.
- "Latest" checkpoint means most recently written, so run directories may be renamed by hand (e.g. `latent4-mask0.5`).
- Images live in [-1, 1]; flow time runs from t=0 (noise) to t=1 (data).
