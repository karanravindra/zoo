# pixel_art

Text-to-sprite generation on [Angais/pixel-art-atlas-1.4m](https://huggingface.co/datasets/Angais/pixel-art-atlas-1.4m)
(1.4M synthetic 16x16 RGBA sprites with captions), a latent diffusion model in two stages, structured like `digitdreamer`: a first-stage autoencoder squeezes
each 4x16x16 sprite into a `latent_channels`x4x4 latent, then a text-conditioned denoiser is trained to turn
Gaussian noise into those latents (with classifier-free guidance at sampling time).

What differs from digitdreamer: sprites have an alpha channel, and the condition is free text. A word-level
tokenizer feeds a small text encoder trained from scratch together with the denoiser; the denoiser is a DiT whose
blocks cross-attend to the caption tokens. Training uses the rectified-flow (velocity) parameterization.

Run from anywhere with `uv run`. Every run writes to `<repo>/runs/pixel_art/<stage>/<run>/`.

| Script | What it does |
| --- | --- |
| `stats.py` | Dataset statistics (subjects, captions, alpha, colors, duplicates) to `runs/pixel_art/stats/`. |
| `pixel_data.py` | Builds the cache in `data/cache/PixelArtAtlas/` (uint8 RGBA, token ids, vocab). Run once: `uv run --with pyarrow projects/pixel_art/pixel_data.py`. |
| `train_ae.py` | Stage 1. Trains the first-stage autoencoder. Writes `best.pt`, `last.pt`, `results.json` (test MSE/PSNR/SSIM), curves and `recon.png`. |
| `compare_ae.py` | Plots AE runs side by side, labelled by the config values that differ. |
| `train_ldm.py` | Stage 2. Trains the latent diffusion model (denoiser + text encoder) on frozen-AE latents. Uses `LDMConfig.ae_ckpt`, or the latest AE run's `best.pt` if it is `None`. Writes `best.pt`, `last.pt`, `results.json` and per-epoch sample grids (real sprite, then 4 samples, per validation caption). |
| `sample.py` | Draws sprites for `--prompt` captions from an ldm checkpoint (default: latest ldm run's `last.pt`); `--guidance` sets the classifier-free guidance scale. |

Modules: `ldm.py` (configs, autoencoder, text encoder, denoiser, loss, sampler, checkpoints), `pixel_data.py` (cache, tokenizer, loaders),
`common.py` (the RGBA `save_grid`); run dirs, metrics.csv, plots, PSNR/SSIM (window 7 for 16x16 images) and EMA come from `zoo.runs`, `zoo.plot`, `zoo.metrics`, `zoo.train`.

Notes:
- The whole dataset (~1.4 GB as uint8) is held on the device, so there are no DataLoaders.
- Splits: `val_size` train sprites are held out for validation; the official validation split is the test set.
- LDM checkpoints are self-contained: they bundle the AE, tokenizer and latent statistics, so `load_ldm(path)` is all sampling needs.
- Images live in [-1, 1] (RGBA, RGB zeroed where alpha is 0); diffusion time runs from t=0 (noise) to t=1 (data).
- Grids are composited over a checkerboard and upscaled 8x.
- Alpha is exactly binary (0 or 255) in the data. With `--binary-alpha` (default on), the AE decodes alpha as an
  opacity logit trained with BCE (weighted by `--alpha-weight`), the RGB loss only counts opaque pixels, and
  `Autoencoder.to_image` thresholds alpha and blanks RGB where transparent. Every consumer of decoder output (AE
  metrics, recon grids, LDM sampling) goes through `to_image`. AE checkpoints saved before the flag existed load
  with `binary_alpha=False` (plain 4-channel MSE). `train_loss` in `metrics.csv` is the objective; the `*_mse`,
  PSNR and SSIM columns are on decoded images, so they stay comparable across both modes.
- `train_ae.py --grad-clip` (default 1.0, 0 disables) clips the global gradient norm. `metrics.csv` logs the
  pre-clip norm per epoch (`grad_norm` mean, `grad_norm_max`) and `clip_frac`, the share of steps that were clipped.
  Without clipping, binary alpha at lr 2.5e-3 had loss spikes at the LR peak and ended 0.7 dB below plain MSE.
- `train_ae.py --hflip-prob p` mirrors each training sprite left-right with probability `p` (default 0). Val and
  test are never flipped, so val/test metrics measure the unflipped distribution. Not used for the LDM: sprites
  are mostly lit from the top-left, so flips would change the style it learns.

## Dev runs

Pass `train_ae.py` / `train_ldm.py` `--dev` to try a change without logging anything: the usual outputs go to a temporary directory that is deleted
when the script exits, so the run never appears under `runs/`.

## Training diagnostics in metrics.csv

Besides the columns above, every row has gradient and GPU columns (blank where a value was not measured):

- `grad_norm`, `grad_norm_max`: mean and largest global gradient norm (before clipping) since the previous row;
  `clip_frac`: fraction of steps clipped, when the script clips.
- `weight_norm`, `update_ratio` (size of one optimizer step relative to the weights), `grad_cos` (cosine between
  consecutive steps' gradients: near 1 steady, near 0 noisy, negative oscillating), averaged over every 100th step.
- `gpu_util` (mean %), `gpu_mem_gb` (peak device memory), `gpu_alloc_gb` (peak memory torch allocated),
  `gpu_reserved_gb`, `gpu_power_w` (mean), `gpu_temp_c` (peak), sampled in the background while training.
