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
| `train_ldm.py` | Stage 2. Trains the latent diffusion model (denoiser + text encoder) on frozen-AE latents. Uses `LDMConfig.ae_ckpt`, or the latest AE run's `best.pt` if it is `None`. Writes `best.pt`, `last.pt`, `results.json` and per-epoch sample grids (real sprite, then 4 samples, per validation caption). |
| `sample.py` | Draws sprites for `--prompt` captions from an ldm checkpoint (default: latest ldm run's `last.pt`); `--guidance` sets the classifier-free guidance scale. |

Modules: `ldm.py` (configs, autoencoder, text encoder, denoiser, loss, sampler, checkpoints), `pixel_data.py` (cache, tokenizer, loaders),
`common.py` (paths, metrics, run dirs, plots).

Notes:
- The whole dataset (~1.4 GB as uint8) is held on the device, so there are no DataLoaders.
- Splits: `val_size` train sprites are held out for validation; the official validation split is the test set.
- LDM checkpoints are self-contained: they bundle the AE, tokenizer and latent statistics, so `load_ldm(path)` is all sampling needs.
- Images live in [-1, 1] (RGBA, RGB zeroed where alpha is 0); diffusion time runs from t=0 (noise) to t=1 (data).
- Grids are composited over a checkerboard and upscaled 8x. Alpha is modelled as a plain fourth channel; revisit
  once `stats.py` shows whether alpha is binary.
