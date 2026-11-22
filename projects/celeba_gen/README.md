# celeba_gen

Image generation on CelebA-HQ faces. This folder fetches and checks the data, embeds the images with a
vision-language encoder, and trains a convolutional autoencoder at 256x256 (stage 1 of a generator); the generator itself comes next.

## Data

`prepare.py` downloads CelebA-HQ from the StarGAN v2 release, via `zoo.data.stargan`, into `data/celeba_hq/`:

```
data/celeba_hq/
  train/<class>/<image>
  val/<class>/<image>
```

The StarGAN release has 28,000 train and 2,000 val images at 1024x1024 (checked with `prepare.py`). Generators here
should pick a training resolution (256 or 512 px) and resize from these originals.

CelebA-HQ and CelebA are released for non-commercial research use. Check their terms before using the weights anywhere else.

## Prepare

```sh
uv run projects/celeba_gen/prepare.py          # download (once) and print the counts
uv run projects/celeba_gen/prepare.py --force  # delete and re-download
uv run projects/celeba_gen/prepare.py --help
```

Run it on the machine with the big disk, since `data/` should point at `/mnt/ai`.

## Embed

`embed.py` embeds every image in `train` and `val` with a vision-language encoder (default `siglip2-base-256`, from
`google/siglip2-base-patch32-256`), plus one text embedding per class from `--prompt-template`. Features are unit-norm,
so a dot product is cosine similarity. The weights and float inputs run in bf16; features are saved as float32.

```sh
uv run projects/celeba_gen/embed.py                          # default model; skips files that already exist
uv run projects/celeba_gen/embed.py --model siglip2-so400m   # another encoder: siglip2-base, siglip2-so400m, clip-l
uv run projects/celeba_gen/embed.py --help
```

Existing files are skipped, including `text.pt`. Delete `text.pt` before changing `--prompt-template`, or the old
prompts will be reused.

Output goes to `data/latents/celeba_hq/vl-<model>/`:

```
train.pt  val.pt  text.pt
```

Each `<split>.pt` holds `features` (N, D), `labels`, `classes` and `paths`. `text.pt` holds `features` (C, D),
`classes` and `prompts`. This is the same layout `projects/latent_analysis/embed_vl.py` writes, so the two projects
share the files.

## Explore

`explore.py` is a marimo notebook with two linked views of the embeddings, both over the top 8 PCA components and colored
by the folder labels. It also runs zero-shot classification against the `text.pt` prompts.

- **Tour and UMAP:** a dtour tour through the PCs next to a UMAP. Lasso in either view, or click a class in the legend.
  The faces appear below, with how often the zero-shot labels agree with the folder labels. The UMAP is cached as
  `umap-2d.npy` next to the embeddings on first run.
- **Pair plot:** a `wigglystuff` `PairPlot` scatterplot matrix with box and lasso selection over every face, and the
  same summary for the selection.

```sh
uv run --group celeba_gen marimo edit projects/celeba_gen/explore.py
```

The `celeba_gen` dependency group holds `dtour`, `marimo`, `polars` and `wigglystuff`.

## Autoencoder

Stage 1 of a latent generator: a convolutional autoencoder that maps each 3x256x256 face to a 4x32x32 latent
(8x per side, 48x fewer values). Files:

- `celeba.py`: `CelebAHQ256`, an `ImageDataset`. The first construction resizes every image to 256x256 (bicubic)
  and caches it as uint8 under `data/cache/CelebAHQ256/{train,test}/`, about 5.9 GB. The size is in the class name,
  so a different resolution needs a different class and cache.
- `autoencoder.py`: `AEConfig`, the model (three stride-2 stages with GroupNorm and SiLU, width 32, 3x3 convs, about
  1.2M parameters), and `save_ae` / `load_ae` checkpoints.
- `train_ae.py`: the training loop. Loss is MSE on [-1, 1] images, AdamW with a one-cycle learning rate, gradient
  clipping at 1.0, and random left-right flips (p=0.5) on the training images only.

Splits: 1,000 train images are held out for validation, and the official 2,000-image `val` split is the test set.

```sh
uv run projects/celeba_gen/train_ae.py --dev --epochs 1   # smoke run: outputs go to a temp dir, nothing is kept
uv run projects/celeba_gen/train_ae.py                    # full run: runs/celeba_gen/ae/<timestamp>/
uv run projects/celeba_gen/train_ae.py --help
```

The first run builds the cache, which decodes all 30,000 originals and takes a while. Later runs read it from disk.

Each run writes to `runs/celeba_gen/ae/<timestamp>/`:

```
config.json  metrics.csv  best.pt  last.pt  results.json  curves.png  psnr.png  ssim.png  recon.png
```

`metrics.csv` has one row per epoch: `train_mse`, `train_psnr`, `val_mse`, `val_psnr`, `val_ssim`, the gradient and
GPU columns, and the learning rate. `best.pt` is the epoch with the lowest validation MSE. `results.json` holds the
test MSE, PSNR and SSIM for `best.pt`. `recon.png` shows 16 test faces (top row) and their reconstructions (bottom row).

Checkpoints are self-contained: `autoencoder.load_ae(path)` returns the model, its config and the checkpoint dict.
The latest run's `best.pt` is `zoo.runs.latest_checkpoint("celeba_gen", "best.pt", stage="ae")`.

## Next

- Train the generator on the autoencoder's latents. Candidates are a latent diffusion model (the `digitdreamer`
  rectified flow, on 4x32x32 latents) or a GAN in the StarGAN lineage.
- Evaluate on `val` with FID and, for conditioning, the class labels.
