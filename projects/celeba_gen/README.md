# celeba_gen

Image generation on CelebA-HQ faces. This folder fetches and checks the data and embeds the images with a
vision-language encoder; the generator itself comes next.

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

## Next

- Train an image generator on the `train` split. Candidates are a latent diffusion model (reusing the `pixel_art` /
  `digitdreamer` setup at 256 or 512 px) or a GAN in the StarGAN lineage.
- Evaluate on `val` with FID and, for conditioning, the class labels.
