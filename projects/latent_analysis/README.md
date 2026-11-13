# latent_analysis

Explore the StarGAN v2 datasets (AFHQv2, CelebA-HQ, or both combined as `stargan`) through DINOv3
embeddings: cluster them, project them to 2-D, browse and tag images in a web explorer, and get
tag suggestions from linear probes.

Run every script with `uv run projects/latent_analysis/<script>.py --dataset <name>` (add `--model`
and `--image-size` consistently across steps if you change them). Each script's docstring lists its
full options and outputs.

## Pipeline

| Step | Script | Output |
| --- | --- | --- |
| 1 | `embed.py` | DINOv3 embeddings per split in `data/latents/`. The checkpoints are gated on Hugging Face: accept the license and `hf auth login` first. |
| 2 | `kmeans.py --k N` | Clusters (e-means by default, or `--method spherical`), summaries and `clusters.png`. |
| 3 | `visualize.py` | PCA, t-SNE and UMAP plots; add `--k N` to color by cluster. |
| 4 | `export.py` | Thumbnails and per-split JSON for the explorer. Re-run only when embeddings or projections change. |
| 5 | `bun run projects/latent_analysis/explorer/server.ts` | The explorer (`explorer/`: `server.ts`, `app.ts`, `index.html`). Tags are saved from here. |
| 6 | `autotag.py` | Suggests tags with linear probes trained on accepted and rejected tags. Re-run after each review round. |

Outputs go to `<repo>/runs/latent_analysis/<dataset>/<model>_<size>/`.

## Helpers

- `emeans.py`: single-file PyTorch e-means (mini-batch k-means with EMA updates). Run it directly for a comparison against Lloyd's k-means.
- `image_index.py`: stable integer image ids plus the tag files (`index.json`, `labels.json`, `rejections.json`), kept at the dataset level so tags are shared across models, splits and k.
