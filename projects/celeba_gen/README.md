# celeba_gen

Image generation on CelebA-HQ faces. This folder fetches and checks the data, embeds the images with a
vision-language encoder, and trains a TiTok-style 1D tokenizer in three phases that change the resolution (stage 1 of a generator); the generator itself comes next.

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
uv run marimo edit projects/celeba_gen/explore.py
```

`dtour`, `marimo`, `polars` and `wigglystuff` are regular dependencies of the `zoo` package, like everything else here.

## Tokenizer

A TiTok-style 1D tokenizer (Yu et al., "An Image is Worth 32 Tokens for Reconstruction and Generation") with FSQ,
trained in three phases where the resolution changes. A ViT encoder turns an image into `num_tokens` (default 64)
discrete tokens, each a codebook index (default FSQ levels 8,8,8,5,5,5, so 64,000 codes), and a ViT decoder maps them
back to pixels. The token count does not depend on the image size, so the same 64 tokens describe 4x the pixels at 256 px
that they do at 128 px. That is what phase 2 adapts the model to. Default 22.1M parameters. Files:

- `celeba.py`: `CelebAHQ256`, an `ImageDataset`. The first construction resizes every image to 256x256 (bicubic)
  and caches it as uint8 under `data/cache/CelebAHQ256/{train,test}/`, about 5.9 GB. The size is in the class name,
  so a different resolution needs a different class and cache. `make_splits` holds out 1,000 train images for
  validation; the official 2,000-image `val` split is the test set. `load_split` reads a split into one uint8 tensor
  in device memory (5.3 GB for the train split) and `Batches` iterates it as normalized float batches on that device;
  together they replace the `DataLoader` in the training scripts.
- `titok.py`: `TiTokConfig`, the model, `PatchDiscriminator`, `set_trainable`, and `save_titok` / `load_titok`. The
  encoder patchifies the image (patch 16), appends the learned latent tokens, runs a ViT and keeps the latent tokens,
  which FSQ quantizes (no codebook, no commitment loss: the losses are reconstruction only). The decoder appends the
  projected tokens to one learned mask token per patch, runs a ViT and unpatchifies the mask tokens.
  `model.tokenize(x)` gives the (B, K) indices and `model.detokenize(idx, size)` decodes them.
- `rfid.py`: rFID (Inception features, running statistics, Fréchet distance).
- `train_titok.py`: one subcommand per phase. The phases are one loop with different defaults, so every flag below works
  on every phase.
- `compare_titok.py`: plots runs side by side, labelled by the config values that differ (see below).

**Resolution.** Only the patch grid depends on the image size (H/16 x W/16, any aspect ratio). The position embeddings
are one learned 16x16 grid, bilinearly resized (antialiased when it shrinks) to each image's patch grid, so a position
keeps its place in the image at any resolution; the latent tokens have their own learned embeddings. Any image side that
is a multiple of the patch size works with no change to the weights, and `model.decode(z, (H, W))` can decode at a
different size than the tokens were encoded from. This is the position-grid part of VibeToken's recipe (Patel et al.,
CVPR 2026); its adaptive patch embedding is not in here, so the patch count grows with the image area.

| phase | trains | resolution | losses | defaults |
|---|---|---|---|---|
| `phase1` | everything | 128 (8x8 patches) | L1 + LPIPS | 20 epochs, lr 2e-4 one-cycle, batch 64; `--model.*` sets the architecture |
| `phase2` | `middle`: encoder's last 2 blocks, norm and projection, decoder's input projection, position grid and first 2 blocks | 256 (16x16 patches) | L1 + LPIPS | 5 epochs, lr 1e-4 constant, batch 32 |
| `phase3` | `decoder-head`: decoder's last 2 blocks, norm and output layer | 128 | L1 + LPIPS + PatchGAN (hinge, weight 0.05) | 5 epochs, lr 5e-5 constant, betas (0.5, 0.9) |

Phase 2 changes how the tokens summarize a larger patch grid without training every layer: the encoder's patch embedding,
position grid and first blocks stay frozen, so autograd keeps none of their activations. Phase 3 only touches the decoder
head, so the tokens do not move and the GAN loss can run at low resolution, never on high-resolution images.

```sh
uv run projects/celeba_gen/train_titok.py phase1 --dev --epochs 1   # smoke run: temp dir, nothing kept
uv run projects/celeba_gen/train_titok.py phase1                    # runs/celeba_gen/titok-phase1/<timestamp>/
uv run projects/celeba_gen/train_titok.py phase1 --name muon-a --optimizer muon   # runs/celeba_gen/titok-phase1/muon-a/
uv run projects/celeba_gen/train_titok.py phase2                    # starts from the latest phase1 best.pt
uv run projects/celeba_gen/train_titok.py phase3                    # starts from the latest phase2 best.pt
uv run projects/celeba_gen/train_titok.py phase2 --init runs/celeba_gen/titok-phase1/<run>/last.pt
uv run projects/celeba_gen/train_titok.py phase1 --model.num-tokens 32 --model.levels 8 5 5 5
uv run projects/celeba_gen/train_titok.py phase2 --help
```

Compare runs:

```sh
uv run projects/celeba_gen/compare_titok.py --phase 1            # all runs under runs/celeba_gen/titok-phase1 -> runs/celeba_gen/titok-phase1_comparison.png
uv run projects/celeba_gen/compare_titok.py <p1 run> <p2 run> <p3 run>   # one chain, any mix of phases -> runs/celeba_gen/titok_comparison.png
```

The plot has train loss, validation rFID / MSE / PSNR / SSIM / LPIPS curves and test rFID / LPIPS / PSNR / SSIM bars. Colour is the run;
line style (curves) and hatch (bars) is the evaluation resolution, and the thick line is the run's training resolution.
On a chain, the 256 px lines of phase 1 against phase 2 show the penalty and its fix. Runs still training get curves but
no bars.

`lpips` and `torch-fidelity` are regular dependencies too. LPIPS downloads the torchvision weights of
the chosen backbone on first use (VGG16 is about 528 MB) and rFID downloads the Inception weights (about 100 MB), both to
`TORCH_HOME` (`~/.cache/torch` unless set), so point it at the big disk if the local one is tight.

**rFID is the evaluation metric.** Every evaluation (each epoch's validation, the epoch-0 row, the final test) computes the
Fréchet distance between Inception features of the images and of their reconstructions (`rfid.py`: torch-fidelity's port of
the TF Inception used for the standard FID, images quantized to uint8, statistics in float64), at every evaluation
resolution. The reconstructions are decoded from the quantized tokens. `best.pt` is the epoch with the lowest validation
rFID at the training resolution (`--best-by`). Validation has 1,000 images and test 2,000, far fewer than the usual 50k, so
the score is biased upwards and noisy: compare runs on the same split, not against published numbers. Raise `--val-size`
for a steadier validation score, at the cost of training images. `--no-rfid` skips it (and needs a `--best-by` other than
`rfid`). The validation images never change, so their Inception statistics and the square root of their covariance are
computed in a run's first evaluation and reused by every later one (the final test pass computes its own); each epoch
only the reconstructions go through Inception.

What each phase trains, and with what, is set by flags:

- `--train`: `all`, `middle`, `decoder-head`, or parameter-name prefixes such as `encoder.blocks.5 decoder.out`.
  Names are `encoder.{patch_embed,pos,latents,blocks.i,norm,proj}` and
  `decoder.{proj,latent_pos,mask,pos,blocks.j,norm,out}`; `decoder.blocks.0` is the one next to the tokens.
  `--edge-blocks N` sets how many blocks `middle` and `decoder-head` cover (default 2). A prefix that matches nothing is
  an error.
- `--resolution`, `--eval-resolutions` (default 128 256; all are logged every epoch next to the training resolution).
  Every resolution must be a multiple of the patch size (16). The cache is 256 px, so a lower resolution is an
  antialiased bicubic downscale on the GPU and 256 is the maximum.
- `--l1-weight`, `--mse-weight`, `--lpips-weight`, `--gan-weight` (> 0 adds the discriminator: `--disc-lr`,
  `--disc-width`). `--lpips-net` (`vgg`, `alex` or `squeeze`, default `vgg`) picks the LPIPS backbone for both the loss
  and `val_lpips`; the values differ between backbones, so use the same one in every phase you want to compare.
- `--optimizer` (`adamw`, the default, or `muon`) with `--lr`, `--betas`, `--weight-decay`, `--muon-momentum`.
  With `muon`, torch's Muon trains the 2D weight matrices inside the transformer blocks (attention and MLP) and AdamW
  trains everything else (patch embedding, position grids, latent and mask tokens, norms, biases, and `encoder.proj`,
  `decoder.proj`, `decoder.out`), since Muon only takes 2D matrices and is meant for hidden layers. Muon runs with
  `adjust_lr_fn="match_rms_adamw"`, so `--lr` and `--weight-decay` apply to both optimizers and the AdamW values carry
  over as a starting point; `--betas` then only reaches the AdamW part. If `--train` leaves no block matrix trainable,
  the run is plain AdamW and says so.
- `--lr-decay`, `--epochs`, `--batch-size`, `--ema`, `--grad-clip`, `--best-by`
  (`rfid`, `mse`, `lpips` or `last`; default `rfid`), `--rfid` / `--no-rfid`, `--init`, `--compile` (tokenizer only),
  `--bf16` / `--no-bf16`, `--dev`.
- `--bf16` (on by default, cuda only; `--no-bf16` for fp32): bfloat16 autocast around the tokenizer, the LPIPS loss and the
  discriminator, in training and in the model call of every evaluation. Weights, gradients, optimizer state and EMA stay fp32
  (so no loss scaling), the losses are computed in fp32 on the cast-back reconstruction, and the LPIPS and Inception
  metrics always run in fp32. LayerNorm and the residual stream stay fp32 under autocast; only the matmuls and attention
  are bf16. Runs from before this flag were fp32: an fp32 and a bf16 run differ slightly even with the same seed, and
  `compare_titok.py` labels them by `bf16` (`-` for the old ones).
- Data: the training script reads the uint8 train and validation images into device memory once (5.3 GB + 0.2 GB; the
  test images are read after training) and batches by indexing, so there are no loader processes (`--num-workers` is
  gone) and no per-sample transform. The values equal the old default transform (to float rounding), but the shuffle
  has its own generator, so the batch order differs from runs made with the `DataLoader`. The images always live on
  the training device: there is no host-memory fallback.
- `--name`: the run's directory name instead of the timestamp, e.g. `--name muon-a`. It must be a plain name without
  `/`. If the directory exists, `-2`, `-3`, ... is appended, so an existing run is never written into. `--dev` ignores
  it. Each phase takes its own `--name`; phases 2 and 3 still start from the most recently written `best.pt` of the
  previous phase whatever its directory is called, so chain a specific run with `--init`. `compare_titok.py` shows the
  directory name on the bars and in its table, and in the legend when no config value tells the runs apart.
- `--skip-duplicate` (on by default; `--no-skip-duplicate` to train anyway): before creating a run directory, look for a
  finished run (one with `results.json`) of the same phase with the same config and the same init checkpoint, and exit
  without training if there is one, so a sweep loop in `scratchpad.sh` can be restarted and only runs what is missing.
  `--name`, `--compile`, `--log-every` and `--dev` do not count, `--init` is compared as the resolved checkpoint path
  (so the default and the same path passed explicitly are the same run), and the git commit and device are not compared.
  A crashed run has no `results.json` and does not block a rerun. A run whose saved config has other keys than today's
  (from before a flag was added or removed) is a different config and runs again. `--dev` never skips. Same for DC-AE.

Each run writes to `runs/celeba_gen/titok-phase<N>/<timestamp>/` (or `<name>/`):

```
config.json  console.log  metrics.csv  best.pt  last.pt  results.json  loss.png  rfid.png  psnr.png  ssim.png  lpips.png  recon.png  (gan.png)
```

`metrics.csv` has one row per epoch with the train losses and, for every evaluation resolution, `val_rfid_<res>`,
`val_mse_<res>`, `val_psnr_<res>`, `val_ssim_<res>` and `val_lpips_<res>`, plus the gradient and GPU columns. Phases 2 and 3 start with an
epoch-0 row, the init checkpoint before any training, so the curves show the high-resolution penalty phase 2 removes.
Checkpoints hold `model_config` and `train_config` (with the phase number), so `titok.load_titok(path)` rebuilds the model.

Not in v1: an adaptive patch embedding (needed to go above 256 px without the patch count growing with the area), a
resolution above 256 (needs a larger cache), an adaptive GAN weight (`--gan-weight` is fixed), and rFID over 50k images.
The default learning rates, step counts, token count and FSQ levels are guesses for this small setup, not tuned, and
nothing says Muon beats AdamW here yet: compare the two with `compare_titok.py`.

## DC-AE baseline

`dcae.py`, `train_dcae.py` and `compare_dcae.py` are the earlier convolutional model (a DC-AE-style autoencoder, f32,
22.5M parameters) on the same three phases and the same flags, with `--model.downsample` and `--edge-stages` in place of
the TiTok model's. Its runs go to `runs/celeba_gen/dcae-phase<N>/`. They share `celeba.py` and `rfid.py` with the
tokenizer and are otherwise independent of it. `--bf16` works the same way, but the convolutional residual stream is bf16
end to end (autocast keeps only GroupNorm in fp32), so if its PSNR plateaus below an fp32 run's, compare with `--no-bf16`
first. The images live on the GPU here too, and the convolutional activations leave little room next to them: on the
16 GB RTX 5070 Ti, DC-AE at the default `--batch-size 64` ran out of memory with the images resident (training alone
peaked at 13.3 GiB, plus 5.3 GB of images). A smaller `--batch-size`, such as 32, should fit but has not been tried.

## Next

- Train the generator on the tokenizer's tokens: 64 discrete codes per image (`model.tokenize` gives the (B, 64) indices,
  `model.detokenize(idx, size)` decodes them). Candidates are a masked-token or autoregressive transformer over the
  indices, a diffusion model over the quantized (64, 6) codes, or a GAN in the StarGAN lineage.
- Evaluate the generator on `val` with FID (`rfid.py` has the Inception features and the Fréchet distance) and, for
  conditioning, the class labels.
