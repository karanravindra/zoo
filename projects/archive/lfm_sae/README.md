# lfm_sae

Sparse autoencoders (SAEs) on [LiquidAI/LFM3-VL-2B](https://huggingface.co/LiquidAI/LFM3-VL-2B), for interpreting
its language model's activations.

The model is a 1.9B VLM. Its language model is LFM3: 24 layers, d_model 2048, mixing gated short convolutions
(13 layers), sliding-window attention (5) and global attention (6), with sandwich norm. The checkpoint ships its own
modeling code (`trust_remote_code`), so `common.MODEL_REVISION` pins the commit.

Needs a CUDA GPU. Run from the repo root with `uv run`.

| Script | What it does |
| --- | --- |
| `train_sae.py` | Trains a TopK SAE on activations generated on the fly (nothing on disk). Evaluates reconstruction, dead latents and loss recovered on held-out documents. |
| `features.py` | Interprets a trained SAE: runs the model over the held-out documents on the fly and records, per latent, firing frequency, top-activating contexts (one per document) and the output tokens its decoder direction promotes and suppresses. |
| `layout.py` | Maps for the viewer: a UMAP of each SAE's decoder directions and each latent's nearest directions in the run's other SAEs. |
| `viewer/` | Feature viewer (Bun): browse, search and map the latents of every SAE, with labels where they exist. |
| `extract.py` | Writes activations of a small held-out set to disk (default 5M tokens, about 20 GB), with token ids and doc/position/source per row, for browsing features and fast experiments. Resumable. |

Modules: `sae.py` (the TopK SAE), `common.py` (model id and revision, hook names, Hub shas, reading shards and
stats). `train_sae.py` reuses the data, model-loading and hook helpers in `extract.py`.

## Why on the fly

One hook point costs 4 KB per token (2048 x bf16), so 100M stored tokens is about 410 GB. The language model makes
about 30k tok/s at layer 12 on the RTX 5070 Ti (about 120 MB/s of activations), and the ZFS pool stalled writing at
that rate. Reading it back would be no faster than recomputing, so each SAE run regenerates its activations. The SAE
itself is cheap next to the language model.

## Data

About 90% [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) and 10%
[SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk), the same in both scripts.

- FineWeb-Edu is web text filtered for educational value. The prose is clean and topically broad, with little
  boilerplate, so features and their top-activating examples are easy to read.
- SmolTalk conversations are rendered with the model's own chat template (`<|im_start|>` roles, `<think></think>`
  before final answers). This is an instruction-tuned model, and chat-structure features (roles, answer starts,
  formatting) don't show up in web text.

The data is downloaded once into `HF_HOME` (about 6 GB) and memory-mapped, not streamed. `--web-files` defaults to
two of the 14 `sample/10BT` parquet files (about 690k documents each; 100M tokens need about 140k), and
`--chat-files` to all of SmolTalk `all` (1.0M conversations). Streaming shuffles parquet row groups across many
remote files, and its read buffers grew to about 30 GB of RAM. Each source gets a full seeded permutation, and a
seeded coin picks each document's source (`--chat-fraction`), so document k depends only on the settings. Both
dataset commits are resolved and recorded in `config.json`.

Each document starts with `<|startoftext|>` and is cut to `--context` 1024 tokens. The BOS position is never used:
its huge-norm activation would dominate SAE training. Padding isn't either.

**Held-out documents.** `train_sae.py` never trains on the first `--holdout-docs` (8192) documents of the stream,
about 5.2M tokens. `extract.py` with the same data settings starts at document 0, so a run of up to about 5M tokens
stores only held-out documents. Check `docs` in its `state.json` stays at or below 8192.

## Hook points

| Name | What it is |
| --- | --- |
| `resid_post.<i>` | output of decoder layer i: the residual stream after it |
| `mixer_out.<i>` | what the conv / attention block adds to the residual (after its post-norm) |
| `mlp_out.<i>` | what the feed-forward block adds to the residual (after its post-norm) |

`resid_post.i = resid_post.(i-1) + mixer_out.i + mlp_out.i`. The default is `resid_post.12` (middle layer). The
model is cut after the deepest hooked layer, so later layers and the output head don't run (except in the
loss-recovered eval).

## SAE

TopK SAE ([Gao et al. 2024](https://arxiv.org/abs/2406.04093)): `z = TopK(ReLU((x - b_dec) W_enc + b_enc), k)`,
`x_hat = z W_dec + b_dec`, unit-norm decoder rows. `k` sets the sparsity directly, so there is no L1 coefficient to
tune. Defaults: 16384 latents (8x), k = 64, batch 4096 tokens, Adam lr 2e-4 with 500 warmup steps and linear decay
over the last 20%, 100M tokens (24.4k steps).

- Inputs are scaled so that E‖x‖² = d_model (`input_scale`, estimated from the first buffer fill and stored in the
  SAE), and `b_dec` starts at the data mean.
- AuxK loss (weight 1/32): latents that haven't fired for `--dead-tokens` (10M) training tokens reconstruct the
  residual with their top `--k-aux` (1024) pre-activations, which gives dead latents a gradient.
- The shuffle buffer holds `--buffer-tokens` (2M, about 8.6 GB) activations in CPU RAM. Each round yields random
  batches from half of it (copied to the GPU, 16 MB per step) and refills those slots, so every token is used once.
- GPU memory goes to the model up to the hook layer, its forward pass and the SAE. The later layers stay in CPU RAM
  and move to the GPU only for the loss-recovered eval. The train log prints peak GPU memory.

Evaluation, every `--eval-every` steps (about 10M tokens) and at the end:

| Metric | Meaning |
| --- | --- |
| `fvu` | fraction of variance unexplained on `--eval-docs` (512) held-out documents (lower is better) |
| `dead_frac` | latents that fire on no held-out token |
| `ce_clean`, `ce_sae`, `ce_ablate` | full-model next-token loss on `--ce-docs` (64) held-out documents: clean, with the SAE reconstruction spliced in at the hook point, and with the activation replaced by its mean |
| `loss_recovered` | `(ce_ablate - ce_sae) / (ce_ablate - ce_clean)`: 1 means the SAE loses nothing |

## Workflow

```bash
# 1. Quick check of the trainer: about 5M tokens, evals every 2M.
uv run projects/lfm_sae/train_sae.py --num-tokens 5000000 --eval-every 500

# 2. Full run: 100M tokens of resid_post.12.
uv run projects/lfm_sae/train_sae.py

# Several sizes side by side on the same activations (the language model runs once): 8x, 16x, 32x.
# Each SAE uses lr / sqrt(n_latents / 16384). About 2.3 h for 100M tokens (the SAE steps become the limit).
uv run projects/lfm_sae/train_sae.py --n-latents 16384 32768 65536

# Other settings: --layer 6 --site mlp_out, --k 32.

# 3. What each latent responds to, on the 8192 held-out documents (about 5.2M tokens, 3 min).
uv run projects/lfm_sae/features.py --checkpoint runs/lfm_sae/train/<timestamp>/sae_16384/best.pt

# UMAP and cross-size neighbors for every SAE in the run, then the viewer (http://localhost:3002).
uv run projects/lfm_sae/layout.py --run runs/lfm_sae/train/<timestamp>
bun run projects/lfm_sae/viewer/server.ts

# 4. Held-out activations on disk, for feature browsing (5M tokens, about 20 GB on /mnt/ai).
uv run projects/lfm_sae/extract.py

# If an extraction stops (crash, Ctrl-C), continue it. The saved config is used; only --device and --compile
# come from the command line.
uv run projects/lfm_sae/extract.py --resume runs/lfm_sae/extract/<timestamp>
```

## Output

`train_sae.py` writes to `runs/lfm_sae/train/<timestamp>/`:

- `config.json`: config, hook, git commit, GPU name, dataset revisions.
- `metrics.csv`: per SAE (`n_latents` column), `split=train` rows every `--log-every` steps (lr, fvu, aux loss, dead fraction over the AuxK window,
  tokens/s) and `split=eval` rows (the metrics above).
- `sae_<n_latents>/best.pt` (lowest eval fvu) and `last.pt`: `{"sae": state_dict, "config", "n_latents", "hook",
  "step", "tokens", "eval"}`, one folder per SAE.
  Load with `TopKSAE(2048, n_latents, k).load_state_dict(ckpt["sae"])`. `sae.latents(x)` and `sae.reconstruct(x)`
  take raw activations.
- `results.json`: final eval per SAE (`saes.<n_latents>`), input scale, minutes, peak GPU memory. `curves.png`: fvu,
  dead latents and loss recovered over training, one color per SAE.

`features.py` writes to `features/` next to the checkpoint (`sae_<n>/features/`):

- `features.json`: per latent `freq` (fraction of held-out tokens it fires on), `mean_act`, `max_act`, `examples`
  (top activations at most one per document: `left` context, `token`, `right` context, `act`, `doc`, `pos`,
  `source`), and `promotes` / `suppresses` (top output tokens of `W_dec` through the final norm weight and the tied
  unembedding, the direct path only).
- `summary.txt`: frequency quantiles and a readable dump of `--sample` random latents, the activating token in
  `[[ ]]`.

`layout.py` adds `umap.json` (2-D position per latent) and `neighbors.json` (per other SAE in the run, the 5 most
similar decoder rows and their cosines) to each `sae_<n>/features/`. Hand-written labels go in `labels.json` there:
`{"<id>": {"label", "category", "confidence", "explanation"}}`, with `category` one of token, topic, grammar,
discourse, format, chat, code, math, dense, unclear; `notes.md` holds free-form findings. The viewer reads all of
these.

## Viewer

`bun run projects/lfm_sae/viewer/server.ts` (`--port`, default 3002; `--runs`, default `runs/lfm_sae/train`) serves
every SAE that has a `features.json`. Files are re-read when they change.

- **Top bar:** the run's SAE sizes, and the SAE's FVU, loss recovered, live latents and labeled count.
- **List:** every latent with its label (or its top token), firing frequency and the output tokens it promotes.
  Search matches labels, top tokens, promoted tokens or an id. Sort by labeled first, frequency or max
  activation, and filter by category. `j` / `k` step through the list.
- **Feature tab:** label, explanation, stats, direct output effect, the top held-out activations with the token
  highlighted by strength (W = web, C = chat), and the closest decoder directions in the other SAEs. Click one to
  follow a feature as it splits across sizes.
- **Map tab:** the UMAP of decoder directions, colored by category or by firing frequency. The search and filters
  dim everything else. Hover for the label, click to open, scroll to zoom, drag to pan, double-click to reset.
- **Notes tab:** the SAE's `notes.md`.

`extract.py` writes to `runs/lfm_sae/extract/<timestamp>/`, a symlink to `/mnt/ai/lfm_sae/extract/<timestamp>/`
(`--data-root`):

- `config.json`: config, hook names, git commit, GPU name (`device_name`), dataset revisions.
- `shards/NNNNN.safetensors`: per hook point `<site>.<layer>` bf16 `[n, 2048]`, and row-aligned `token_ids` int32,
  `doc_ids` int64, `positions` int32 and `sources` uint8 (0 = web, 1 = chat). A document's rows rebuild its token
  context (prepend BOS), which is what top-activating-example views need. Shards hold at least `--shard-tokens`
  (262144) rows and end on chunk boundaries, so no document is split across shards. Within a shard, rows are grouped
  by document, not shuffled.
- `stats.safetensors`: per hook point, `<hook>.mean` `[2048]` and `<hook>.sq_norm_mean`, plus the running sums.
- `state.json`: documents consumed and tokens/shards written. `--resume` continues from it and deletes any shard
  written after it.
- `metrics.csv`: per shard tokens, docs, seconds, seconds blocked on the data worker (`data_wait_s`) and on the
  shard writer (`write_wait_s`), tokens/s. Shards are saved by a background thread with one shard of queue, so a
  large `write_wait_s` means the disk is the limit.
- `results.json`: totals, per-hook RMS/max norm, bytes, peak GPU memory.

Read shards with `common.iter_shards(run, ["resid_post.12", "token_ids"])` and stats with `common.load_stats`.

## Dev runs

Pass `train_sae.py` `--dev` to try a change without logging anything: the usual outputs go to a temporary directory that is deleted
when the script exits, so the run never appears under `runs/`.

## Training diagnostics in metrics.csv

Besides the columns above, every row has gradient and GPU columns (blank where a value was not measured):

- `grad_norm`, `grad_norm_max`: mean and largest global gradient norm (before clipping) since the previous row;
  `clip_frac`: fraction of steps clipped, when the script clips.
- `weight_norm`, `update_ratio` and `grad_cos` are not logged here: they need copies of the gradients and weights,
  which is gigabytes for the larger dictionaries.
- `gpu_util` (mean %), `gpu_mem_gb` (peak device memory), `gpu_alloc_gb` (peak memory torch allocated),
  `gpu_reserved_gb`, `gpu_power_w` (mean), `gpu_temp_c` (peak), sampled in the background while training.
