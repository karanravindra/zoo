# GPT

A Llama-style language model (pre-norm RMSNorm, rotary embeddings, SwiGLU, bias-free linears, tied embeddings),
pretrained on a token corpus. It starts on TinyStories (about 5.9M parameters with the defaults) and is built to scale to
FineWeb-Edu: the corpus is stored as token shards, and SFT, DPO and GRPO will be further stages of this project, each
starting from the previous stage's checkpoint.

| | |
| --- | --- |
| W&B project | `gpt`, group `pretrain` (later `sft`, `dpo`, `grpo`) |
| Data | `<DATA_ROOT>/gpt/<dataset>/`: `tokenizer.json` and `train/`, `validation/` token shards |
| Local run files | `<RUNS_ROOT>/gpt/<run id>/` |
| Extra dependencies | none |

```
uv run projects/gpt/cli.py prepare                       # download, train the tokenizer, write token shards
uv run projects/gpt/cli.py pretrain                      # pretrain; --max-steps, --dim, --n-layers, --block-size, ...
uv run projects/gpt/cli.py sample <checkpoint> --prompt "Once upon a time"
uv run projects/gpt/cli.py fit-lr | fit-scaling | check-anneal | fit-shape | fit-optim   # analyse the scaling sweeps (see "Scaling laws" below)
```

`cli.py <command> --help` lists a command's flags. Defaults are `--compile` and `--precision bf16-mixed`;
`--no-compile` and `--precision 32-true` for a quick CPU check, `--wandb offline` for no network, `--dev` for a
throwaway run, `--ema 0` for no weight average. The defaults are a constant learning rate (after 200 warmup steps) and EMA 0.999.
The average warms up (its window is about a ninth of the steps so far until it reaches `1 / (1 - ema)` steps), so a short run is not
dominated by the initial weights.

| Command | What it does |
| --- | --- |
| `prepare` | Streams a dataset into temporary text shards, trains a byte-level BPE with gigatoken, encodes every shard and writes the token shards. `--dataset tinystories` (default) or `fineweb-edu` (the 10BT sample, 10,000 documents held out for validation, 32,768 tokens); `--max-docs` caps the training documents. |
| `pretrain` | Trains on the corpus. Takes the vocabulary and end-of-text id from it. Checkpoints the best by `val/loss`. `--aspect A` sizes the model by `--n-layers` alone (width = A x layers), `--max-flops C` trains to a compute budget (steps = C / (FLOPs per token x `--global-batch`), at most one pass over the corpus), and `--lr-width-exp p` scales the peak LR by `(256 / width) ** p`, so one learning-rate rule covers a ladder of sizes. |
| `sample` | Continues a prompt from a checkpoint (the EMA weights when the run had `--ema`). |
| `fit-lr` | Scaling study, step 1: from the finished runs of a learning-rate sweep, the best peak LR at each width and one rule `LR*(width)`, with the flags to use it. |
| `fit-scaling` | Scaling study, step 2: on a ladder sweep, seed noise, IsoFLOP profiles, the law `L(N, D) = E + A N^-alpha + B D^-beta` with bootstrap error bars and the compute-optimal split, and held-out checks (`--lr rule`: every size at the rule's LR instead of its best). |
| `check-anneal` | Scaling study, step 3: constant LR with EMA against a run that anneals its LR to zero over the same steps. |
| `fit-shape` | Scaling study, shape: at equal compute, the loss against depth x width (a fitted surface, the best aspect ratio at each size), MLP size, head size and context length, with seed noise. |
| `fit-optim` | Scaling study, optimizer settings: at equal compute, weight decay, AdamW beta2, warmup, tokens per step (with its learning rate), gradient clipping and a document mask of one size against the shape grid's run of that size. |

Logged metrics: `train/loss`, `train/tokens_seen` and `train/flops` (cumulative over all devices; `flops` counts
matmul FLOPs at the full context, see `Config.train_flops_per_token`), `val/loss`, `val/bpb` (bits per UTF-8 byte,
comparable across tokenizers), `val/tokens_seen` and `val/flops` (the totals again, logged with the validation metrics so they can be the x-axis of a `val/` plot), and a text sample at every validation (`samples`), plus the shared `lr-`, `perf/` and
`stats/` groups. There is no test split in these corpora: validation picks the checkpoint and is the final score. The validation set is
`--val-tokens` tokens (2M) of evenly spaced windows, the same in every run whatever the micro-batch, so runs are comparable. The
run's config in W&B also has `n_params`, `n_params_body` (outside the tied embedding), `flops_per_token` and `peak_lr`.

Files:
- `cli.py`: the one entry point; each command is a short function.
- `config.py`: `Config` (the model, optimization and evaluation fields on top of `zoo.training.RunConfig`), `PrepareConfig`, `SampleConfig`, and the analysis commands' `LRFitConfig`, `ScalingConfig`, `ShapeConfig`, `OptimConfig`, `AnnealConfig`.
- `scaling.py`: the analysis behind `fit-lr`, `fit-scaling`, `check-anneal`, `fit-shape` and `fit-optim` (numpy and matplotlib only; reads the finished runs' `config.json` and `csv/metrics.csv`).
- `model.py`: the network and its KV-cache `generate`, shared by every stage.
- `pretrain.py`: `LitGPT`, the pretraining `LightningModule`.
- `data.py`: the corpus location, special tokens and `GPTData` (shuffled windows over the shards).
- `prepare.py`: `SOURCES` (a new dataset is one more entry) and the sharded tokenize-and-write pipeline.
- `sweeps/scaling_*.yaml` and `scaling_*.sh`: the W&B sweeps of the scaling study below, and the one-factor cells a grid would cross (the commands to start them are in each file).

Notes:
- **Shards.** Text goes to temporary shards of `--shard-mb` MB and tokens to `--shard-tokens` tokens per `.npy`
  shard, so RAM stays bounded whatever the corpus size. `zoo.data.text.Shards` reads them as one array.
- **Tokenizer.** gigatoken trains the BPE and encodes (in Rust, in parallel). Ids 256, 257, 258 are `<|endoftext|>`, `<|im_start|>`
  and `<|im_end|>`; the chat markers are unused in pretraining and reserved so SFT can use them without a new vocabulary.
- **Loss.** Apple's cut cross-entropy (`--loss auto`: CUDA with bf16) computes the loss from hidden states and the tied
  embedding, so the logits are never materialized. Elsewhere `auto` falls back to its `torch.compile` version.
- **Attention.** Plain causal attention is `F.scaled_dot_product_attention`, which uses FlashAttention on CUDA in bf16.
  `--doc-mask` keeps each token inside its own document with FlexAttention.
- **Batches.** A step is `--global-batch` tokens (16,384); `--batch-size` is the per-device micro-batch in sequences and gradient accumulation makes up the rest (it must divide evenly). `--micro-tokens N`
  sets `--batch-size` from a token count instead (`N // block_size`, at most the step), so a sweep over `--block-size` or `--global-batch` keeps one micro-batch size in tokens and, with `N` the step, no accumulation. CUDA graphs
  cannot accumulate gradients (the next replay overwrites the previous micro-batch's gradient tensors), so a run that accumulates compiles without them (`compile_(..., cudagraphs=False)`, automatic here): the same
  numbers, a little slower. Windows are consecutive, shuffled each epoch; a run is usually under one epoch, so `--max-steps` ends it (`--epochs` is -1).
- **Multi-GPU.** `--devices N` trains with DDP. It is wired up but not yet tried.

## Scaling laws

A study of how the validation loss depends on model size `N` and training tokens `D`, first on TinyStories (cheap, and the place
to debug the method), then on FineWeb-Edu (where the law is meant to live: TinyStories is one pass over 486M tokens, so its IsoFLOP
profiles are data-limited and the parametric fit is the main tool there).

The method rests on a constant LR and an EMA whose window is a share of the steps so far (see above): the first k steps of a run do
not depend on how long it was planned to be, so every validation row (`val/loss` against `val/flops`) is the result of a shorter run
too, and one run per size gives a whole loss-versus-compute curve. The sweeps are therefore over model size, not size times budget.
Sizes come from one knob, `--n-layers` with `--aspect 64` (width = 64 x layers, head dim 64), and every ladder run goes to
`--max-flops 1e17` or one epoch, whichever is first. All the sweeps use `--ema 0.99999` (it only caps the window), `--val-every 500`
(200 in the shape sweeps) and `--sample-tokens 0`.

| Step | Sweep (`sweeps/`) | Group | Analysis |
| --- | --- | --- | --- |
| 1. Learning rate | `scaling_lr.yaml`: 3, 6 and 8 layers x 5 LRs, 5000 steps | `tinystories-lr` | `cli.py fit-lr`: the best LR per width, and the rule `LR*(width) = LR0 (width / 256) ^ -p` to pass to later sweeps as `--lr LR0 --lr-width-exp p` |
| 2. Ladder | `scaling_ladder.yaml`: 2 to 12 layers (0.9M to 88M parameters) at the rule's LR; `scaling_bracket.yaml`: the LR x0.7 and x1.4 for 2 and 4 layers, and `scaling_bracket_ext.sh` (a script: a W&B grid would cross the sizes with every LR): x0.5 for 2 and 4 layers and x0.7 for 3, since the best was at the bracket's lower edge, and `scaling_bracket_mid.sh`: 6 layers at x0.5 and x0.7 for 10,000 steps and at x0.5 for the full run; `scaling_seeds.yaml`: seeds 1 and 2 of 3 and 4 layers | `tinystories-ladder` | `cli.py fit-scaling`: seed noise, IsoFLOP profiles, `L(N, D) = E + A N^-alpha + B D^-beta` with bootstrap error bars, the compute-optimal split, and held-out checks (the two largest sizes; the rows beyond 2e8 tokens) |
| 3. Anneal | `scaling_anneal.yaml`: 3, 6 and 8 layers x 2500, 10000 and 29500 steps, cosine to zero, no EMA; `scaling_anneal_mid.yaml`: 3 and 6 layers x 5000 and 20000 steps | `tinystories-anneal` | `cli.py check-anneal`: the final loss against the ladder run's loss at the same step |
| Shape | `scaling_shape.yaml`: 2 to 12 layers x width 128 to 384, each to 3e15 FLOPs; `scaling_shape_seeds.yaml`: two more seeds of two cells; `scaling_ff.yaml`: MLP size; `scaling_heads.yaml`: head size; `scaling_block.yaml`: context length; `scaling_ff_aspect.yaml`: MLP size at other aspect ratios; `scaling_shape_lr.yaml` (group `tinystories-shape-lr`): 2, 4, 8 and 12 layers at width 192 x two LRs around the rule's | `tinystories-shape` | `cli.py fit-shape`: all of them at equal compute (1e15, 2e15, 2.5e15 FLOPs, read off each run's validation curve), and the LR by depth: whether the width-only LR rule is fair to every depth of the grid |
| Optimizer | `scaling_wd.yaml`, `scaling_beta2.yaml`, `scaling_warmup.yaml`: weight decay 0 and 0.3, AdamW beta2 0.9 and 0.99, warmup 50 and 1000 steps; `scaling_batch.yaml`: 8192 and 32768 tokens per step x three LRs, and `scaling_batch_edge.sh`: three more LRs beyond its bracket's ends; `scaling_beta2_ext.sh`: beta2 0.99 with two more seeds and beta2 0.999; `scaling_wd_ext.sh`: weight decay 0 with two more seeds; `scaling_misc.sh` (a script, not a sweep: a W&B grid would cross the settings): gradient clipping off and 0.3, a document mask; all on the 4-layer size (4.3M parameters), to 3e15 FLOPs | `tinystories-optim` | `cli.py fit-optim`: each against the shape grid's run of that size at equal compute, the best LR at each batch size |
| Shape, larger | `scaling_shape_large.yaml`: 3, 6, 12 layers x width 256 to 512, each to 8e15 FLOPs | `tinystories-shape-large` | `cli.py fit-shape --group tinystories-shape-large --flops 4e15 6e15 7.5e15` |

Each sweep is created once (`uv run wandb sweep --project gpt projects/gpt/sweeps/<file>`) and run with `uv run wandb agent
<entity>/gpt/<sweep id>`; the commands are also in each file. The analysis commands read the finished runs of a group from
`<RUNS_ROOT>/gpt/<run id>/` (no network), print the fits and write their figures and `results.json` to
`<RUNS_ROOT>/gpt/scaling/<group>/`. Choices worth knowing:
- `N` is every parameter with the tied embedding counted once (`--params total`); `--params body` leaves the embedding out, and the
  fit prints both. A ladder run is its size's best learning rate (the rule's, or a bracket's) with seed 0; `--lr rule` takes only the runs at the
  rule's LR (`--rule-lr`, the ladder's base LR), so every size has the same recipe (2 to 4 layers have brackets, the larger sizes do not).
- `fit-scaling` uses validation rows from `--min-tokens` (1e8) on: earlier ones are the warmup and the early transient, where the loss
  is not yet a power law, and including them moves the exponents. It refits at several cuts (`--cuts`) and prints the exponents, the
  compute-optimal size and the loss at 1e17 FLOPs for each, so the dependence is on the page (the cut moves `beta` a lot; the
  compute-optimal size and the loss it predicts much less). `beta` is also the exponent the data pin down least: the printout
  refits with it fixed to show how the compute-optimal split moves.
- The shape sweeps compare at equal compute, not equal parameters: a wider network sees fewer tokens, and that is the trade that matters.
  `fit-shape` reads each run's validation curve at 1e15, 2e15 and 2.5e15 FLOPs (every cell of the grid gets to 2.5e15) and leaves a run out
  of a budget it reaches with fewer than `--min-tokens` (1e7, 3x the warmup) tokens; 3e7 would drop every cell above about 12M parameters,
  the ones that show where the loss turns up for a model that is too big for the budget. The transient costs a big cell a little, so
  rerun with `--min-tokens 3e7` to see what the cut moves (the best aspect ratio at a given size moves little; the size axis more).
  The MLP variants also print a size-corrected gap (the raw gap minus what the fitted depth x width surface gives for moving to the
  variant's size); it extrapolates the surface at the biggest size, where it is not to be trusted. The better estimate is the joint fit
  that follows: one surface for every cell and seed plus one offset per MLP / head variant (with standard errors and the residuals by
  cell), so the variants help pin the surface down. Its offsets are at equal parameters, aspect ratio and compute: a bigger MLP also
  sees a few percent more tokens (less of its FLOPs go to attention and the output layer), which is part of what it measures. With
  `scaling_ff_aspect.yaml` it also prints how an MLP offset changes with the aspect ratio.
- The mean validation loss over a window is not comparable across context lengths (a longer window has more late, easy positions), so
  `--val-by-pos` also logs it by position range (`val/loss_pos_<from>_<to>`) and `fit-shape` compares the ranges every length has.
- `fit-optim` compares one size (the 4-layer, 4.3M parameters) at a few optimizer settings with the shape grid's run of the same size,
  at the same budgets. The base cell's seed noise (printed) is the scale for comparing two different runs; a setting is compared with the base cell
  of the same seed (same initialization and data order), and a replicate (the same setting with another seed) with the base cell of its own seed,
  printed as `(seed n)`. Such paired differences repeat much more tightly than the seed noise (weight decay 0 gave -0.0099, -0.0102 and -0.0101 at 2.5e15),
  so a replicated gap of a few thousandths is real, but it is for this one size and length.
- A run of an interrupted sweep that did not finish is skipped (and listed), so a lost grid cell is visible as a missing size; rerun it
  by hand with the same flags.
- A sweep with a boolean (`--val-by-pos`) needs `${args_no_boolean_flags}` in its `command`: tyro takes a bare `--flag`, not `--flag=True`.

### Every variable of a run

| | Variable (flag) | Ladder / default | In the study |
| --- | --- | --- | --- |
| Axes | model size `N` (`--n-layers` with `--aspect`) | 2 to 12 layers, width 64 x layers, 0.9M to 88M parameters | swept |
| | training tokens `D` (`--max-flops`, or `--max-steps` x `--global-batch`) | one run's validation curve covers every budget up to its own | swept |
| Tuned | peak LR (`--lr` at width 256, `--lr-width-exp`) | 2e-3, falling as 1 / width | calibrated at 5000 steps (`fit-lr`); bracketed at full length for 2 to 4 layers (the rule is 0.02 too high there, the best x0.5 or lower) and for 6 layers (x0.5 is 0.015 better than the rule at 29,500 steps, equal at 10,000); depth at fixed width: `scaling_shape_lr.yaml` (no effect at width 192 up to 2.5e15 FLOPs) |
| Shape | depth against width at a given `N` (aspect = width / layers) | 64 | `scaling_shape*.yaml`: the best aspect is about 26 (N / 1M)^0.30; aspect 64 costs +0.017 at 1M, +0.005 at 4M and under 0.002 from 8M |
| | MLP size (`--ff-mult`: hidden / width; 0 = Llama's 8/3, rounded up to 32) | 8/3 | `scaling_ff.yaml`: 1, 2, 4, 6 at aspect 64; `scaling_ff_aspect.yaml`: 1 and 4 at aspect 24 to 128: x1 is +0.015 worse, the others are within 0.006, and there is no interaction with the aspect ratio |
| | head size (`--head-dim`; heads = width / head size) | 64 | `scaling_heads.yaml`: 32, 128: within 0.006 |
| | context length (`--block-size`; `--micro-tokens` keeps the micro-batch size in tokens) | 256 | `scaling_block.yaml`: 128 to 1024: a longer window is no better here (TinyStories documents are short: median 194 tokens, 11% of tokens sit beyond position 256 of their document) |
| | vocabulary (`prepare --vocab-size`; the embedding is tied) | 4096 | not tested (it changes the corpus, so the tokens) |
| Optimizer | tokens per step (`--global-batch`) | 16,384 | one size (4-layer): `scaling_batch.yaml` and `scaling_batch_edge.sh`, 8192 and 32768 with four or five LRs each: the best LR goes as batch^0.7 (1.2e-3, 2e-3, 3.2e-3), 16,384 is fine; the LR rule is for 16,384, and the critical batch size grows with the model |
| | weight decay, AdamW beta2, warmup | 0.1, 0.95, 200 steps | one size (4-layer): `scaling_wd.yaml` (0: -0.010 over three seeds, 0.3: +0.031 at 2.5e15 FLOPs), `scaling_beta2.yaml` (0.9: +0.008, 0.99: -0.005), `scaling_warmup.yaml` (50: 0, 1000: +0.002, early +0.023), replicates in `scaling_beta2_ext.sh` and `scaling_wd_ext.sh` |
| | AdamW beta1 | 0.9 | not tested (not a flag) |
| | gradient clipping (`--grad-clip`) | 1.0 | one size (4-layer): `scaling_misc.sh`, off and 0.3: no difference: -0.0012 / -0.0016 at 2.5e15 for off / 0.3 |
| | schedule and weight average (`--lr-schedule`, `--ema`) | constant, EMA with a window of a ninth of the steps | against cosine to zero in `scaling_anneal*.yaml`: annealing is worse below about 5,000 steps and better beyond (-0.05 at 29,500 steps for a small model) |
| | initialization, RoPE theta (`--rope-theta`), norm epsilon, precision | std 0.02 (0.02 / sqrt(2 x layers) on the residual outputs), 1e4, 1e-5, bf16 | not tested |
| Data | corpus, one pass at most | TinyStories, 486M training tokens (a full ladder run, 29,500 steps, uses 483M) | FineWeb-Edu next |
| | document mask (`--doc-mask`: attention stays inside a document; off = documents concatenated) | off | one size (4-layer): `scaling_misc.sh`: no difference: +0.003 at 2.5e15 |
| Evaluation | validation tokens (`--val-tokens`), validation interval (`--val-every`) | 2M, 500 steps (524,288 and 200 in the shape sweeps) | fixed windows, the same in every run |
| | seed (`--seed`: initialization and data order) | 0 | noise from extra seeds in `scaling_seeds.yaml` and `scaling_shape_seeds.yaml`: std about 0.005 (0.0065 for the 3-layer ladder size at 29,500 steps) |

### Stage 1 results (TinyStories)

Everything below is TinyStories (4096-token BPE vocabulary, 486M training tokens of which a full run's 29,500 steps use 483M, context 256), one seed unless a count is given, read off
validation curves at equal compute. Single-run noise is a standard deviation of about 0.005 in validation loss (three seeds of the 4-layer and
6-layer cells of the shape grid; 0.0065 at the end of a full ladder run and 0.008 to 0.010 earlier for three seeds of the 3-layer size), so a difference under about 0.01 is not a finding
unless the two runs share a seed. For scale: a model twice as big or small as the compute-optimal one costs
0.02 to 0.05, a learning rate 2x above the best costs 0.01 to 0.02, the best depth/width split is worth 0.003 to 0.006 at the compute-optimal size.

#### Learning rate

- The best peak LR falls as 1 / width: `LR*(width) = 2.06e-3 (width / 256) ^ -0.99` (widths 192, 384, 512 at 5000 steps: 2.5e-3, 1.3e-3, 1.0e-3;
  jackknife on the exponent +-0.07; the loss is quadratic in log LR, 0.014 x (octaves from the best) squared: an LR 2x off the best costs 0.014). The sweeps use `--lr 2e-3 --lr-width-exp 1.0`.
  The wide models are the fragile ones: at 5000 steps the loss blows up at 6.5e-3 for width 384 and at 3.6e-3 for width 512, 4.7x and 3.5x the rule's LR (2.6x and 1.9x still trained).
- Depth barely moves it; run length does (2 to 6 layers). At width 192 the rule's LR is within 0.003 of the best of three LRs at 4, 8 and 12
  layers at budgets up to 2.5e15 FLOPs (6,000 steps or fewer); 2 layers prefer a lower LR (by up to 0.008, at the edge of the three). In the ladder's full
  runs (29,500 steps, 483M tokens) the best constant LR is lower: four LRs (x0.5, x0.7, x1, x1.4 of the rule) at 2 and 4 layers put the best at x0.5, the
  lowest, in both (2.0e-3 and 1.0e-3 at their widths; a parabola through the four points puts the minimum at about x0.4, only 0.001 lower), and the rule's own LR
  is 0.021 and 0.020 worse than that cell (2.0494 against 2.0283 at 2 layers, 1.6346 against 1.6148 at 4); at 3 layers x0.7 is 0.020 better than the rule (1.7776
  against 1.7977; no lower cell was run). At 10,000 steps the rule costs 0.010, 0.013 and 0.007 at 2, 3 and 4 layers, but nothing at 6 layers (12M parameters:
  x0.5, x0.7 and x1 give 1.5540, 1.5549 and 1.5552). At 29,500 steps 6 layers also prefer less: x0.5 gives 1.4546 against the rule's 1.4692, so the rule costs
  0.0146 there (two LRs were run at that length). A constant LR with an EMA wants a lower peak the longer it runs, so the rule, calibrated at
  5,000 steps, is about right for runs of that length and costs 0.015 to 0.021 at 29,500 steps for 2 to 6 layers; x0.5 was the lowest LR tried at every
  full-length bracket, so the best may be lower still (`bracket.png`). Why is not established: tokens per parameter does not explain it (2, 3 and 4 layers cost the same
  0.020 at 29,500 steps though their tokens per parameter differ by 2x, and 6 layers at 10,000 steps cost nothing). The ladder's other sizes (5, 8, 10 and 12 layers)
  were run only at the rule, so the law below is shown both with every size at the rule and with 2 to 4 and 6 layers at their best LR. Whether an annealed
  run wants the same LR was not measured (the annealing runs use the rule's).

#### Depth against width (`scaling_shape*.yaml`)

- At a given parameter count the best aspect ratio (width / layers) grows with size, about `aspect* = 26 (N / 1M) ^ 0.30`:
  26 at 1M parameters, 32 at 2M, 39 at 4M, 44 at 6M, 48 at 8M, 54 at 12M, 59 at 16M (68% intervals within +-3 to 6 at 2.5e15 FLOPs; nearly the same
  at 1e15 and 2e15, so it hardly depends on tokens per parameter). The loss rises about 0.010 per doubling of the aspect ratio away from it.
- The ladder's aspect 64 therefore costs +0.017 at 1M, +0.010 at 2M, +0.005 at 4M, +0.003 at 6M, +0.002 at 8M and nothing from 12M. At the
  compute-optimal size (3.2M at 1e15, 4.7M at 2e15, 5.3M at 2.5e15 FLOPs) the best aspect is 39 to 43 and the gain over aspect 64 is 0.003 to
  0.006: below the noise of one run. Refitting the ladder with that cost removed leaves the law where it was (alpha 0.443 to 0.431, the
  compute-optimal exponent unchanged), so the ladder's fixed aspect does not bias the compute-optimal split.
- The cells too big for the budget show where the loss turns up: at 2.5e15 FLOPs a 12-layer, 384-wide model is 0.18 worse than the best cell.
- Nine larger cells (3, 6 and 12 layers x width 256, 384 and 512, 3.5M to 40M parameters, at 4e15, 6e15 and 7.5e15 FLOPs; `scaling_shape_large.yaml`)
  agree. At 7.5e15 three cells of nearly one size, 11 to 12M parameters, put the best aspect near 55: 21 gives 1.638, 64 gives 1.628, 171 gives
  1.646. The nine-cell surface (6 parameters, so 3 degrees of freedom: a summary, not a test) has the best aspect at 50 to 57 for 8M to 12M parameters
  (the grid's formula says 48 to 55) and aspect 64 costing under 0.002 everywhere. Its minimum over size and aspect is at 7.3M, 9.0M and 10.1M
  parameters (aspect 52 to 53) for the three budgets and only 0.0006 below the minimum along aspect 64. With the grid's 3.2M, 4.7M and 5.3M at
  1e15, 2e15 and 2.5e15, the size that minimizes the loss grows as C ^ 0.58 (6M at 3e15, 12M at 1e16), the same as the ladder's direct IsoFLOP
  minima (6.0M at 2.9e15, 12.4M at 1.2e16).

#### MLP size, head size, context length

- Joint fit of the grid's surface and one offset per variant (equal parameters, aspect ratio and compute; positive = worse than the 8/3 MLP and
  head size 64), at 2.5e15 FLOPs: MLP x1 +0.015 +-0.002, x2 +0.002 +-0.003, x4 +0.001 +-0.002, x6 +0.006 +-0.003; head size 32 -0.005 +-0.004,
  128 +0.006 +-0.004. Llama's 8/3 sits on a flat optimum (2 to 4 are the same, 6 is a little worse, 1 is clearly worse); head size makes no
  difference beyond the noise. The MLP offset does not change with the aspect ratio (24 to 128: slopes +0.0015 +-0.003 for x1 and +0.0002 +-0.003 for x4 per doubling), so there is no
  depth x width x MLP interaction to speak of.
- Context length at equal compute (128, 256, 512, 1024 on 4 and 6 layers): a longer window is no better at any range of positions that all the lengths have.
  At 2.5e15 FLOPs, loss minus the 256 window's over the positions 0 to 32 / 32 to 64 / 64 to 128 / 128 to 256: 4 layers, 512: -0.003 / +0.005 / +0.004 /
  +0.014 and 1024: +0.129 / +0.060 / +0.030 / +0.046; 6 layers, 512: +0.019 / +0.029 / +0.033 / +0.043 and 1024: +0.178 / +0.100 / +0.068 / +0.099; the
  128 window is better at the positions it has (-0.07 to -0.01). Two things are mixed in. Attention FLOPs grow with the window, so at equal FLOPs the 1024 window sees 66M tokens
  against 87M for 256 and 92M for 128 (4 layers), which the ladder law prices at about +0.04. And TinyStories documents are short (median 194 tokens, 11% of
  tokens past position 256 of their document, 0% past 1024), so the extra context has nothing to say, and the windows' cold start (positions 0 to 32) comes up
  4x less often per token. This is a property of the corpus; do not carry it to FineWeb-Edu, where the context should be at least 1024 and the comparison redone.

#### The ladder law (constant LR + EMA, 8 sizes of aspect 64, 0.9M to 88M parameters)

Eight sizes (2 to 12 layers at width 64 x layers; parameters counted with the tied 4096-token embedding), each trained once for 29,500 steps
(483M tokens, one pass over the corpus; the larger ones stop at 1e17 FLOPs) at a constant LR with an EMA of 0.99999, so a run's validation curve is
also its loss at every smaller budget. The fit uses the rows from 1e8 tokens on (`cli.py fit-scaling`; `--lr rule` takes only the runs at the rule's LR).

- Law, all eight sizes at the rule's LR: `L = 1.164 + 0.789 N^-0.443 + 0.031 D^-0.780` (N in millions of parameters, D in billions of tokens, FLOPs =
  6.61 N D; 315 rows, rms residual 0.0089; bootstrap over sizes: alpha 0.43 to 0.47, beta 0.65 to 0.87). With 2 to 4 and 6 layers at their best LR from the
  bracket (the rest stay at the rule): E 1.148, A 0.771, alpha 0.443, B 0.040, beta 0.720 (0.63 to 0.80), rms 0.0079. The prediction at 1e17 FLOPs is the same to
  0.004 (1.382 and 1.378): the loss is robust, the split is not (beta is pinned only to about 0.5 to 1.0: with it fixed anywhere in that range the rms residual is
  0.0089 to 0.0091, while the compute-optimal exponent goes from 0.53 to 0.69 and N* at 1e17 FLOPs from 46M to 54M).
- Compute-optimal size from the fit: N* ~ C^0.64 (68% interval 0.59 to 0.66; 0.62, interval 0.58 to 0.64, with the best-LR runs): 5.4M parameters at 3e15 FLOPs (16 tokens per
  parameter), 11.6M at 1e16 (11), 23M at 3e16 (8), 50M at 1e17 (6) (with the best-LR runs: 5.1M, 10.8M, 21M and 45M, 17 to 7 tokens per parameter). From the IsoFLOP minima instead (five sizes per budget, a quadratic in ln N): 6.0M at
  2.9e15, 8.6M at 5.8e15, 12.4M at 1.2e16, 16.1M at 2.0e16, 17.8M at 2.4e16, that is C^0.51 and 11 to 12 tokens per parameter at every budget (with the best-LR runs: 5.4M, 11.4M and 17.7M at 2.9e15, 1.2e16 and 2.4e16, C^0.55, 12 to 15 tokens per parameter). The two
  disagree because the fit's beta is high (0.78); the schedule section below says why the IsoFLOP reading is the better one.
- A wrong size at equal compute, from the fit: twice or half the compute-optimal size costs 0.054 / 0.038 / 0.028 / 0.020 (twice) and 0.046 / 0.033 /
  0.024 / 0.017 (half) at 3e15 / 1e16 / 3e16 / 1e17 FLOPs; 1.5x too big or 1.5x too small costs 0.018 / 0.012 / 0.009 / 0.007.
- The row cut matters (the fit on rows from 3e7, 6e7, 1e8 and 1.5e8 tokens: beta 0.60, 0.68, 0.78, 0.85, exponent 0.57, 0.61, 0.64, 0.66), and so does the
  parameter count: with body parameters (without the embedding, 0.5M of the smallest size's 0.9M) alpha is 0.27 instead of 0.44. Alpha is a property of
  "total parameters at this vocabulary"; with FineWeb-Edu's vocabulary, count both.
- The law is not exact, and its misfit has a pattern: the residuals (observed minus fitted, the rule-LR fit) go from -0.021 over the first third of the 2-layer
  run's rows to +0.012 over the last third, and from +0.010 to -0.014 for the 8-layer run (27M parameters). Small models flatten more late than the
  fit allows and large ones keep falling faster, that is beta depends on size, and part of it is the LR (the rule is too high at 29,500 steps, which the
  six smaller sizes run; the two largest stop at 1e17 FLOPs, 17,700 and 10,500 steps; the fit's rms falls from 0.0089 to 0.0079 once 2 to 4 and 6 layers use
  their best LR). Fits that hold data out miss by up to 0.03 to 0.035 in loss (without the two largest sizes: mean -0.009 / -0.024, worst 0.031 / 0.035;
  the first 2e8 tokens predicting the rest: mean 0.000, worst 0.035), the size of the LR and schedule effects, so the extrapolation past 2.4e16 FLOPs (the
  largest budget with five sizes) is good to about 30% in N.
- The ladder's fixed aspect ratio does not bias this: removing the cost of aspect 64 (from the shape grid) leaves alpha at 0.431 (from 0.443) and the exponent
  unchanged.
- Seeds: the 3-layer size (2.1M parameters) was run with three seeds at the rule's LR. The standard deviation of its validation loss between runs is 0.0104 at 2,000
  steps, 0.0091 at 5,000, 0.0080 at 10,000 and 0.0065 at 29,500 (range 0.013 at the end). Replacing that size's curve by the seed mean leaves the law where it was (alpha 0.443,
  beta 0.780, loss at 1e17 1.3815 against 1.3816, rms 0.0086 against 0.0089); the seed-0 run is 0.007 above the mean over the last rows. Seeds of the 4-layer size were planned
  too, but the run was interrupted (seed 1 stopped at 19,500 of 29,500 steps and seed 2 did not run), so only the 3-layer size has a noise estimate from the ladder.

#### Learning-rate schedule

The ladder's runs are a constant LR with an EMA. Annealing (cosine to zero, no EMA, `scaling_anneal*.yaml`) over the same number of steps gives,
as annealed minus constant-LR + EMA at the end of the run (negative = annealing is better; 12 runs, 3 sizes):

| steps (tokens) | L3, 2.1M |  (x0.5 is 0.015 better than the rule at 29,500 steps, equal at 10,000), 12.2M | L8, 27.4M |
| --- | --- | --- | --- |
| 2,500 (41M) | +0.016 | +0.016 | +0.012 |
| 5,000 (82M) | -0.002 | +0.004 | |
| 10,000 (164M) | -0.022 | -0.010 | -0.003 |
| 20,000 (328M) | -0.042 | -0.028 | |
| 29,500 (483M) | -0.052 | -0.039 | |

The EMA wins on runs of a few thousand steps; annealing wins beyond about 5,000 and the gap keeps growing with the length, and it is larger for
the smaller model (gap = -0.003 - 0.021 ln(D / 1e8) + 0.003 ln(N / 1e7) + 0.005 ln(D / 1e8) ln(N / 1e7), rms 0.0024, where the interaction is 4
standard errors). So the rankings at equal compute among nearby sizes and shapes carry over (they differ by under 0.005 on the grid's runs), but the
law's level, `beta` and the compute-optimal split are those of constant-LR + EMA curves. Adding the fitted gap to every ladder row and refitting
(`beta` from 0.78 to 0.55-0.60, the compute-optimal exponent from 0.64 to 0.52-0.57, N*(1e17 FLOPs) from 50M to 33-37M, which is 11-14 tokens per
parameter instead of 6, and the loss at 1e17 from 1.382 to 1.343-1.358) says an annealed model wants about twice the tokens per parameter the
constant-LR fit gives. The direct
IsoFLOP minima (5.98M at 2.9e15 to 17.8M at 2.4e16 FLOPs, 11 to 12 tokens per parameter throughout, exponent 0.51) agree with the annealed
version. In the measured range (up to 2.4e16 FLOPs), use about 10 to 12 tokens per parameter; the extrapolation to 1e17 is good to about
30% in N whichever way it is fitted.

#### Optimizer pilot (4-layer size)

Weight decay, AdamW beta2, warmup, tokens per step with the LR, gradient clipping and a document mask, at one size (4 layers, width 256, 4.3M parameters,
runs of 3e15 FLOPs; `scaling_wd.yaml`, `scaling_beta2.yaml`, `scaling_warmup.yaml`, `scaling_batch.yaml`, `scaling_batch_edge.sh`, `scaling_beta2_ext.sh`,
`scaling_wd_ext.sh`, `scaling_misc.sh`; `cli.py fit-optim`). Each is read at 1e15, 2e15 and 2.5e15 FLOPs against the shape grid's run of the same size and seed. Loss minus the base
recipe (weight decay 0.1, beta2 0.95, warmup 200 steps, 16,384 tokens per step, LR 2e-3, clipping at 1.0, no document mask); positive = worse. The base cell's
seed noise (3 seeds) is 0.0057, 0.0050, 0.0047 at the three budgets: that is the scale for comparing two different runs. A setting compared with the base cell of its
own seed (same initialization and data order) repeats much more tightly: weight decay 0 gave -0.0099, -0.0102 and -0.0101 at 2.5e15 over three seeds, beta2 0.99 gave
-0.0071, -0.0038 and -0.0045. So here 0.003 is about what a single paired run can resolve, and the single-seed rows below (clipping, mask, warmup 50) say "no effect to about 0.003".

| setting | 1e15 | 2e15 | 2.5e15 | reading |
| --- | --- | --- | --- | --- |
| weight decay 0 (seed 0) | -0.0045 | -0.0082 | -0.0099 | |
| weight decay 0 (seed 1) | -0.0030 | -0.0081 | -0.0102 | each against the base cell of its own seed |
| weight decay 0 (seed 2) | -0.0054 | -0.0084 | -0.0101 | three seeds, the same 0.010 better at 2.5e15: with 6,000 steps there is nothing for it to regularize; retest at the real length |
| weight decay 0.3 | +0.0146 | +0.0267 | +0.0313 | worse, and more so the longer the run |
| AdamW beta2 0.9 | +0.0087 | +0.0087 | +0.0079 | worse |
| AdamW beta2 0.99 (seed 0) | -0.0050 | -0.0069 | -0.0071 | |
| AdamW beta2 0.99 (seed 1) | -0.0018 | -0.0022 | -0.0038 | each against the base cell of its own seed |
| AdamW beta2 0.99 (seed 2) | -0.0030 | -0.0040 | -0.0045 | three seeds, one sign: about 0.005 better (mean -0.005 at 2.5e15), each within the noise |
| AdamW beta2 0.999 | -0.0047 | -0.0045 | -0.0041 | one seed, the same as 0.99 |
| warmup 50 steps | -0.0030 | +0.0004 | +0.0009 | no difference |
| warmup 1,000 steps | +0.0228 | +0.0045 | +0.0024 | worse early, gone by 2.5e15 |
| gradient clipping off | -0.0002 | -0.0007 | -0.0012 | no difference |
| gradient clipping 0.3 | -0.0007 | -0.0011 | -0.0016 | no difference |
| document mask | -0.0007 | +0.0029 | +0.0031 | no difference (TinyStories documents are short: median 194 tokens) |

Tokens per step and the LR (8,192 and 32,768 tokens per step with five and four LRs each, against the base's 16,384 at its 2e-3): the best LR at 2.5e15 FLOPs is
1.2e-3 at 8,192 and 3.2e-3 at 32,768 tokens per step (the rule's 2.0e-3 at 16,384 is in between), that is proportional to batch^0.7 (between square-root and linear scaling). At equal compute the best-LR small batch is a little worse (+0.008 at 2.5e15, 1.5 sigma; at 1e15 it is
0.004 better), and 32,768 tokens at its best LR is 0.003 better than 16,384 (within the noise; its optimum is at the edge of the LRs tried at the shorter budgets), so 16,384 tokens per step is fine.

#### What to carry to FineWeb-Edu

- LR: `2e-3 * (256 / width)` at 16,384 tokens per step with 200 warmup steps and the EMA for the readout; scale it about as batch^0.7 if the batch changes. It is
  about right for runs of 5,000 steps; the longer a constant-LR run, the lower its best LR: at 10,000 steps x0.6 to 0.7 for 2 to 4 layers, at 29,500 steps x0.5 or less for 2 to 6 layers
  (the rule costs 0.015 to 0.021 there). Bracket the LR (x0.35, x0.5, x0.7, x1: x0.5 was the edge here) at the middle size of the new ladder, at the length the
  ladder will run, before running the ladder (here the bracket came after it, and the sizes from 5 layers up were never re-tuned).
- Shape: aspect ratio (width / layers) 40 to 64 for 4M to 90M parameters (the best at the compute-optimal size is 40 to 55 and 64 costs under 0.005 there), MLP 8/3
  (2 to 4 are the same), head size 64.
- Size against tokens: about 10 to 12 tokens per parameter (total, tied embedding included) from 1e15 to 2.4e16 FLOPs, the compute-optimal size growing about as
  C^0.5 to 0.6. A model 1.5x off the optimum costs 0.006 to 0.018, 2x off 0.02 to 0.05. With FineWeb-Edu's vocabulary count total and body parameters.
- Schedule: anneal the real run (cosine to zero beats a constant LR with EMA beyond about 5,000 steps, by 0.01 to 0.05 and more the longer the run), and fit the
  FineWeb-Edu ladder on annealed runs (decay branches off one constant-LR run per size) rather than constant LR + EMA: the constant-LR law has beta 0.78 and 6
  tokens per parameter at 1e17 FLOPs, the annealed-equivalent one beta 0.55 to 0.60 and 11 to 14.
- Context: at least 1024 on FineWeb-Edu (TinyStories positions beyond 256 do nothing); redo the context comparison there. Document mask: no difference at 256 tokens on TinyStories (+0.003, inside the noise); retest it on FineWeb-Edu, where a 1024-token window holds many documents of different lengths..
- Optimizer: weight decay 0.1 (0 was 0.010 better over three seeds at one 4M size over 6,000 steps and 0.3 was 0.031 worse: the effect scales with LR x decay x steps, so retest 0 to 0.1 at the real length), AdamW beta2 0.95 (0.99 to
  0.999 about 0.005 better here, 0.9 0.008 worse; 0.95 is the safe choice at a high LR), warmup 200 (50 the same, 1,000 worse early), gradient clipping 1.0
  (off and 0.3 are the same within 0.002: not a lever here; keep it for stability at a high LR).
- Noise: single-run noise is 0.005, so decide anything under 0.01 with two or more seeds.
