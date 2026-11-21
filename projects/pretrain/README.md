# Pretrain

Forked from [`projects/tinystories`](../tinystories/README.md). Runs go to `runs/pretrain/` and data to
`data/cache/pretrain/`, so the two projects don't share outputs. The data is a mix of story corpora (see `prepare.py`
below) instead of TinyStories alone.

A small Llama-style decoder (pre-norm RMSNorm, rotary embeddings, SwiGLU, bias-free linears, causal attention via
`F.scaled_dot_product_attention`, tied input/output embeddings) trained on a mix of story corpora: [`TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories),
[`SimpleStories`](https://huggingface.co/datasets/SimpleStories/SimpleStories),
[Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) `stories`,
[`WritingPrompts`](https://huggingface.co/datasets/euclaise/writingprompts) and
[Project Gutenberg](https://huggingface.co/datasets/common-pile/project_gutenberg). Defaults: dim 256, 6 layers,
8 heads, context 256, vocab 4096, about 5.9M parameters.

Run in this order, from anywhere:

```
uv run projects/pretrain/prepare.py --dry-run   # first: download, count stories and bytes per source, estimate tokens
uv run projects/pretrain/prepare.py             # once: train the tokenizer, write token streams
uv run projects/pretrain/train.py
uv run projects/pretrain/sample.py --prompt "Once upon a time"
uv run projects/pretrain/compare.py                    # all runs side by side -> runs/pretrain/comparison.png
uv run projects/pretrain/bench.py                      # zero-shot benchmarks of every run -> runs/pretrain/benchmarks.csv
uv run projects/pretrain/make_prompts.py               # once: shared story openings -> projects/pretrain/eval_prompts.jsonl
uv run projects/pretrain/sample.py <run>/best.pt --prompts projects/pretrain/eval_prompts.jsonl --max-tokens 512 --out <run>/eval_stories.jsonl   # per run, and the tinystories baseline with projects/tinystories/sample.py
uv run --group pretrain projects/pretrain/judge.py     # Claude grades every run's stories -> runs/pretrain/story_eval_<judge>.csv / .png (an OpenAI model: --provider openai --judge <model id>)
```

Files:
- `llama.py`: `Config` (model, optimization and runtime fields, all CLI flags), the model, checkpoint save/load.
- `prepare.py`: data prep for the mix. Each source in `SOURCES` is a Hub dataset plus a text column (`--sources` picks a
  subset, a new source is one more entry). Books are cut into ~8,000-character chunks of whole paragraphs (Gutenberg's
  hard line wraps are joined), WritingPrompts' Penn-Treebank spacing (`did n't`, `don ’ t`, a space before each full stop) is undone, and sources without a
  validation split hold out random rows (`val_docs`). It trains one 4096-token byte-level BPE with
  `gigatoken.train_bpe` on `--vocab-mb` of text from every source (equal bytes each), writes
  `data/cache/pretrain/tokenizer.json` (end-of-text token is id 256), then encodes each source with gigatoken's file
  path (`TextFileSource`, parallel in Rust) into `parts/<source>/<split>/`, and concatenates those into
  `train/tokens.npy` (uint16) and `validation/tokens.npy` through `zoo.data.text`, each with a `meta.json` (the train
  one has a per-source breakdown). `parts/` is what makes the run resumable and lets you add a source later without
  re-encoding the others. `validation/` is the held-out `--val-source` (TinyStories) only, so val bpb stays comparable
  with the TinyStories runs; the other sources' held-out stories are in `validation_<source>/`, which `train.py` scores
  at every eval. `--max-docs 2000` runs the whole pipeline on a few rows per source; `--vocab-size` must match `train.py`'s.
- `train.py`: step-based loop (not epochs): random windows from `zoo.data.text.TokenDataset`, AdamW with weight decay on
  2D parameters only, linear warmup then cosine LR, grad clipping, optional bf16 autocast on cuda, optional weight EMA (`--ema`, averaging window `--ema-frac` of the steps taken so far, default 0.1, uncapped so it is the same share of training at every scale; evals, samples
  and `best.pt`/`last.pt` then use the averaged weights, the optimizer keeps training the raw ones; the `train_*` and gradient columns of metrics.csv still come from the raw weights, so a train/val gap on an EMA run mixes two models), an eval every
  `--eval-every` steps over `--eval-batches` fixed val batches, and a text sample every `--sample-every` steps
  (independent of evals; 0 disables). Each eval also scores every source's held-out stories (`validation_<source>/`
  from `prepare.py`) with `--val-source-batches` batches each (0 skips them) and logs `val_<source>_loss` /
  `val_<source>_bpb`; the `validation/` source gets the same columns copied from `val_loss`. `best.pt` is still picked on
  `val_loss` alone. A run also writes `val_sources.png` and `final_val_<source>_*` into `results.json`.
- `sample.py`: generate from a checkpoint (default: `best.pt` of the latest run). `--num-samples N` generates a batch,
  `--out FILE` also writes the samples to a file, and `--prompt ""` starts whole new stories from the end-of-text
  token. Generation uses a KV cache: one new token per step against cached keys / values, falling back to a
  sliding window once the context reaches `block_size`.
- `compare.py`: compare runs (default: all under `runs/pretrain`, or pass run dirs). The top row is val loss
  against training time, training FLOPs and `--x step|tokens|elapsed_s|flops` (`tokens` compares runs with different
  batch sizes); the val panels clip the steep start of the curves. Below are train loss, tokens/s and a
  best-val-bpb bar chart, plus a table with parameter counts and the time and FLOPs each run took. Runs are ordered by
  FLOP budget, then by the config values that differ, and labelled in short form (`3e17 d640 L8 64k tok/step lr0.0008`;
  batch size is tokens per step, 1k = 1024; batch splits from OOM restarts and a FLOP budget's `max_steps` don't count as
  differences), in one legend below the panels; runs still training are included. `--log-x` for log x axes. The loss shown
  is `val_doc_loss` when every run has it, else `val_loss` (`--metric` picks one). When any run has per-source columns,
  `comparison_sources.png` (next to the first figure) and a table show each run's bits per byte on every source at its
  best eval row.
- `bench_memory.py`: speed and peak memory of the 1e18 / 3e18 run configs for each micro-batch size, with and without
  `--grad-checkpoint` (see "Batch size and GPU memory"). Writes `runs/pretrain/memory_bench.csv`.
- `make_prompts.py`: writes `eval_prompts.jsonl` (committed): the first sentence of 100 held-out TinyStories stories
  plus 10 from each other source's held-out split, so every model continues the same openings. Each row is
  `{id, source, prompt, reference}`; `reference` (the rest of the story) is only for a human spot check.
- `judge.py`: grades the `eval_stories.jsonl` files that `sample.py --prompts` wrote (the same flag exists in
  `projects/tinystories/sample.py`, so the top TinyStories run is the baseline) with Claude through the Batches API.
  `--mode abs` scores grammar, coherence, plot and creativity 1-10; `--mode pairwise` compares each run's story with the
  baseline's for the same prompt in both orders and reports a win rate. Both are reported separately for TinyStories
  openings (in-domain) and the other sources (out-of-domain), with the share of stories that ended within the token
  limit. `--judge` picks the grader (default `claude-haiku-5-5`), `--subset 0.2 --judge claude-sonnet-5-5
  --compare-judge claude-haiku-5-5` re-grades a stratified fifth with a stronger model and prints the agreement,
  `--dry-run` prints the request count and calls nothing. Grades are cached per run, so reruns only grade what is
  missing. Needs `ANTHROPIC_API_KEY` (or `ant auth login`). `--provider ollama --judge lfm3:latest` grades with a local Ollama model (schema-constrained JSON, temperature 0, nothing
  billed) and `--provider d3 --judge d3:9b` asks the d3 decision model for A / B / tie probabilities (comparisons only; 4096-token
  context). Each judge's report is `story_eval_<judge>.csv` / `.png`. `--provider openai --judge <model id>` grades with an
  OpenAI model through its Batch API instead (Chat Completions, strict JSON schema; `--reasoning-effort` is sent only
  when set) and needs an `OPENAI_API_KEY` with API billing, which a ChatGPT subscription does not include. Both
  packages are in the `pretrain` uv group.
- `bench.py`: zero-shot benchmarks (see "Benchmarks"), for every run or the run dirs passed. Writes `bench.json` into each
  run dir and `runs/pretrain/benchmarks.csv`, and prints a table with the chance level and a standard error.

## Dependencies

`gigatoken` (tokenizer training and encoding/decoding) and `cut-cross-entropy` (installed from the
`apple/ml-cross-entropy` git repo, since the PyPI release is older and carries no project links) are in the main
environment. `tokenizers` is not needed.

## Loss: hidden states and cut cross-entropy

The model's `forward(idx, return_hidden=True)` returns the final normed hidden states instead of logits. The training
and eval loss is `linear_cross_entropy(hidden, model.classifier_weight, y)` with the tied embedding as the classifier,
so the `(batch, block, vocab)` logits are never materialized. The batcher already yields shifted `(x, y)`, so CCE's
own `shift` option is not used. `generate` computes logits for the last position only.

`--loss` picks the implementation: `auto` (default) is the Triton `cce` kernel on cuda with bf16 autocast (Ampere or
newer) and `torch_compile` everywhere else (CPU, MPS, fp32); `torch` is plain `F.cross_entropy` on explicit logits, for
devices where CCE does not work. CCE casts to the autocast dtype itself. `--compile` compiles only the model; the loss
runs outside it, and every model output is consumed before the next call.

## Metrics

- `loss` is mean next-token cross-entropy in nats per token.
- `bpb` (bits per byte) = `loss / ln 2 * tokens / bytes`, so it is comparable across tokenizers. `tokens` and `bytes`
  are the split's totals recorded by `prepare.py`: tokens include the end-of-text tokens (the loss covers them too),
  bytes are the UTF-8 size of the story text only. `train_bpb` and `val_bpb` use their own split's ratio. A cache
  prepared before byte counts were recorded makes `train.py` stop and ask for `prepare.py --force`.
- `tokens_per_s` is training tokens (`batch_size * block_size * grad_accum` per step) over wall time since the
  previous `metrics.csv` row, with eval, sampling and checkpointing excluded and the device synced before reading the
  clock.
  With `--compile` the first window includes compilation.

## Run outputs

`<repo>/runs/pretrain/<name>/`, where the name is built from the model shape and the budget that ends the run:
`d256-l6-ctx512-20k` (steps), `d384-l6-ctx512-3e16` (FLOPs), `d256-l6-ctx512-600s` (seconds). `--run-name` overrides
it, and `-2`, `-3`, ... is appended rather than writing into an existing directory:

| File | Contents |
| --- | --- |
| `config.json` | full config plus git commit, device and the loss implementation used |
| `metrics.csv` | a row every `--log-every` steps (20) and at every eval; see "metrics.csv columns" below |
| `layers.csv` | every `--stats-every` steps (100): per layer group, weight norm, update ratio, gradient norm, gradient cosine |
| `best.pt` | checkpoint with the lowest validation loss |
| `last.pt` | checkpoint from the latest eval (the final step at the end) |
| `results.json` | best step with val loss and bpb, final val loss and bpb, final gradient norm, total steps and training FLOPs, mean tokens/s, peak GPU memory and mean GPU utilisation |
| `curves.png` | train and val loss against step |
| `grads.png` | gradient norm and update-to-weight ratio against step (log scale) |
| `bpb.png` | train and val bits per byte against step |
| `samples.txt` | one text sample every `--sample-every` steps and at the final step, from a fixed seed |
| `stories.txt` | not written by `train.py`: 50 whole new stories from `best.pt`, made with `sample.py --prompt "" --num-samples 50 --max-tokens 512 --out <run>/stories.txt` (temperature 0.8, top-k 40, seed 0) |
| `eval_stories.jsonl` | not written by `train.py`: one story per `eval_prompts.jsonl` prompt from `best.pt` (`{id, source, prompt, text, n_tokens, ended}`), made with `sample.py --prompts` (temperature 0.8, top-k 40, seed 0) |
| `judge_<judge>_abs.jsonl`, `judge_<judge>_pairwise.jsonl` | not written by `train.py`: cached `judge.py` grades per story (scores and a one-line comment) or per comparison (winner and reason) |
| `bench.json` | not written by `train.py`: the run's `bench.py` scores, per task `n`, `acc`, `acc_stderr` (and `acc_norm`, or lambada `ppl`), plus blimp per phenomenon and per subset |

Checkpoints are self-describing (weights plus the config that built them): `llama.load_checkpoint(path)` returns a
ready model.

## Benchmarks

`bench.py` scores checkpoints zero-shot on the usual small-model tasks, with the lm-evaluation-harness conventions but
no dependency on it: `blimp` (67 grammatical minimal-pair subsets, also averaged per phenomenon), `hellaswag`,
`lambada` (last-word greedy accuracy and its perplexity), `piqa`, `arc_easy`, `arc_challenge`, `winogrande`, `sciq`,
`openbookqa` and `copa`. The datasets are downloaded from the Hugging Face Hub on first use. A candidate text is scored by
the sum of its token log-probabilities after the context; `acc` takes the highest, `acc_norm` divides by the length in
characters (the table shows `acc_norm` for hellaswag, arc_challenge and openbookqa, `acc` for the rest).

Every sequence starts with the end-of-text token and runs in fp32 (`--bf16` for autocast); a context that does not
fit the block size is cut from the left. A windowed model (`--attn-window`) is run through the KV-cache priming pass,
which applies the window with plain SDPA, because FlexAttention without compile runs out of memory on long rows
(it matches the training forward to 1e-5). The scorer reproduces the training `val_doc_loss` to within 0.006 nats per token.

A model trained on TinyStories alone has seen almost none of these tasks' text: the 12-32M-parameter runs score a few
points above chance on hellaswag, piqa and arc_easy, around 12-14% on lambada, and at or below chance on the rest. The
gaps between runs are within a couple of standard errors (`acc_stderr` in `benchmarks.csv`) on everything except
`blimp`, whose 67k pairs separate the runs by about 3 points (stderr 0.2); the small tasks (copa has 100 examples) and
lambada's perplexity (heavy-tailed) are noise for comparing runs, and so is the `avg` column. `--limit N` is for a quick
check and writes nothing.

## Dev runs

Pass `train.py` `--dev` to try a change without logging anything: the usual outputs go to a temporary directory that is deleted
when the script exits, so the run never appears under `runs/`.

## metrics.csv columns

`metrics.csv` columns. Blank cells mean "not measured on this row"; the runs viewer and `compare.py` skip them.

| Columns | When | Meaning |
|---|---|---|
| `step`, `tokens`, `flops`, `elapsed_s`, `lr` | every row | cumulative training tokens, FLOPs and seconds |
| `train_loss`, `train_bpb` | every row | mean since the previous row (runs from before this change: since the previous eval) |
| `grad_norm`, `grad_norm_max`, `clip_frac` | every row | global gradient norm before clipping: mean and max since the previous row, and the fraction of steps clipped |
| `steps_per_s`, `tokens_per_s` | every row | training speed since the previous row (eval and sampling excluded) |
| `gpu_util`, `gpu_mem_gb`, `gpu_alloc_gb`, `gpu_reserved_gb`, `gpu_power_w`, `gpu_temp_c` | every row | mean utilisation (%), peak device memory, peak memory torch allocated, memory torch holds, mean power, peak temperature; training only |
| `weight_norm`, `update_ratio`, `grad_cos` | every `--stats-every` steps | weight L2 norm; size of that optimizer step relative to the weights; cosine between this step's gradient and the previous step's (near 1 steady, near 0 noisy, negative oscillating) |
| `val_loss`, `val_bpb`, `val_doc_loss`, `val_doc_bpb` | eval rows | see "Document handling" |

`layers.csv` has the `--stats-every` statistics per group: `emb`, `L<i>.attn`, `L<i>.mlp` and `norms`, as
`<group>.weight_norm`, `<group>.update_ratio`, `<group>.grad_norm` and `<group>.grad_cos`. `--stats-every 0` turns
the detailed statistics off (they copy the gradients and the weights on those steps).

## Document handling

The token stream is every story followed by an end-of-text token, concatenated. By default a training window is a
random slice of it with plain causal attention, so a window usually begins mid-story and a token can
attend to earlier, unrelated stories. Two flags change that; they can be combined:

- `--doc-mask`: attention stays inside each story (`Llama.attention_mask` in `llama.py`). An end-of-text token counts
  as the start token of the story after it, which is how generation begins.
- `--doc-align pack`: every window starts at a story (on its end-of-text start token) and runs on through the
  following stories, so there is no padding and only the last story in a window is cut short.
- `--doc-align pad`: one story per row, cut at the context length and padded; padding is left out of the loss but
  still costs compute, and it counts in `tokens` and `flops` (most stories are far shorter than a 512 context).

- `--attn-window 512`: attention lookback. A token sees only the last 512 tokens (itself included), whatever the
  context length, e.g. `--block-size 2048 --attn-window 512`. Generation's KV cache keeps only the window too.
- `--attn-global-every 5`: with `--attn-window`, every 5th layer (the 5th, 10th, ..., counting from 1) attends over
  the whole context instead, so with the default 6 layers the fifth layer is global and the other five are windowed
  (`L4` in `layers.csv`, which counts from 0). A global layer keeps the `--doc-mask` story mask, and its KV cache
  keeps every position.

The masks run on FlexAttention (`torch.nn.attention.flex_attention`) on CUDA: the mask is built inside the model's
forward, so it is part of the compiled graph, and blocks the mask rules out are skipped. FlexAttention is only fast
when compiled, which is why `train.py` compiles by default (`--no-compile` to turn it off); it also needs a head
size (`dim / n_heads`) of at least 16. Off CUDA the same mask is applied as a dense boolean mask instead. Without
either flag, attention is the plain causal `scaled_dot_product_attention` as before. The analytic FLOP count
shortens the attention term for `--attn-window` (not in the global layers), and is an upper bound for `--doc-mask`.

`val_loss` is measured the way the run trains (same windows, same mask), so it means something different in each
mode. `val_doc_loss` / `val_doc_bpb` are measured the same way for every run: whole val stories from their start
token, each token predicted from its own story only. Compare runs with different document options on those. Runs
named automatically get `-win512` (`-win512g5` with `--attn-global-every 5`), `-mask`, `-pack` or `-pad` appended.

## Batch size and GPU memory

Training memory is dominated by the activations kept for the backward pass, which grow with
`batch_size * block_size * dim` (a 21M-parameter model at batch 64 and context 2048 needs about 15 GiB). With
`--auto-batch` (the default), a run that hits a CUDA out-of-memory error before its first checkpoint is restarted in
a fresh process with the micro-batch halved and `--grad-accum` doubled, repeating as needed. Tokens per optimizer
step stay the same, so the FLOPs per step, the step count of a FLOP budget and the run name do not change, and the
saved config records the batch size that was used. Each failed attempt costs one start-up (model build and compile).
An out-of-memory error after the first checkpoint is not retried. `--no-auto-batch` runs in a single process and
fails on the first error.

`--grad-checkpoint` is the other way to fit: each block's activations are recomputed in the backward pass instead of
stored (`torch.utils.checkpoint`, non-reentrant), which costs about a third more compute per step and leaves the
results unchanged (gradients match to the last bit on CPU). Whether that beats a smaller micro-batch depends on how
much a small batch slows the kernels down, so measure it:

```
uv run projects/pretrain/bench_memory.py --compile
```

It runs every (micro-batch, checkpointing) case of both run configs in its own process at fixed sequences per step,
and prints tokens/s, peak allocated and reserved memory and the hours the whole FLOP budget would take. The
`--grad-checkpoint` flag does not change the run name, and `compare.py` does not count it as a config difference.

## Comparing at equal compute

`metrics.csv` has a `flops` column: cumulative training FLOPs from an analytic count of the model's matmuls
(`Config.train_flops_per_token` in `llama.py`: 2 FLOPs per multiply-add, backward = two forwards; the forward count
equals `torch.utils.flop_counter` on this model). Two flags train to a budget instead of a step count:

- `--flop-budget 2.92e16`: trains for the number of steps that fit (it replaces `max_steps`, so the LR schedule is the
  usual warmup + cosine over those steps).
- `--time-budget-s 557`: trains until that much training time has passed (eval, sampling and checkpointing excluded),
  with the cosine following elapsed time. With `--compile`, compilation counts against the budget.
