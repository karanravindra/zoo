# TinyStories

A small Llama-style decoder (pre-norm RMSNorm, rotary embeddings, SwiGLU, bias-free linears, causal attention via
`F.scaled_dot_product_attention`, tied input/output embeddings) trained on
[`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories). Defaults: dim 256, 6 layers,
8 heads, context 256, vocab 4096, about 5.9M parameters.

Run in this order, from anywhere:

```
uv run projects/tinystories/prepare.py   # once: download, train the tokenizer, write token streams
uv run projects/tinystories/train.py
uv run projects/tinystories/sample.py --prompt "Once upon a time"
uv run projects/tinystories/compare.py                    # all runs side by side -> runs/tinystories/comparison.png
uv run projects/tinystories/bench.py                      # zero-shot benchmarks of every run -> runs/tinystories/benchmarks.csv
```

Files:
- `llama.py`: `Config` (model, optimization and runtime fields, all CLI flags), the model, checkpoint save/load.
- `prepare.py`: data prep. Trains a 4096-token byte-level BPE with `gigatoken.train_bpe` on the first
  `--vocab-docs` train stories, writes `data/cache/tinystories/tokenizer.json` (end-of-text token is id 256), then
  encodes both splits with gigatoken's file path (`TextFileSource`, parallel in Rust) and writes
  `<split>/tokens.npy` (uint16) and `<split>/meta.json` through `zoo.data.text`.
- `train.py`: step-based loop (not epochs): random windows from `zoo.data.text.TokenDataset`, AdamW with weight decay on
  2D parameters only, linear warmup then cosine LR, grad clipping, optional bf16 autocast on cuda, an eval every
  `--eval-every` steps over `--eval-batches` fixed val batches, and a text sample every `--sample-every` steps
  (independent of evals; 0 disables).
- `sample.py`: generate from a checkpoint (default: `best.pt` of the latest run). `--num-samples N` generates a batch,
  `--out FILE` also writes the samples to a file, and `--prompt ""` starts whole new stories from the end-of-text
  token. Generation uses a KV cache: one new token per step against cached keys / values, falling back to a
  sliding window once the context reaches `block_size`.
- `compare.py`: compare runs (default: all under `runs/tinystories`, or pass run dirs). The top row is val loss
  against training time, training FLOPs and `--x step|tokens|elapsed_s|flops`; below are train loss, tokens/s and a
  best-val-bpb bar chart, plus a table with parameter counts and the time and FLOPs each run took. Runs are labelled
  by the config values that differ, in short form (`ctx2048 bs64 d512 win512 g5 mask`), in one legend below the panels (batch splits from OOM restarts and a FLOP budget's `max_steps` don't count as differences), and ordered by them; runs still training are included. `--log-x` for log x axes. The loss shown is `val_doc_loss` when every run has it,
  else `val_loss` (`--metric` picks one).
- `bench.py`: zero-shot benchmarks (see "Benchmarks"), for every run or the run dirs passed. Writes `bench.json` into each
  run dir and `runs/tinystories/benchmarks.csv`, and prints a table with the chance level and a standard error.

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

`<repo>/runs/tinystories/<name>/`, where the name is built from the model shape and the budget that ends the run:
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

## Comparing at equal compute

`metrics.csv` has a `flops` column: cumulative training FLOPs from an analytic count of the model's matmuls
(`Config.train_flops_per_token` in `llama.py`: 2 FLOPs per multiply-add, backward = two forwards; the forward count
equals `torch.utils.flop_counter` on this model). Two flags train to a budget instead of a step count:

- `--flop-budget 2.92e16`: trains for the number of steps that fit (it replaces `max_steps`, so the LR schedule is the
  usual warmup + cosine over those steps).
- `--time-budget-s 557`: trains until that much training time has passed (eval, sampling and checkpointing excluded),
  with the cosine following elapsed time. With `--compile`, compilation counts against the budget.
