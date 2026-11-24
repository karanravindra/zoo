# fastdocling

A test bench for drafter models that speed up granite-docling-258M with speculative decoding, mainly on Apple silicon (MLX),
and on vLLM where the method allows. A drafter proposes `k` tokens, the target checks them all in one forward pass, and the
longest matching prefix is kept (plus one free token), so the output is identical to plain greedy decoding.

| | |
| --- | --- |
| Target | `ibm-granite/granite-docling-258M` (Idefics3: SigLIP2 + a 30-layer, 576-wide Llama), frozen |
| W&B project | `fastdocling`, group = the drafter's name |
| Data | self-distilled DocTags in `DATA_ROOT/fastdocling/distill/current/` (page PNG bytes + the target's greedy token ids) |
| Local run files | `<RUNS_ROOT>/fastdocling/<run id>/` |
| Extra dependencies | none in the project env. `prepare` runs `generate.py` in its own uv env (`vllm`, Linux GPU), because current vLLM pins an older torch than the project |

```
uv run projects/fastdocling/cli.py prepare                    # target generates DocTags for n_pages pages (vLLM)
uv run projects/fastdocling/cli.py train --drafter mtp        # train a drafter on the frozen target's hidden states
```

| Command | What it does |
| --- | --- |
| `prepare` | Runs `generate.py` (own env, first run installs vLLM). Greedy-decodes pages from `--dataset` (default DocLayNet-v1.2; check `--image-column`) with the target in vLLM, in shards of 500. |
| `train` | Trains `--drafter` on target hidden states over those sequences. Logs `train/loss`, `val/loss`, `val/acc@1..k` and `val/accept_len`; checkpoints the best by `val/accept_len`. |

`val/accept_len` is the mean number of draft tokens accepted per target pass, teacher-forced over every response position
(so tokens per pass is that + 1). It is the quantity the checkpoint is chosen on, and the one to compare drafters by before timing them.

`prepare` stages the parquet shards and a manifest of the target and generation settings, then switches `current`
when the build is ready. Training and benchmarks use that build. Legacy flat shard directories remain readable.
Old builds stay on disk so active readers can finish. Training truncates responses at `--max-len` and drops pages
with fewer than two response tokens after truncation.

## Drafters

Each is one file in `drafters/` registered with `@register("name")`; `--drafter name` picks it. All share `TrainConfig`'s knobs
(`k`, `n_layers`, `feature_layers`, ...) and read what they need.

| Name | Status | Idea |
| --- | --- | --- |
| `mtp` | done | `k` independent heads on the target's final state + the next token's embedding; the frozen LM head is shared. |
| `eagle3` | done | One decoder layer over fused low/mid/high target features (`--feature-layers`) plus the next token's embedding, unrolled `k` steps on its own hidden states (training-time test). `--n-layers` is unused. |
| `dflash` | done | Block-diffusion drafting (arXiv 2602.06036): anchor + `k` mask tokens in one pass, target features injected as keys/values in every layer, position-weighted loss, random anchors per page. `--n-layers` is the depth. |
| `dspark` | done | DFlash backbone + a low-rank previous-token (Markov) head + a confidence head, trained with CE + TV + confidence loss (arXiv 2607.05147). `--markov-rank` sets the head size (most of its parameters). The confidence scheduler is not implemented yet. |
| `eagle`, `dflash2` | planned | DFlash 2 exists only as a blog post and model cards; its training objective is unpublished. |

To add one: subclass `Drafter` (`drafters/base.py`), implement `forward(feats) -> DraftOut` (loss + per-offset matches) and
`propose(...)`, decorate it. `Features` gives you the target's hidden states, final state, token ids, response mask and
embedding; `shifted_targets` and `chunked_ce` do the 100k-vocab bookkeeping.

Notes:
- The drafter at position t knows the target's state through t and the next token x[t+1] (the target's own, free after each
  verification), and proposes x[t+2..t+1+k]. Training includes the last prompt position, where drafting starts at inference.
- The target is held outside the LightningModule's parameters, so checkpoints, EMA and param stats cover the drafter only.
- `--compile` is off by default: the valid positions per batch change shape every step, which would recompile every step.
- vLLM's greedy output and HF's can differ in the last bits (bf16, kernels). A token that flips there costs a little acceptance, never correctness.

Files:
- `cli.py`: the entry point. `generate.py`: the vLLM worker `prepare` launches. `config.py`: `PrepareConfig`, `TrainConfig`. `data.py`: `prepare_data`, `DistillData`.
- `module.py`: `LitDrafter`. `drafters/`: the interface, the registry and one file per method.
- `sweeps/drafters.yaml`: method x `k` x `n_layers` (for `mtp`). `sweeps/eagle3.yaml`: `k` x `lr`.

## Benchmark

```
uv run projects/fastdocling/cli.py bench <ckpt> [<ckpt> ...] [--tasks val chart formula code] [--n-pages 20] [--k 4]
uv run projects/fastdocling/cli.py bench <ckpt> ... --dtype float32 --tasks val --n-pages 5   # exactness check
```

Batch-1 greedy decoding on the HF target with a KV cache (latency), plain decoding as the baseline and each drafter proposing `k`
tokens per verification pass. Reports tokens per target pass, decode tokens/s, speed-up (a throughput ratio, for decode only and
end to end) and the share of pages whose output equals plain greedy. Drafters decode with their cached path (`propose_cached`).
- `val`: held-out DocLayNet pages (the first `val_pages` of the prepared shards), the training distribution.
- `chart`, `formula`, `code`: out of domain, from SynthChartNet / SynthFormulaNet / SynthCodeNet with the target's own prompts
  ("Convert chart to table.", "Convert formula to LaTeX.", "Convert code to text."): different images and a different output syntax.
- Exactness: in float32 speculative output is token-for-token identical to plain greedy (checked), in bf16 it is not, because a
  (k+1)-token verification pass does not give bit-identical logits to one token at a time and near-ties flip. The `exact` column
  in a bf16 run measures that numerical drift, not a bug.

`vllm-bench` measures the target's throughput in vLLM on the same pages (own uv env, like `prepare`): plain decoding and vLLM's
built-in n-gram drafter. None of the four drafters runs in vLLM here: vLLM main has `eagle3`, `dflash` and `dspark` methods, but its
Idefics3 class (granite-docling) does not expose the target hidden states they read, our checkpoints are not in its drafter formats,
`mtp` has no method for a custom head, and DSpark has per-model paths with none for a Llama text model. That is from vLLM's docs and
source, not from running them.

To come: the `dspark` confidence scheduler and its calibration (they need `bench`'s wall-clock numbers), the MLX loop + `export`, and a vLLM export for the EAGLE family.
