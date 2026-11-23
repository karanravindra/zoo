# ledger

Typed field extraction: given a text (`state`) and a schema of fields (`text` = one value, `list` = every matching
value), return the field values as JSON, with null for a missing scalar and `[]` for an empty list. This is the
request format of Cloudflare's [Clairic](https://developers.cloudflare.com/workers-ai/models/clairic/) model.

The project does two things:
- **Fine-tune our own model.** A LoRA fine-tune of `Qwen/Qwen3.5-0.8B` (`common.MODEL`), a small VLM with hybrid
  attention: Gated-DeltaNet linear-attention layers plus a full-attention layer every 4th layer. LoRA r=16 goes on
  every decoder projection (`q/k/v/o_proj`, `in_proj_qkv`, `in_proj_z`, `out_proj`, `gate/up/down_proj`). The vision
  tower stays frozen. `--quantize` turns it into QLoRA (4-bit NF4 base). It trains on
  [tasksource/procedural-extraction](https://huggingface.co/datasets/tasksource/procedural-extraction) and
  [tasksource/synthetic-extraction](https://huggingface.co/datasets/tasksource/synthetic-extraction).
- **Evaluate** Clairic and our model on the same held-out test examples, with the same scoring.

Needs a CUDA GPU. Run from anywhere with `uv run --group ledger` (the `ledger` dependency group holds the ledger-only packages: peft, bitsandbytes, accelerate, flash-linear-attention, pytesseract, rapidfuzz, requests). `flash-linear-attention` (Triton) provides the Gated-DeltaNet
kernels. Without it, transformers falls back to a PyTorch reference that is about 3x slower. `causal-conv1d` isn't
installed, so its PyTorch fallback is used.

| Script | What it does |
| --- | --- |
| `train.py` | LoRA fine-tune (`--quantize` for QLoRA). Writes to `runs/ledger/train/<timestamp>/`: `config.json` (with model and dataset revisions and package versions), `sample_ids.json`, `metrics.csv`, `checkpoints/`, `adapter/` (LoRA adapter, processor and `ledger.json`), `preview.jsonl` and `results.json` (final validation loss, peak GPU memory, over-length rejections per source, preview score). |
| `merge_adapter.py` | Merges a LoRA adapter into its base model and writes `<run>/merged/` for vLLM. Checks the weight names against the Hub checkpoint. |
| `evaluate.py` | Scores `--backend openai` (a vLLM server), `--backend clairic`, or `--backend local [--adapter <run>/adapter]` (in-process HF generate, a fallback) on test sets. Writes to `runs/ledger/eval/<timestamp>/`: `config.json`, `sample_ids.json`, `predictions.jsonl`, `results.json` and `scores.png`. |

`vllm/` is a separate uv project for the vLLM server. vLLM 0.33 pins torch 2.14 and transformers <5.21, which
clash with the repo's environment.

Modules: `common.py` (dataset conversion, prompt, label masking, run dir wrapper over `zoo.runs.new_run_dir`), `model.py` (loading the model and
adapter, greedy generation), `scoring.py` (metrics).

## Workflow

```bash
# 1. Check the data without loading the model: token lengths, over-length rejections per source,
#    and one example's supervised span.
uv run --group ledger projects/ledger/train.py --data-only

# 2. First training run: 20 optimizer steps, then check peak memory (printed, and in results.json)
#    and the predictions in preview.jsonl.
uv run --group ledger projects/ledger/train.py --max-steps 20

# 3. Full run (1 epoch, about 6.9k examples, effective batch 16 = 8 x 2 accumulation, about 430 steps).
uv run --group ledger projects/ledger/train.py

# 4. Merge the adapter for serving.
uv run --group ledger projects/ledger/merge_adapter.py --adapter runs/ledger/train/<run>/adapter

# 5. Serve a model with vLLM (continuous batching, paged attention, CUDA graphs), in its own env.
#    Base: Qwen/Qwen3.5-0.8B --revision <sha from adapter/ledger.json>.  Fine-tuned: runs/ledger/train/<run>/merged
uv run --project projects/ledger/vllm vllm serve runs/ledger/train/<run>/merged \
    --served-model-name qwen3.5-0.8b-ledger --max-model-len 8192 --gpu-memory-utilization 0.85 \
    --limit-mm-per-prompt '{"image": 1, "video": 0}' \
    --mm-processor-kwargs '{"size": {"shortest_edge": 262144, "longest_edge": 524288}}'

# 6. Evaluate against the server (all backends use the same seeded test examples).
uv run --group ledger projects/ledger/evaluate.py --backend openai
export CLOUDFLARE_ACCOUNT_ID=... CLOUDFLARE_AUTH_TOKEN=...
uv run --group ledger projects/ledger/evaluate.py --backend clairic
```

Don't pass `-O0` to `vllm serve`. In vLLM 0.33 it also disables CUDA graphs, and eager decoding of a 0.8B model
is limited by CPU kernel launches: about 1k generated tokens/s at roughly 20% GPU, versus about 5k tokens/s at 99%.
Startup takes about 105 s (of which 15 s is `torch.compile`). The first-ever start takes longer while Triton compiles
the Gated-DeltaNet kernels. Both are cached on disk.

`evaluate.py --datasets synthetic procedural/state_update_extraction --count 20` runs a smaller subset.

## Training data and prompt

- Train and validation each take equal seeded samples per split: 1000 / 100 from each of 5 procedural configs
  (`record_field`, `lookalike_field`, `state_update`, `business_document`, `structure_lookup`), plus 2000 / 200 from
  synthetic. Official split boundaries are kept. The evaluation uses the test splits.
- `system`: the JSON instruction. `user`: JSON with only `state` and `fields`. `assistant`: `json.dumps(target)`.
- Labels cover only the assistant answer and its `<|im_end|>`. The system and user turns, padding and image tokens
  are all `-100`. Each prompt prefix is encoded on its own and must be a token-exact prefix of the full encoding.
  If it is not, the example raises an error.
- Examples longer than `--max-len` (default 2048 tokens) are rejected, never truncated. Rejection rates are
  printed per source and saved in `results.json`, and the rejected ids are in `sample_ids.json`.
- `fields` and `target` are stored as JSON strings in the `datasets` table. Their keys differ per row, and Arrow
  would merge them into one wide struct full of nulls.
- Batches are left-padded, so every answer sits at the end of its row. The collator passes `logits_to_keep` (the
  answer tail), and `tail_loss` computes cross-entropy only there. With a 248k-token vocabulary, full-sequence
  logits were the largest cost of a step. The tail loss equals the full-logits loss on the same batch.
- Speed defaults: batch 8 x 2 accumulation, length-grouped batches (`--group-by-length`, using the precomputed
  `n_tokens`), 4 collator workers, gradient checkpointing on, no 4-bit. The pseudocode's recipe (batch 1 x 16, QLoRA)
  kept the GPU about 20% busy, with one CPU core at 100%. The new defaults run at 95–100% GPU with a peak of 7.1 GiB.
  Without checkpointing, batch 4 x 2048 tokens runs out of memory on 16 GB.
- The collator already handles images (`pixel_values`, `image_grid_thw`) for later image training data. The
  processor caps an image at 256–512 visual tokens (32x32 pixels per token: 16px patches, 2x2 merge).
- Validation loss is diagnostic only. Use `evaluate.py` for accuracy.

## Evaluation

| Set | Input | Score |
| --- | --- | --- |
| `synthetic`, `procedural/<config>` (test) | Same conversion as training | Mean per-field score: scalar = exact match after normalization (case, edge punctuation, leading article, whitespace), null must be null; list = set F1 (both empty = 1). |
| `docvqa` (DocVQA validation) | Tesseract OCR of the image as `state`, plus the image itself; one `text` field with the question | ANLS against the alternative answers |
| `chartqa_numeric` (ChartQA test) | The image only; one `number` field with the question | Relaxed accuracy (within 5%) |

- 100 examples per set by default, sampled with `--seed 42`. The ids are saved, so every backend sees the same examples.
- The ChartQA gold labels are used only to pick the numeric subset (one plain number per answer). They are never
  part of the input. DocVQA OCR reads only the image, never the annotations.
- Images are re-encoded as real JPEGs under 4 MiB, at lower quality first and then smaller. Both backends get the same bytes.
- `results.json` per set: `field_score` (the metric reported on the dataset cards), `example_exact` (every field
  correct), `null_recall` (gold null predicted null), `false_null_rate` (gold value predicted null), failures,
  parse errors, truncation rate, latency and usage totals. Procedural sets also get `field_score_by_level` (levels 0–4).
- Clairic: requests retry with exponential backoff only for 429, 500, 502, 503, 504, timeouts and connection
  errors. Other errors fail immediately. A failed request is recorded with its error, scores 0, and stays in the
  denominator. Successful responses are cached in `data/cache/ledger/clairic/`, keyed by the payload hash, dataset
  revision and `PROTOCOL`. Bump `PROTOCOL` in `evaluate.py` when the inputs change. `predictions.jsonl` keeps
  `confidence`, `p_null`, `usage`, `truncated` and `state_text`.
- OpenAI backend: the prompts are the training ones (images as data URLs, before the text),
  `temperature=0`, `chat_template_kwargs={"enable_thinking": false}`. All requests from all sets go out at once
  (`--concurrency 256`), so the server never drains between sets. 800 requests take about 14–22 s.
- DocVQA OCR runs 8 Tesseract processes in parallel with `OMP_THREAD_LIMIT=1`. Without that limit, each
  Tesseract spawns its own OpenMP threads, which oversubscribes the CPU (15–65 s per page). Results are cached in
  `data/cache/ledger/ocr/` by image hash.
- Local: greedy decoding, batched (`--batch-size 16`, sorted by prompt length), non-thinking chat template (`<think>\n\n</think>` prefilled). `--quantize` defaults to
  whatever the adapter was trained with (from `adapter/ledger.json`). Output that isn't JSON counts as a parse error, and every field is scored as missing.

## Results

The first full run (`train/20261113-223025`, 1 epoch, 433 steps, about 16 min on an RTX 5070 Ti) ended at validation
loss 0.022. Below is the test-split eval through vLLM, 100 examples per set, mean per-field score (`exact` = every
field correct). Runs: base `eval/20261113-231922`, fine-tuned `eval/20261113-232159`.

| Set | Base | Fine-tuned | Exact (base / ft) | Base parse errors |
| --- | --- | --- | --- | --- |
| synthetic | 0.600 | 0.867 | 0.27 / 0.46 | 12 |
| procedural/record_field | 0.307 | 0.984 | 0.02 / 0.94 | 27 |
| procedural/lookalike_field | 0.273 | 0.977 | 0.03 / 0.91 | 30 |
| procedural/state_update | 0.350 | 0.909 | 0.09 / 0.72 | 5 |
| procedural/business_document | 0.458 | 0.909 | 0.15 / 0.68 | 18 |
| procedural/structure_lookup | 0.351 | 0.842 | 0.05 / 0.72 | 26 |
| docvqa (ANLS) | 0.227 | 0.603 | 0.19 / 0.50 | 33 |
| chartqa_numeric (relaxed acc.) | 0.060 | 0.650 | 0.06 / 0.65 | 86 |

- The fine-tuned model produced no parse errors except one on synthetic. The base model often answered with a bare value or ran
  past 512 tokens. On examples where the base output parsed, the gap stays large (record_field 0.31 vs 0.98,
  synthetic 0.69 vs 0.87, docvqa 0.34 vs 0.62), so the gain is not only formatting. ChartQA has just 14 such examples.
- The model was trained on text only, yet DocVQA and ChartQA improved. Most of the ChartQA gain is answering in
  the required JSON format.
- The weakest spot is structure_lookup at level 4 (0.61). Null handling: the fine-tuned model returns null for
  0.81–0.99 of gold nulls on the procedural sets but only 0.60 on synthetic. It wrongly returns null for a gold value
  at most 3% of the time on the text sets.
- Clairic has not been run yet (no API credentials in this environment).

## Notes

- The pseudocode targeted transformers 4.57 and Qwen2.5-VL-3B. The repo uses transformers 5.x (`dtype=`,
  `AutoModelForImageTextToText`) and Qwen3.5-0.8B.
- `docvqa` needs the `tesseract` binary (`sudo apt install tesseract-ocr`). The DocVQA validation split is
  about 1.1 GB to download.
- `--compile` follows the repo convention (`model.compile(mode="reduce-overhead", dynamic=False)`). Sequence
  lengths vary from example to example, so it recompiles often. Expect it to help little here.

## Dev runs

Pass `train.py` `--dev` to try a change without logging anything: the usual outputs go to a temporary directory that is deleted
when the script exits, so the run never appears under `runs/`.

## Training diagnostics in metrics.csv

Besides the columns above, every row has gradient and GPU columns (blank where a value was not measured):

- `grad_norm` is the one the Hugging Face Trainer reports; the other gradient columns are not logged here.
- `gpu_util` (mean %), `gpu_mem_gb` (peak device memory), `gpu_alloc_gb` (peak memory torch allocated),
  `gpu_reserved_gb`, `gpu_power_w` (mean), `gpu_temp_c` (peak), sampled in the background while training.
