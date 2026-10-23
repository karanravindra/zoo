# Chat pretraining

Pretrain a decoder-only GPT on a mix of datasets with a custom 8,192-token byte-level BPE tokenizer.
This is the base-model stage for the chat project. The code started from `projects/tinystories/gpt`.

## Data mix

| Share | Source | Notes |
|---|---|---|
| 55% | `HuggingFaceFW/fineweb-edu` (`sample-10BT`) | Web text. |
| 15% | `HuggingFaceTB/smoltalk` | Chat, rendered with role markers. |
| 10% | `glaiveai/glaive-function-calling-v2` | JSON tool schemas, `<functioncall>` calls and JSON tool responses. About 100M tokens exist, so this source can run dry on large budgets and the rest is renormalised. |
| 8% | `codeparrot/codeparrot-clean` | Python. |
| 7% | `HuggingFaceTB/finemath` (`finemath-4plus`) | Math web text. |
| 5% | `wikimedia/wikipedia` (`20231101.en`) | Encyclopedia. |

Weights are shares of characters, a proxy for tokens. The real token counts are written to `metadata.json` under `tokens_by_source`.
Code and JSON take slightly more tokens per character than prose, so their real shares run a little above target.

Chat and tool data are rendered as `<|role|>\n{text}<|end|>\n` turns.
The tokenizer reserves `<|endoftext|>`, `<|end|>`, `<|system|>`, `<|user|>`, `<|assistant|>` and `<|tool|>`, so SFT does not need to resize embeddings.
Sources with no validation split hold out their first rows (`holdout` in `SOURCES`) from train.

## Setup

Run these commands from the repository root:

```bash
uv sync --extra tinystories
```

## Prepare the data

```bash
uv run --extra tinystories python projects/chat/pretrain/prepare.py
```

Preparation interleaves the sources in `SOURCES`, always drawing from the one furthest below its weight, trains the tokenizer on the first `--tokenizer-chars` characters of the mix, and packs train and validation splits until `--train-tokens` and `--val-tokens` (defaults 1B and 5M).
Every document ends with `<|endoftext|>`.

Files go into `data/chat/pretrain/`: `tokenizer.json`, `train.bin`, `validation.bin` (uint16 token IDs) and `metadata.json`.

## Train and sample

```bash
uv run --extra tinystories python projects/chat/pretrain/train.py
uv run --extra tinystories python projects/chat/pretrain/generate.py --prompt "The capital of France is"
```

Outputs (checkpoints, metrics, curves) go to `projects/chat/pretrain/outputs/`, which is git-ignored.
