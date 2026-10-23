# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Zoo is a collection of small, independent ML training projects (MNIST, CIFAR-100, TinyStories GPT, chat pretraining GPT). Each lives at `projects/<dataset>/<task>/` with its own `README.md`. Shared code lives in `src/zoo/` (data loaders, tensor cache, `StepProgress`). Python 3.12+, managed with `uv`; run everything from the repository root.

`.opencode/skills/zoo-project-conventions/SKILL.md` is the authoritative per-project contract (output artifacts, checkpoint semantics, resume behavior). Read it before changing anything under `projects/`, and update it in the same change when you alter a convention, CLI argument, or output file.

## Commands

```bash
uv sync --extra tinystories                      # GPT projects need the optional extra
uv run --extra tinystories python projects/tinystories/gpt/prepare.py
uv run --extra tinystories python projects/tinystories/gpt/train.py
uv run --extra tinystories python projects/tinystories/gpt/generate.py --prompt "..."
uv run python projects/mnist/classify/train.py   # image projects need no extra
uv run python projects/<dataset>/<task>/model.py # prints torchinfo summary

# Tests are stdlib unittest (not pytest-style fixtures)
uv run --extra tinystories python -m unittest discover -s tests -v
uv run --extra tinystories python -m unittest discover -s tests -p 'test_tinystories.py' -v
uv run --extra tinystories python -m unittest tests.test_data.<Class>.<test>   # single test
```

Ruff is used for linting (`.ruff_cache` present); there is no configured lint script.

## Architecture

- **Projects are scripts, not packages.** Scripts in a project dir import siblings directly (`from model import GPT`, `from data import ...`) because the script's directory is on `sys.path`. Don't convert to package-relative imports. Shared code is imported as `zoo.*`.
- **Standard project contract:** `train.py` takes `--output-dir` (default `<project>/outputs/`, git-ignored) and writes `metrics.csv`, `summary.json`, `curves.png` (Matplotlib `Agg`), and an atomically-replaced resumable `checkpoint.pt`. Generated data goes to top-level `data/` (also ignored).
- **Image projects** (`mnist`, `cifar100`) use `zoo.data.make_loader` with two interchangeable backends, `pil` and `tensor-cache`, which must preprocess identically. Covered by `tests/test_data.py` and `tests/test_cifar100.py`.
- **GPT projects** (`tinystories/gpt`, `chat/pretrain`):
  - `prepare.py` trains a byte-level BPE tokenizer (Gigatoken) and writes packed uint16 `train.bin`/`validation.bin` plus `tokenizer.json` and `metadata.json`.
  - `train.py` is step-based, using Muon for hidden matrices and AdamW for the rest, warmup + cosine LR, gradient clipping, `torch.compile`, and fixed-window validation. `checkpoint.pt` holds both optimizer states, RNG state, tokenizer, and config; `best.pt` is lowest validation loss.
  - `generate.py` reads tokenizer and model config from the checkpoint; it must not need prepared data.
  - `chat/pretrain` began as a fork of `tinystories/gpt` and has diverged (multi-source data mix in `prepare.py`/`SOURCES`, English filter in `english.py`, MoE config options in `model.py`, extra `benchmark.py`/`evaluate.py`). Changes to one don't automatically apply to the other.
- `tinystories/gpt/evaluate.py` scores generations against a local System One endpoint (`laya` model), not a Hugging Face model.
