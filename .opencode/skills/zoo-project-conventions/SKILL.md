---
name: zoo-project-conventions
description: Use when adding, changing, reviewing, or debugging a project under projects/ in this Zoo repository. Follow the repository-wide and project-specific conventions for training scripts, data, outputs, plotting, checkpoints, documentation, and tests.
---

# Zoo Project Conventions

Apply these conventions to work under `projects/`. Inspect the relevant project
before changing it; this document records the current contract, not a reason to
override a project-specific behavior.

## Keep This Skill Current

When a change adds, removes, or changes a convention documented here, update
this skill in the same change. This includes CLI arguments, output artifacts,
checkpoint semantics, data preparation, shared utilities, test commands, and
project layout. Review this skill during code review for any change under
`projects/` or `src/zoo/`.

## Repository-Wide Rules

- Put independent experiments at `projects/<dataset>/<task>/`. Keep a concise
  `README.md` beside each project.
- Put reusable data and training support in `src/zoo/`, not in duplicated
  project helpers. Image projects use `zoo.data.make_loader` and their dataset
  module's argument/config/split helpers.
- Each `model.py` supports direct execution and uses `torchinfo.summary` to
  print its default model with a native-resolution example input.
- Training commands accept an explicit `--output-dir`, create it with parents,
  and default it to the project's `outputs/` directory. Data and outputs are
  generated artifacts and are ignored by Git.
- Training scripts seed PyTorch, validate CLI values with `parser.error`, and
  write machine-readable artifacts. Use `StepProgress` for concise progress
  rate and ETA reporting.
- Every training project writes `metrics.csv`, `summary.json`, and a latest
  `checkpoint.pt`. Preserve the documented resume behavior when changing these
  files.
- Every training project writes `curves.png` from recorded metrics. Use the
  noninteractive Matplotlib `Agg` backend so plots work in CI and headless
  runs. Plot only metrics meaningful to the project.
- Write output files atomically when they represent resumable state. Existing
  projects save to a temporary path then replace checkpoints.
- Use the standard library `unittest` suite under `tests/`. Prefer compact,
  deterministic fake datasets and temporary directories over real downloads.
- Update the affected project README whenever commands, defaults, outputs, or
  behavior change.

## MNIST Classifier

Location: `projects/mnist/classify/`

- Train by optimizer step in `train.py`; its default output directory is
  `projects/mnist/classify/outputs/`.
- Use `zoo.data.mnist` for configured train, validation, and test splits.
  Preserve the interchangeable `pil` and `tensor-cache` backends and their
  identical preprocessing behavior.
- Report train/validation loss and accuracy in `metrics.csv`, then evaluate on
  the test split.
- Produce `curves.png`, `confusion_matrix.png`, `predictions.png`, and
  `per_class.csv` in addition to the shared artifacts. The per-class report
  contains precision, recall, F1, and support for digits 0-9.
- Save model, optimizer, scheduler, and summary in `checkpoint.pt`.
- Keep cache and data-loader changes covered by `tests/test_data.py`.

## CIFAR-100 Classifier

Location: `projects/cifar100/classify/`

- Train by epoch in `train.py`; validation can be logged multiple times within
  an epoch. Preserve `--resume`, `--eval-only`, smoke batch limits, and
  `--stop-after-epoch` behavior.
- Use `zoo.data.cifar100` for seeded stratified splits and the shared loader
  configuration. Preserve equivalent `pil` and `tensor-cache` preprocessing.
- Record sample-weighted train and validation loss, top-1, and top-5 metrics.
  Select the best model by validation top-1, retaining the earlier model on a
  tie.
- `checkpoint.pt` is resumable epoch-boundary state. `best.pt` is
  inference-only and may be used with `--eval-only`.
- Produce `curves.png`, `confusion_matrix.png`, `confusion_matrix.csv`,
  `per_class.csv`, and `predictions.png` with named CIFAR-100 classes.
- Keep model, scheduler, split, cache, and resume behavior covered by
  `tests/test_cifar100.py`.

## TinyStories GPT

Location: `projects/tinystories/gpt/`

- `prepare.py` trains a byte-level BPE tokenizer, appends the end-of-story
  token, and writes `tokenizer.json`, packed uint16 token files, and
  `metadata.json`. Support paired local train/validation fixtures for offline
  tests.
- `train.py` trains by optimizer step with Muon for hidden matrices and AdamW
  for remaining parameters. Keep fixed-window validation, gradient clipping,
  and the warmup/cosine learning-rate schedule intact unless deliberately
  changing documented model behavior.
- Record sampled token count, train loss, validation loss, validation
  perplexity, and learning rate. `curves.png` contains training/validation
  loss, validation perplexity, and learning-rate panels.
- `checkpoint.pt` holds latest weights, both optimizer states, sampler and
  device RNG state, tokenizer, data metadata, model config, and arguments.
  `best.pt` is the checkpoint with the lowest validation loss.
- `generate.py` loads tokenizer and model configuration from the checkpoint;
  do not require prepared data for generation.
- Keep the complete local preparation, train, resume, plot, and generation
  flow covered by `tests/test_tinystories.py` when the optional dependencies
  are installed.

## Before Finishing

1. Run the focused `unittest` module for the changed project and any affected
   shared-data tests.
2. Confirm output names and README documentation match the code.
3. If this change established or changed a convention, update this skill now.
