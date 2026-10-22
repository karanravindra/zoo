# CIFAR-100 classification

Train a compact ConvNeXt from scratch on native 32×32 RGB images and 100 fine-label classes.
Run all commands from the repository root.

## Model and data

The default model has 701,764 parameters, stage widths `32 64 128`, and depths `2 2 4`.
Each stage halves the spatial resolution. Blocks use depthwise convolution, layer normalization,
a channel MLP, and a learned residual scale. Global average pooling feeds the 100-class head.
Use `--widths` and `--depths` to configure one to five stages.

The seeded, stratified split reserves 50 images per class for validation:

| Split | Images | Images per class |
| --- | ---: | ---: |
| Training | 45,000 | 450 |
| Validation | 5,000 | 50 |
| Official test | 10,000 | 100 |

Training applies a random 32×32 crop with four pixels of zero padding and a random horizontal flip.
All splits normalize RGB channels with mean `(0.5071, 0.4867, 0.4408)`
and standard deviation `(0.2675, 0.2565, 0.2761)`.
Validation and test preprocessing is deterministic. Their transforms are independent of training transforms.

## Smoke test

This command trains on two batches and evaluates two batches per split:

```sh
uv run python projects/cifar100/classify/train.py \
  --epochs 1 --warmup-epochs 0 --batch-size 16 \
  --max-train-batches 2 --max-eval-batches 2 \
  --output-dir projects/cifar100/classify/outputs/smoke
```

Batch limits default to zero, meaning the whole split. Limited runs set `limited_run: true`
in `summary.json` and record the number of evaluated samples.
Use separate output directories for smoke runs and full training.

## Full training

```sh
# Apple GPU. Use --device cpu or --device cuda on other systems.
uv run python projects/cifar100/classify/train.py \
  --epochs 100 --warmup-epochs 5 --batch-size 128 \
  --lr 0.001 --weight-decay 0.05 --seed 42 --device mps \
  --output-dir projects/cifar100/classify/outputs/full
```

Training uses AdamW with per-step linear warmup and cosine decay.
Warmup is capped below the total step count for short runs.
`--log-every` controls validation and metric logging in optimizer steps, with an additional log at every epoch end.
Training metrics aggregate all processed samples in the current epoch. Validation metrics cover the validation split.
Top-1 accuracy measures whether the highest-scoring class matches the label.
Top-5 accuracy measures whether the label appears among the five highest-scoring classes.
Accuracies in CSV and JSON files are fractions between zero and one.

The best model is selected by validation top-1 accuracy across logged evaluations.
Ties keep the earlier model. After training, the script evaluates that model on the official test set.

## Checkpoints and evaluation

`checkpoint.pt` stores the latest epoch-end model, optimizer, scheduler, configuration,
class names, metric history, progress, and best model weights.
`best.pt` stores the selected model and reporting metadata for inference.
Both files are written through a temporary file before replacing the destination.

```sh
# Continue at the next epoch using the saved training configuration.
uv run python projects/cifar100/classify/train.py \
  --resume projects/cifar100/classify/outputs/full/checkpoint.pt \
  --device mps --output-dir projects/cifar100/classify/outputs/full

# Re-evaluate the selected model without training.
uv run python projects/cifar100/classify/train.py \
  --resume projects/cifar100/classify/outputs/full/best.pt --eval-only \
  --device mps --output-dir projects/cifar100/classify/outputs/evaluation
```

Resume restores the original architecture, epochs, warmup, seed, data settings, and batch limits.
The original epoch count defines the learning-rate schedule and stays fixed on resume.
Device, threads, logging interval, output directory, and worker settings remain configurable.
Training resume requires `checkpoint.pt`; standalone evaluation accepts either checkpoint.
The epoch and step in a checkpoint describe completed training. `best_epoch` identifies the selected model's epoch.

Use `--stop-after-epoch 1` with a larger `--epochs` value to test an interrupted run.
This saves checkpoints and exits before test evaluation. Resume starts at the next epoch.
Epoch-local seeds reproduce augmentation and shuffling with zero workers on CPU.
Persistent worker random states are not restored, so their augmentation sequence can differ after resume.

## Tensor caching and loaders

```sh
# Build deterministic caches without model execution.
uv run python -m zoo.data.cifar100

# Training also builds missing caches automatically.
uv run python projects/cifar100/classify/train.py \
  --backend tensor-cache --num-workers 2 --persistent-workers --device mps \
  --output-dir projects/cifar100/classify/outputs/cached
```

The `pil` backend is the default. Both backends use the same split and transforms.
The training cache stores RGB `uint8` tensors; every read applies a fresh PIL crop and flip before normalization.
Validation and test caches store final normalized `float32` tensors.
Validation caches all training-source images so different split seeds can reuse the same cache.

Cache identities include source pixels, fine labels, class names, normalization,
preprocessing version, dependency versions, and chunk size.
Storage uses the shared `TensorChunks` implementation with bounded per-process memory mappings.
Default cache location is `data/cifar100-tensor-cache`.
Loader options include `--num-workers`, `--persistent-workers`, `--prefetch-factor`, and `--pin-memory`.
Persistent workers require a positive worker count. Pinned memory is mainly useful with CUDA.

## Preview and benchmark

```sh
uv run python projects/cifar100/classify/scripts/preview_augments.py

# Full passes, with three timed repeats after a separate first pass.
uv run python projects/cifar100/classify/scripts/benchmark_data.py \
  --batches 0 --repeats 3 --workers 0

uv run python projects/cifar100/classify/scripts/benchmark_data.py \
  --batches 0 --repeats 3 --workers 2 --persistent-workers \
  --output projects/cifar100/classify/outputs/dataloader_persistent_benchmark.json
```

The benchmark uses `zoo.data.benchmark` and the same split factories as training.
It records setup time, first-batch latency, median and 95th-percentile batch latency,
samples per second, environment, and loader configuration.
Setup includes cache construction or reopening and is shared across splits in each configuration.
The first pass is not a cold-disk measurement. Timings exclude model execution and device transfer.
Nonpersistent workers restart each pass, so startup can dominate short benchmarks.

## Outputs

Generated files default to `projects/cifar100/classify/outputs/`, which Git ignores.

| File | Contents |
| --- | --- |
| `checkpoint.pt` | Epoch-boundary training state for resume |
| `best.pt` | Best validation model and evaluation metadata |
| `metrics.csv` | Training and validation loss, top-1, top-5, sample counts, learning rate, and elapsed seconds |
| `summary.json` | Configuration, class names, model size, progress, and final test metrics |
| `curves.png` | Training and validation learning curves |
| `confusion_matrix.png`, `confusion_matrix.csv` | Test confusion counts with class names |
| `per_class.csv` | Class ID, name, accuracy, correct count, and support |
| `predictions.png` | Denormalized RGB images with true and predicted class names |

Classes absent from a limited test subset have blank per-class accuracy.
Preview and benchmark scripts write `augmentations.png` and `dataloader_benchmark.json` by default.

## Verification and measured smoke results

```sh
uv run python -m unittest discover -s tests -v
```

Tests cover stratified balance, split disjointness, reproducibility, independent transforms,
tensor shapes, label bounds, cache equality, fresh augmentation, model backward propagation,
sample-weighted evaluation, and warmup/cosine scheduler restoration.
The checkpoint integration test compares uninterrupted and resumed CPU model weights and metrics,
then checks that standalone best-checkpoint evaluation matches the original test metrics.

Real-data smoke runs passed on CPU and MPS on macOS ARM64:

- A small CPU model (`--widths 8 16 32 --depths 1 1 1`) completed two epochs of two batches each.
  Batch size was 16; validation and test each used two batches. Its selected model scored
  3.125% top-1 and 6.25% top-5 on 32 test images. Loading `best.pt` reproduced those metrics.
- The default 701,764-parameter model completed two training batches on MPS using the tensor cache
  and two persistent workers. It evaluated 32 test images and scored 0% top-1 and top-5.
- Cache precomputation, augmentation preview, both loader backends, and two-worker cached loading passed smoke checks.

These measurements verify execution and outputs. They do not estimate full-training accuracy.
No full 100-epoch training result has been measured.
