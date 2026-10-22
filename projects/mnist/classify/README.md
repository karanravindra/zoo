# MNIST classification data loading

Training supports the original `pil` pipeline and an optional `tensor-cache` backend.
Both use the same train/validation split, pixels, normalization, and random affine augmentation.

Run these commands from the repository root.

## Precompute and use the cache

```sh
# Build all three deterministic caches without running a model.
uv run python -m zoo.data.mnist --cache-dir data/mnist-tensor-cache

# Training also builds missing caches automatically.
uv run python projects/mnist/classify/train.py --backend tensor-cache

# Reuse worker processes across training epochs and validation calls.
uv run python projects/mnist/classify/train.py \
  --backend tensor-cache --num-workers 2 --persistent-workers
```

The training cache stores resized `uint8` images. Each read still applies a fresh
PIL affine transform, converts to float, and normalizes. This preserves the original
interpolation and rounding behavior. Caching augmented images would freeze the
augmentation, so the cache contains only deterministic preprocessing.

Validation and test caches store the final normalized `float32` tensors.
Validation uses the training source with its own deterministic preprocessing cache.
The validation subset remains 512 samples; all training-source evaluation images
are cached so different split seeds can reuse the cache.

`--chunk-size` controls samples per `.pt` file (default: 8192).
`--max-open-chunks` bounds each process's lazily opened memory mappings (default: 16).
Memory mapping lets the operating system load tensor pages as needed instead of
loading every chunk into Python memory. Random access across more chunks than
the mapping limit can cause repeated file opens; tune the limit for larger datasets.

Each cache has a JSON manifest and an identity derived from source pixels, labels,
preprocessing version, dependency versions, and chunk size. Builds use a temporary
directory and publish only complete caches. Changed identities create a new cache;
old caches remain available until their directories are deleted.

## Benchmark

```sh
# Full epochs, median of three passes after a separate first pass.
uv run python projects/mnist/classify/scripts/benchmark_data.py \
  --batches 0 --repeats 3 --workers 0 2

# Measure persistent workers separately.
uv run python projects/mnist/classify/scripts/benchmark_data.py \
  --batches 0 --repeats 3 --workers 2 --persistent-workers \
  --output projects/mnist/classify/outputs/dataloader_persistent_benchmark.json
```

The default run measures 100 batches per pass. `--backends`, `--splits`,
`--batch-size`, and `--threads` control the workload. Loader options also include
`--prefetch-factor` and `--pin-memory`.

The JSON report includes setup time, first-batch latency, median and 95th-percentile
batch latency, samples per second for every pass, and environment/configuration.
Setup includes dataset construction and cache building or reopening. It is shared
by all splits for that backend/worker configuration, not charged once per split.
The first pass is not a cold-disk measurement because operating-system file pages
may already be cached. Measurements include iterator/worker startup, but exclude
model execution and device transfer. Nonpersistent workers restart for every pass.

### Measured results

Measured on macOS ARM64 with batch size 128, four PyTorch threads, full epochs,
and the median of three repeated passes. Training uses 59,488 samples; test uses
10,000 samples. Results vary by machine and workload.

| Workers | Persistent | Split | Original PIL (images/s) | Tensor cache (images/s) | Speedup |
| ---: | :---: | --- | ---: | ---: | ---: |
| 0 | No | Train | 19,780 | 21,435 | 1.08× |
| 0 | No | Test | 30,772 | 694,861 | 22.58× |
| 2 | No | Train | 16,398 | 16,370 | 1.00× |
| 2 | No | Test | 8,079 | 8,307 | 1.03× |
| 2 | Yes | Train | 36,541 | 38,208 | 1.05× |
| 2 | Yes | Test | 55,595 | 585,885 | 10.54× |

Setup, including building the three caches, took 3.82 seconds in the measured run. Reopening them
took about 0.06 seconds in later configurations. Worker startup took about one
to two seconds, which explains the nonpersistent-worker results for this small
dataset. PIL affine augmentation still dominates cached training. These are loader
speedups; they do not measure end-to-end model training speed.

## Shared modules and extension points

Reusable code lives in `src/zoo/data/`:

| Module | Responsibility |
| --- | --- |
| `cache.py` | Dataset-independent chunk writing, manifests, lazy tensor reads, and bounded per-worker mappings |
| `transforms.py` | Apply runtime transforms independently of storage |
| `loaders.py` | Loader construction and worker, prefetch, and pinned-memory settings |
| `benchmark.py` | Dataset-independent loader timing and repeated-pass summaries |
| `mnist.py` | MNIST preprocessing, source identity, splits, and backend factories |

For another dataset, give `TensorChunks.prepare()` a deterministic map-style dataset
returning fixed-shape image tensors and integer labels. Supply metadata identifying
the source and preprocessing. Wrap the result in `TransformDataset` for runtime
augmentation. Returned tensor views must not be mutated in place.

For another MNIST optimization, add a factory to `mnist.BACKENDS`. A factory takes
`(source, training, config)` and returns a map-style dataset. Training and benchmarks
then select it through the same interface. Keep storage changes separate from runtime
transforms so future batched augmentation or alternate storage can be measured independently.

## Verification

```sh
uv run python -m unittest discover -s tests -v
```

Tests cover chunk boundaries, shuffled reads, bounded mapping eviction, cache reuse
and invalidation, interrupted builds, missing chunks, worker-process loading, and
exact output equality with the PIL pipeline. The equality test resets the random
seed for each training sample and separately checks that augmentation stays random.
