"""Loader-only measurements, independent of datasets and model execution."""

import statistics
import time


def measure(loader, batches=0):
    """Time iteration, including worker startup; zero batches means a full pass."""
    if batches < 0 or not len(loader):
        raise ValueError("batches must be nonnegative and loader must be nonempty")
    start = time.perf_counter()
    iterator = iter(loader)
    latencies = []
    samples = 0
    first_batch_seconds = None
    for _ in range(batches or len(loader)):
        batch_start = time.perf_counter()
        try:
            _images, labels = next(iterator)
        except StopIteration:
            break
        latencies.append(time.perf_counter() - batch_start)
        if first_batch_seconds is None:
            first_batch_seconds = time.perf_counter() - start
        samples += len(labels)
    elapsed = time.perf_counter() - start
    ordered = sorted(latencies)
    return {
        "samples": samples,
        "batches": len(latencies),
        "seconds": elapsed,
        "samples_per_second": samples / elapsed,
        "first_batch_seconds": first_batch_seconds,
        "batch_p50_ms": 1000 * statistics.median(latencies),
        "batch_p95_ms": 1000 * ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
    }


def benchmark_loader(loader, batches=0, repeats=3):
    """Separate the first pass from repeated passes over the same loader.

    The first pass is not a cold-disk measurement: OS file pages may already
    be resident, especially after cache construction. No model or device copy
    is timed. Nonpersistent workers restart on each pass.
    """
    if repeats < 1:
        raise ValueError("repeats must be positive")
    first = measure(loader, batches)
    passes = [measure(loader, batches) for _ in range(repeats)]
    return {
        "first_pass": first,
        "passes": passes,
        "median_samples_per_second": statistics.median(
            row["samples_per_second"] for row in passes
        ),
    }
