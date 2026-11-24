"""fastdocling settings: one config per command, training on top of the shared `RunConfig`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tyro.conf import Positional

from pages import PROMPT  # noqa: F401  (re-exported: the page prompt)
from zoo.paths import DATA_ROOT
from zoo.training import RunConfig

TARGET = "ibm-granite/granite-docling-258M"
DATA_DIR = DATA_ROOT / "fastdocling" / "distill"


@dataclass
class PrepareConfig:
    """Self-distillation data: run the target over page images with greedy decoding (vLLM) and store its DocTags tokens.

    The drafter is trained on what the target itself writes, which is exactly what greedy verification will accept."""

    dataset: str = "docling-project/DocLayNet-v1.2"  # HF dataset with page images
    split: str = "train"
    image_column: str = "image"  # PIL image column of the dataset
    n_pages: int = 20000  # pages to run through the target
    max_new_tokens: int = 4096  # per page; DocTags for a dense page is long
    target: str = TARGET
    out: Path = DATA_DIR  # parquet shards: image bytes + target token ids
    gpu_memory_utilization: float = 0.85  # vLLM


@dataclass
class TrainConfig(RunConfig):
    """Train one drafter against the frozen target. `--drafter` picks the method (see drafters/); the shared knobs below
    are read by every drafter, and a drafter ignores the ones it does not need. Keeps the best checkpoint by val/accept_len."""

    resume: str | None = None  # continue from a run's last.ckpt (same flags as that run): restores weights, optimizer, schedule and step; a new run id
    drafter: str = "mtp"  # registry name in drafters/: mtp, eagle3, ...
    target: str = TARGET
    data: str = str(DATA_DIR)  # prepare's output directory (a str: the config is saved as JSON)
    val_pages: int = 200  # held out (by page) for checkpoint selection
    k: int = 4  # draft length: tokens proposed per verification pass
    n_layers: int = 1  # depth of the drafter (decoder layers or head blocks)
    feature_layers: tuple[int, ...] = (3, 15, 27)  # target hidden states fed to the drafter (low, mid, high)
    anchors: int = 128  # blocks per page per step (dflash, dspark); 4x as many at validation, with fixed positions
    loss_gamma: float = 0.0  # decay of the per-position loss weight exp(-j / gamma) (dflash, dspark); 0 = (k + 1) / 2
    markov_rank: int = 256  # rank of dspark's previous-token bias head
    max_len: int = 3072  # tokens per sequence (image tokens + prompt + DocTags), longer pages are dropped
    batch_size: int = 4
    lr: float = 1e-3  # peak of the schedule
    weight_decay: float = 0.0  # AdamW
    ce_chunk: int = 4096  # rows per cross-entropy chunk over the 100k vocab (memory vs speed)
    epochs: int = 3
    compile: bool = False  # off by default: the valid positions per batch change shape every step, which would recompile every step
    lr_schedule: Literal["constant", "cosine", "linear", "onecycle"] = "cosine"
    warmup: int = 200
    log_every: int = 20  # steps between logged train metrics


@dataclass
class BenchConfig:
    """Greedy decoding with and without each drafter on the HF target (batch 1, KV cache): speed-up, tokens per target pass, and
    whether the output equals plain greedy decoding. In-domain pages (`val`, held out from training) and out-of-domain tasks."""

    ckpts: Positional[tuple[Path, ...]]  # drafter checkpoints to compare (each run's best.ckpt)
    tasks: tuple[str, ...] = ("val", "chart", "formula", "code")  # val = held-out DocLayNet pages; the rest are out of domain
    n_pages: int = 20  # per task
    k: int = 4  # draft length (capped at what the drafter was trained for)
    max_new_tokens: int = 1024  # per page
    dtype: str = "bfloat16"  # bfloat16 for speed; float32 is the exactness check (bf16 verification is not bit-identical to 1-token decoding)
    target: str = TARGET
    data: str = str(DATA_DIR)  # prepare's output; the first pages are the validation pages
    out: Path = DATA_ROOT / "fastdocling" / "bench.json"  # results, one record per (task, method)


@dataclass
class VllmBenchConfig:
    """Throughput of the target in vLLM on the same pages as `bench` (batch 1, one request at a time), without a drafter and with
    vLLM's built-in n-gram drafter. Runs `vllm_bench.py` in its own uv environment, one engine per setting."""

    tasks: tuple[str, ...] = ("val", "chart", "formula", "code")
    n_pages: int = 20  # per task
    max_new_tokens: int = 1024  # per page
    specs: tuple[str, ...] = ("none", "ngram")  # none = plain decoding; ngram = vLLM prompt-lookup speculation, k = 4
    target: str = TARGET
    data: str = str(DATA_DIR)
    out: Path = DATA_ROOT / "fastdocling" / "vllm_bench.json"
    gpu_memory_utilization: float = 0.85
