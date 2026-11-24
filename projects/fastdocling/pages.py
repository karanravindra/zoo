"""Benchmark pages, shared by `bench.py` (HF) and `vllm_bench.py` (vLLM, which runs in its own env): no zoo imports here."""

from __future__ import annotations

import glob
import io
from pathlib import Path

from datasets import load_dataset
from PIL import Image

PROMPT = "Convert this page to docling."  # the target's page-conversion prompt, what the drafters were trained on

# task -> (HF dataset, split, the target's prompt for it). `val` reads the pages `prepare` wrote instead.
OOD = {
    "chart": ("docling-project/SynthChartNet", "train", "Convert chart to table."),
    "formula": ("docling-project/SynthFormulaNet", "test", "Convert formula to LaTeX."),
    "code": ("docling-project/SynthCodeNet", "test", "Convert code to text."),
}


def load_pages(task: str, data: str, n_pages: int) -> tuple[list[Image.Image], str]:
    """(n_pages images, the prompt for the task). `val` = the first pages of the prepared shards, which training held out."""
    if task == "val":
        files = sorted(glob.glob(str(Path(data) / "shard_*.parquet")))
        ds = load_dataset("parquet", data_files=files, split="train")
        return [Image.open(io.BytesIO(ds[i]["image"])).convert("RGB") for i in range(n_pages)], PROMPT
    name, split, prompt = OOD[task]
    stream = load_dataset(name, split=split, streaming=True).shuffle(seed=0, buffer_size=500)
    return [ex["images"][0].convert("RGB") for ex in stream.take(n_pages)], prompt
