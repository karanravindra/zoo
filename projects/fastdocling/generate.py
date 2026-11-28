# /// script
# requires-python = ">=3.12"
# dependencies = ["vllm", "datasets", "pillow", "pyarrow", "transformers"]
# ///
"""Self-distillation worker for `cli.py prepare`: the target greedy-decodes page images in vLLM; image PNG bytes and the
response token ids are written as parquet shards.

Runs in its own uv environment (the inline dependencies above), not the project's: current vLLM pins an older torch than the
project's. Not meant to be run by hand; `cli.py prepare` passes the PrepareConfig as one JSON argument."""

import io
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

SHARD = 500  # pages per shard


def main(cfg: dict) -> None:
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(cfg["target"])
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": cfg["prompt"]}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    llm = LLM(model=cfg["target"], gpu_memory_utilization=cfg["gpu_memory_utilization"], limit_mm_per_prompt={"image": 1},
              max_model_len=cfg["max_new_tokens"] + 4096)
    sampling = SamplingParams(temperature=0.0, max_tokens=cfg["max_new_tokens"])
    shards = []
    stream = load_dataset(cfg["dataset"], split=cfg["split"], streaming=True)
    chunk, done = [], 0

    def flush():
        nonlocal chunk, done
        if not chunk:
            return
        outs = llm.generate([{"prompt": prompt, "multi_modal_data": {"image": im}} for im in chunk], sampling)
        if len(outs) != len(chunk):
            raise RuntimeError(f"expected {len(chunk)} generated pages, got {len(outs)}")
        rows = []
        for im, o in zip(chunk, outs):
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            rows.append({"image": buf.getvalue(), "tokens": list(o.outputs[0].token_ids), "finish": o.outputs[0].finish_reason})
        name = f"shard_{len(shards):04d}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), out / name)
        shards.append(name)
        done += len(rows)
        chunk = []
        print(f"{done}/{cfg['n_pages']} pages", flush=True)

    for ex in stream:
        chunk.append(ex[cfg["image_column"]].convert("RGB"))
        if len(chunk) == SHARD or done + len(chunk) >= cfg["n_pages"]:
            flush()
        if done >= cfg["n_pages"]:
            break
    flush()
    if not done:
        raise ValueError("the input dataset produced no pages")
    settings = {key: value for key, value in cfg.items() if key != "out"}
    (out / "manifest.json").write_text(json.dumps({
        "settings": settings, "num_pages": done, "shards": shards,
    }, indent=2))


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
