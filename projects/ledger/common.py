"""Shared plumbing for ledger: dataset loading/conversion, the chat prompt, label masking, run dirs."""

from __future__ import annotations

import json
import random
import subprocess
from dataclasses import asdict
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "runs" / "ledger"
CACHE_ROOT = REPO_ROOT / "data" / "cache" / "ledger"

MODEL = "Qwen/Qwen3.5-0.8B"
PIXELS_PER_TOKEN = 32 * 32  # Qwen3.5 vision: 16px patches, 2x2 merge -> one LLM token per 32x32 pixels
MIN_PIXELS = 256 * PIXELS_PER_TOKEN
MAX_PIXELS = 512 * PIXELS_PER_TOKEN  # image-token budget (256-512 tokens per image)

PROCEDURAL = "tasksource/procedural-extraction"
SYNTHETIC = "tasksource/synthetic-extraction"
CONFIGS = (
    "record_field_extraction",
    "lookalike_field_extraction",
    "state_update_extraction",
    "business_document_extraction",
    "structure_lookup_extraction",
)

SYSTEM_PROMPT = (
    "Return only JSON mapping field names to values. Use null for missing scalar values and [] for empty lists."
)

PACKAGES = ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "datasets", "requests", "rapidfuzz", "pytesseract", "pillow")


# ---------------------------------------------------------------- data


def convert_row(row: dict) -> dict:
    """tasksource extraction row -> {state, fields, target}. Same format for procedural- and synthetic-extraction.

    `fields` is the request schema (name -> {type, instructions[, items]}); `target` maps each name to its gold
    value: a string or None for scalar fields, a list of strings for list fields."""
    fields, target = {}, {}
    for field, answer in zip(row["fields"], row["answers"], strict=True):
        name = field["name"]
        fields[name] = {"type": field["type"], "instructions": field["instruction"]}
        values = [item["text"] for item in answer["values"]]
        if field["type"] == "list":
            fields[name]["items"] = "text"
            target[name] = values
        else:
            target[name] = None if answer["null"] else values[0]
    return {"state": row["state"], "fields": fields, "target": target}


def to_record(row: dict, source: str) -> dict:
    """convert_row, with `fields`/`target` as JSON strings: their keys differ per row, which Arrow would turn into
    one wide struct full of nulls."""
    ex = convert_row(row)
    return {
        "state": ex["state"],
        "fields_json": json.dumps(ex["fields"], ensure_ascii=False),
        "target_json": json.dumps(ex["target"], ensure_ascii=False),
        "source": source,
        "level": int(row["level"]) if row.get("level") is not None else -1,
    }


def sample_indices(n: int, count: int, seed: int) -> list[int]:
    """`count` distinct indices of range(n), fixed by `seed` (all of them, shuffled, if count >= n)."""
    return random.Random(seed).sample(range(n), min(count, n))


def hub_sha(repo: str, repo_type: str = "dataset") -> str:
    """Current commit sha of a Hub repo, so loads can be pinned and recorded."""
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(repo) if repo_type == "dataset" else HfApi().model_info(repo)
    return info.sha


def load_extraction(repo: str, config: str | None, split: str, count: int, seed: int, revision: str):
    """A seeded sample of one extraction dataset/config split, converted with `to_record` (plus its `id`)."""
    from datasets import Features, Value, load_dataset

    ds = load_dataset(repo, name=config, split=split, revision=revision)
    ds = ds.select(sample_indices(len(ds), count, seed))
    source = f"procedural/{config}" if repo == PROCEDURAL else "synthetic"
    # Fixed schema, so procedural (int32 level) and synthetic (no level) parts concatenate.
    features = Features({k: Value("int32" if k == "level" else "string") for k in ("id", "state", "fields_json", "target_json", "source", "level")})
    return ds.map(lambda r: to_record(r, source), remove_columns=[c for c in ds.column_names if c != "id"], features=features)


# ---------------------------------------------------------------- prompt / encoding


def build_messages(state: str, fields: dict, target: dict | None = None, has_image: bool = False) -> list[dict]:
    """system: the JSON instruction; user: JSON with only state + fields (after the image, if any);
    assistant (when `target` is given): json.dumps(target)."""
    user = [{"type": "image"}] if has_image else []
    user.append({"type": "text", "text": json.dumps({"state": state, "fields": fields}, ensure_ascii=False)})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": user},
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": json.dumps(target, ensure_ascii=False)}]})
    return messages


def encode_example(processor, messages: list[dict], images: list | None = None) -> dict[str, torch.Tensor]:
    """Tokenize a full conversation and build labels that keep only the assistant answer and its <|im_end|>.

    The prompt prefix (system + user + assistant header, image tokens included) is encoded separately and must be
    a token-exact prefix of the full encoding; anything else raises rather than mislabelling."""
    full = processor.apply_chat_template(messages, tokenize=False)
    prompt = processor.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    if not full.startswith(prompt):
        raise ValueError("prompt text is not a prefix of the full conversation")

    kw = {"images": images} if images else {}
    enc = processor(text=[full], return_tensors="pt", **kw)
    prompt_ids = processor(text=[prompt], return_tensors="pt", **kw)["input_ids"][0]
    ids = enc["input_ids"][0]
    p = len(prompt_ids)
    if not torch.equal(ids[:p], prompt_ids):
        raise ValueError("prompt tokens are not a prefix of the full tokens")

    tok = processor.tokenizer
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    ends = (ids[p:] == im_end).nonzero()
    if len(ends) == 0:
        raise ValueError("no <|im_end|> after the assistant answer")
    end = p + int(ends[0]) + 1  # keep the end-of-turn token, drop the trailing newline
    labels = torch.full_like(ids, -100)
    labels[p:end] = ids[p:end]
    image_pad = tok.convert_tokens_to_ids("<|image_pad|>")
    assert not (ids[p:end] == image_pad).any(), "image tokens inside the answer span"

    out = {"input_ids": ids, "attention_mask": enc["attention_mask"][0], "labels": labels}
    if images:
        out["pixel_values"] = enc["pixel_values"]
        out["image_grid_thw"] = enc["image_grid_thw"]
    return out


def parse_json_object(text: str) -> dict | None:
    """The model's JSON answer (tolerating ```json fences and text around one object), or None."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) else None


# ---------------------------------------------------------------- runs


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True)
        return out + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def package_versions() -> dict[str, str | None]:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    out["cuda"] = torch.version.cuda
    return out


def new_run_dir(stage: str, config, device: str, **extra) -> Path:
    """runs/ledger/<stage>/<timestamp>/ with config.json (config, git commit, device, package versions, extras)."""
    run_dir = RUNS_ROOT / stage / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    meta = {"config": asdict(config), "git_commit": git_commit(), "device": device, "packages": package_versions(), **extra}
    (run_dir / "config.json").write_text(json.dumps(meta, indent=2))
    print(f"Run dir: {run_dir}")
    return run_dir
