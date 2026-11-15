"""Merge a LoRA adapter from train.py into its base model, for serving with vLLM.

  uv run projects/ledger/merge_adapter.py --adapter runs/ledger/train/<run>/adapter

Writes `<run>/merged/` (bf16 weights, processor with the training image budget, chat template). Checks that the
saved weight names match the base checkpoint on the Hub, so vLLM's loader sees the usual layout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import tyro
from model import load_for_inference


@dataclass
class MergeConfig:
    adapter: str  # adapter dir from train.py (contains ledger.json)
    out: str | None = None  # output dir (default: next to the adapter, `<run>/merged`)


def hub_weight_names(model: str, revision: str | None) -> set[str]:
    from huggingface_hub import hf_hub_download

    index = hf_hub_download(model, "model.safetensors.index.json", revision=revision)
    return set(json.loads(Path(index).read_text())["weight_map"])


def saved_weight_names(path: Path) -> set[str]:
    from safetensors import safe_open

    names = set()
    for f in path.glob("*.safetensors"):
        with safe_open(f, "pt") as st:
            names |= set(st.keys())
    return names


def main(cfg: MergeConfig):
    out = Path(cfg.out) if cfg.out else Path(cfg.adapter).resolve().parent / "merged"
    model, processor, meta = load_for_inference(cfg.adapter, quantize=False)  # merge needs real (not 4-bit) weights
    model = model.merge_and_unload()
    model.save_pretrained(out)
    processor.save_pretrained(out)
    (out / "ledger.json").write_text(json.dumps({**meta, "merged_from": str(Path(cfg.adapter).resolve())}, indent=2))

    hub, saved = hub_weight_names(meta["model"], meta["model_revision"]), saved_weight_names(out)
    missing, extra = hub - saved, saved - hub
    # The Hub checkpoint also carries the multi-token-prediction head, which transformers doesn't load.
    missing = {k for k in missing if not k.startswith("mtp.")}
    print(f"Wrote {out}: {len(saved)} tensors; missing vs Hub: {len(missing)}, extra: {len(extra)}")
    for k in sorted(missing)[:5] + sorted(extra)[:5]:
        print("  ", k)
    if missing or extra:
        raise SystemExit("merged weight names differ from the base checkpoint")


if __name__ == "__main__":
    main(tyro.cli(MergeConfig))
