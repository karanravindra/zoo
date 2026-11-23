"""Shared bits for lfm_sae: the model id, hook names, and reading activation shards."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import torch

MODEL = "LiquidAI/LFM3-VL-2B"
MODEL_REVISION = "23387beda43c60310348e83e25f78f33c3140d3a"  # pinned: the checkpoint runs its own modeling code


def hook_name(site: str, layer: int) -> str:
    return f"{site}.{layer}"


def hub_sha(repo: str, repo_type: str = "dataset") -> str:
    """Current commit sha of a Hub repo, so loads can be pinned and recorded."""
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(repo) if repo_type == "dataset" else HfApi().model_info(repo)
    return info.sha


# ---------------------------------------------------------------- reading an extraction run


def shard_paths(run: Path) -> list[Path]:
    return sorted((Path(run) / "shards").glob("*.safetensors"))


def iter_shards(run: Path, keys: list[str]) -> Iterator[dict[str, torch.Tensor]]:
    """Each shard of an extraction run, loaded with only `keys` (hook names and/or token_ids, doc_ids, positions,
    sources)."""
    from safetensors import safe_open

    for path in shard_paths(run):
        with safe_open(path, framework="pt") as f:
            yield {k: f.get_tensor(k) for k in keys}


def load_stats(run: Path, hook: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(mean [d], mean squared norm []) of a hook point over the whole run."""
    from safetensors.torch import load_file

    stats = load_file(Path(run) / "stats.safetensors")
    return stats[f"{hook}.mean"], stats[f"{hook}.sq_norm_mean"]


def run_hooks(run: Path) -> list[str]:
    return json.loads((Path(run) / "config.json").read_text())["hooks"]
