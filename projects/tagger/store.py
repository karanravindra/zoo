"""Where a dataset's files live, and reading/writing them: `<DATA_ROOT>/tagger/<name>/`.

  items.parquet       one row per item: `id` (0..N-1), `value` (image path or text), plus the source's scalar columns
  meta.json           modality, source, value column
  emb-<model>.npy     (N, D) float32 unit-norm embeddings, row i = item id i
  umap-<model>-....npy (N, 2) UMAP of the embeddings (see projection.py)
  tags.parquet        long format, one row per (id, tag): an item can carry several tags
"""

import json
import os
import re
from pathlib import Path

import numpy as np
import polars as pl

from zoo.paths import DATA_ROOT

ROOT = DATA_ROOT / "tagger"
TAG_SCHEMA = {"id": pl.Int64, "tag": pl.String}


def dataset_dir(name: str) -> Path:
    return ROOT / name


def list_datasets() -> list[str]:
    return sorted(p.parent.name for p in ROOT.glob("*/items.parquet"))


def slug(model: str) -> str:
    """A model id as a file-name part: `google/siglip2-base` -> `google-siglip2-base`."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-")


def list_models(name: str) -> list[str]:
    return sorted(p.name[len("emb-") : -len(".npy")] for p in dataset_dir(name).glob("emb-*.npy"))


def resolve_model(name: str, model: str) -> str:
    """`model` as the slug of an existing embedding; "" picks the only one."""
    have = list_models(name)
    if model:
        model = slug(model)
        if model not in have:
            raise SystemExit(f"{name} has no embedding {model!r}; it has {have}")
        return model
    if len(have) != 1:
        raise SystemExit(f"{name} has {len(have)} embeddings {have}; pick one with --model")
    return have[0]


def load_items(name: str) -> pl.DataFrame:
    return pl.read_parquet(dataset_dir(name) / "items.parquet")


def load_meta(name: str) -> dict:
    return json.loads((dataset_dir(name) / "meta.json").read_text())


def load_embeddings(name: str, model: str) -> np.ndarray:
    return np.load(dataset_dir(name) / f"emb-{model}.npy")


def load_tags(name: str) -> pl.DataFrame:
    path = dataset_dir(name) / "tags.parquet"
    return pl.read_parquet(path) if path.exists() else pl.DataFrame(schema=TAG_SCHEMA)


def save_tags(name: str, tags: pl.DataFrame) -> None:
    """Atomic, so a crash mid-write cannot lose the tags."""
    path = dataset_dir(name) / "tags.parquet"
    tmp = path.with_suffix(".tmp")
    tags.select("id", "tag").unique().sort("id", "tag").write_parquet(tmp)
    os.replace(tmp, path)


def load_thumbs(name: str, items: pl.DataFrame, size: int = 192, quality: int = 75) -> list[str]:
    """One `data:image/jpeg` URI per item, for dtour's hover tooltip (a browser cannot open our file paths). Cached in
    `thumbs-<size>.parquet`; every thumbnail is inlined in the page, so this suits thousands of items, not millions."""
    import base64
    import io
    from concurrent.futures import ThreadPoolExecutor

    from PIL import Image

    path = dataset_dir(name) / f"thumbs-{size}.parquet"
    if path.exists():
        have = pl.read_parquet(path)["thumb"].to_list()
        if len(have) == len(items):
            return have

    def one(p: str) -> str:
        img = Image.open(p).convert("RGB")
        img.thumbnail((size, size))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=quality)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    with ThreadPoolExecutor(8) as pool:
        thumbs = list(pool.map(one, items["value"].to_list()))
    tmp = path.with_suffix(".tmp")
    pl.DataFrame({"thumb": thumbs}).write_parquet(tmp)
    os.replace(tmp, path)
    return thumbs


def load_image(path: str, size: int):
    """The image at `path` as a `size`-pixel square (for the grids; the plots' hover uses the cached thumbnails)."""
    from PIL import Image

    return Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
