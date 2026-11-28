"""Turn a source (image folder, table file, Hugging Face dataset) into the items table: `id`, `value`, extra columns."""

from pathlib import Path

import numpy as np
import polars as pl

from config import EmbedConfig

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
SCALARS = (pl.String, pl.Boolean, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
           pl.Float32, pl.Float64)  # fmt: skip
TABLES = {".jsonl": pl.read_ndjson, ".ndjson": pl.read_ndjson, ".json": pl.read_json, ".csv": pl.read_csv,
          ".parquet": pl.read_parquet}  # fmt: skip


def load_source(cfg: EmbedConfig, out_dir: Path) -> pl.DataFrame:
    """The items of every source, stacked: `id` (0..N-1), `value` (an image path or the text), `source` (which source it
    came from) and the sources' other scalar columns (null where a source lacks one). A folder's sub-folder names
    become a `label` column; images of a Hugging Face dataset are saved under `out_dir/images/<source>/`."""
    names = [_source_name(s) for s in cfg.source]
    if len(set(names)) != len(names):
        raise SystemExit(f"sources need distinct names, got {names}")
    parts = [_load_one(src, name, cfg, out_dir) for src, name in zip(cfg.source, names)]
    df = pl.concat(parts, how="diagonal_relaxed")
    return df.with_row_index("id").with_columns(pl.col("id").cast(pl.Int64))


def _source_name(source: str) -> str:
    """`/data/celeba_hq` -> `celeba_hq`, `merkol/ffhq-256` -> `ffhq-256`."""
    return Path(source).name or source


def _load_one(source: str, name: str, cfg: EmbedConfig, out_dir: Path) -> pl.DataFrame:
    src = Path(source).expanduser()
    col = cfg.column or cfg.modality
    if src.is_dir() and not any(src.rglob("*.parquet")):
        if cfg.modality != "image":
            raise SystemExit(f"{source} is a folder of images; use a table file or dataset id for text")
        files = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        df = pl.DataFrame({"value": [str(p.resolve()) for p in files]}, schema={"value": pl.String})
        labels = [p.parent.relative_to(src).as_posix() for p in files]
        if any(labels):
            df = df.with_columns(label=pl.Series(labels, dtype=pl.String))
    elif src.suffix.lower() in TABLES:
        df = _with_value(TABLES[src.suffix.lower()](src), col)
    else:  # a Hugging Face dataset id, or a local folder of parquet files
        df = _load_hf(source, cfg, col, out_dir / "images" / name)
    if cfg.limit and cfg.limit < len(df):
        df = df[_sample(len(df), cfg)]
    if not len(df):
        raise SystemExit(f"no items found in {source}")
    return df.rename({"source": "source_orig"} if "source" in df.columns else {}).with_columns(source=pl.lit(name))


def _sample(n: int, cfg: EmbedConfig) -> np.ndarray:
    """`cfg.limit` of `n` row indices, at random (a source is sorted by folder, so its head is one class) but in order."""
    return np.sort(np.random.default_rng(cfg.seed).choice(n, cfg.limit, replace=False))


def _with_value(df: pl.DataFrame, col: str) -> pl.DataFrame:
    if col not in df.columns:
        raise SystemExit(f"no column {col!r}; the source has {df.columns} (set --column)")
    extras = [c for c, t in df.schema.items() if c != col and isinstance(t, SCALARS)]
    # An extra column named like one of ours would clash.
    extras = [pl.col(c).alias(f"{c}_orig" if c in ("id", "value") else c) for c in extras]
    return df.select(pl.col(col).cast(pl.String).fill_null("").alias("value"), *extras)


def _load_hf(source: str, cfg: EmbedConfig, col: str, img_dir: Path) -> pl.DataFrame:
    from datasets import load_dataset

    ds = load_dataset(source, split=cfg.split)
    rows = _sample(len(ds), cfg) if cfg.limit and cfg.limit < len(ds) else np.arange(len(ds))
    ds = ds.select(rows)
    if col not in ds.column_names:
        raise SystemExit(f"no column {col!r}; the dataset has {ds.column_names} (set --column)")
    if cfg.modality == "text":
        return _with_value(ds.to_polars(), col)
    img_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    # One row at a time (ds[col] would decode the whole column), named by the row's index in the full dataset.
    for i, row in zip(rows, ds.select_columns([col])):
        img = row[col]
        path = img_dir / f"{i:07d}.png"
        if not path.exists():
            img.convert("RGB").save(path)
        paths.append(str(path))
    df = pl.DataFrame({col: paths}, schema={col: pl.String})
    rest = ds.remove_columns(col)
    if rest.column_names:  # FFHQ has no other column, and an empty table cannot be joined to
        df = pl.concat([df, rest.to_polars()], how="horizontal")
    return _with_value(df, col)
