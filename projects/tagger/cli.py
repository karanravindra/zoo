"""Tagger: embed a dataset, look at it, tag it.

  uv run projects/tagger/cli.py embed --name faces --source /path/to/images [more sources...] [flags]
  uv run projects/tagger/cli.py embed --name reviews --modality text --source reviews.jsonl
  uv run projects/tagger/cli.py project --name faces                      precompute the UMAP the app shows
  uv run marimo edit projects/tagger/app.py                               explore the embeddings and tag items

`<command> --help` lists a command's flags. Files go to `<DATA_ROOT>/tagger/<name>/` (see store.py).
"""

import json
from typing import Annotated

import numpy as np
import polars as pl
import tyro

from config import EmbedConfig, ProjectConfig
from data import load_source
from embedders import make_embedder, resolve
from projection import umap_2d
from store import dataset_dir, load_embeddings, resolve_model, slug


def embed(cfg: EmbedConfig) -> None:
    out = dataset_dir(cfg.name)
    out.mkdir(parents=True, exist_ok=True)
    key = slug(resolve(cfg.modality, cfg.model)[0])
    emb_path = out / f"emb-{key}.npy"
    if emb_path.exists() and not cfg.overwrite:
        print(f"{emb_path} exists (--overwrite to recompute)")
        return

    items_path = out / "items.parquet"
    items = load_source(cfg, out)
    if items_path.exists() and not cfg.overwrite:
        # A second model for the same dataset must line up row for row with the first.
        have = pl.read_parquet(items_path)
        if have["value"].to_list() != items["value"].to_list():
            raise SystemExit(f"{list(cfg.source)} no longer matches {items_path} (--overwrite to replace the dataset)")
    else:
        items.write_parquet(items_path)
        (out / "meta.json").write_text(
            json.dumps({"modality": cfg.modality, "source": list(cfg.source), "column": cfg.column}, indent=2)
        )
        (out / "tags.parquet").unlink(missing_ok=True)  # tags refer to item ids, which just changed
    print(f"{len(items):,} items")

    _, embedder = make_embedder(cfg.modality, cfg.model)
    X = embedder.embed(items["value"].to_list(), batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    np.save(emb_path, X)
    print(f"wrote {emb_path} {X.shape}")


def project(cfg: ProjectConfig) -> None:
    model = resolve_model(cfg.name, cfg.model)
    X = load_embeddings(cfg.name, model)
    xy = umap_2d(cfg.name, model, X, cfg.n_neighbors, cfg.min_dist, cfg.seed, cfg.overwrite)
    print(f"{cfg.name}/{model}: UMAP {xy.shape}")


Command = Annotated[EmbedConfig, tyro.conf.subcommand("embed")] | Annotated[ProjectConfig, tyro.conf.subcommand("project")]


def main(cmd: Command) -> None:
    match cmd:
        case EmbedConfig():
            embed(cmd)
        case ProjectConfig():
            project(cmd)


if __name__ == "__main__":
    main(tyro.cli(Command, description=__doc__))
