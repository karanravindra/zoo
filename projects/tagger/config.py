"""Settings for each step. Not training configs: this project has no model to train, so no `RunConfig`."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class EmbedConfig:
    name: str  # dataset name: everything is written to <DATA_ROOT>/tagger/<name>/
    source: tuple[str, ...]  # one or more: a folder of images (or of parquet files), a .jsonl/.json/.csv/.parquet file, or a Hugging Face dataset id; several make one corpus
    modality: Literal["image", "text"] = "image"  # what the items are
    model: str = ""  # a key of embedders.IMAGE_MODELS / TEXT_MODELS, or any Hugging Face id ("" = the modality default)
    column: str = ""  # the column holding the image path or the text, in a table or HF source ("" = "image" / "text")
    split: str = "train"  # split to load from a Hugging Face dataset
    limit: int = 0  # embed a random sample of N items of each source (0 = all)
    seed: int = 0  # seed of that sample
    batch_size: int = 64  # items per forward pass
    num_workers: int = 4  # DataLoader workers decoding images
    overwrite: bool = False  # recompute an embedding that already exists (and rewrite items.parquet)


@dataclass
class ProjectConfig:
    name: str  # dataset name, as given to `embed`
    model: str = ""  # which embedding to project ("" = the only one; required if there are several)
    n_neighbors: int = 30  # UMAP neighbourhood size
    min_dist: float = 0.1  # UMAP minimum distance between points
    seed: int = 0  # UMAP seed
    overwrite: bool = False  # recompute a projection that already exists
