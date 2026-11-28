# Tagger

Take a dataset (images or text), embed it, look at the embeddings, and tag items. Embedding is a CLI step;
looking and tagging happen in a marimo app. No training, so no Lightning or W&B here.

| | |
| --- | --- |
| Data | one or more `--source`s: a folder of images (or of parquet files), a `.jsonl`/`.json`/`.csv`/`.parquet` file, or a Hugging Face dataset. Several sources make one corpus |
| Files | `<DATA_ROOT>/tagger/<name>/` (see `store.py` for the layout) |
| Extra dependencies | none |

```
uv run projects/tagger/cli.py embed --name faces --source /path/to/images                     # images (default SigLIP 2)
uv run projects/tagger/cli.py embed --name faces --source data/afhq_v2 data/celeba_hq data/ffhq_256   # several sources, one corpus
uv run projects/tagger/cli.py embed --name reviews --modality text --source reviews.jsonl    # text (default bge-small)
uv run projects/tagger/cli.py embed --name reviews --modality text --source reviews.jsonl --model bge-base   # a second embedding
uv run projects/tagger/cli.py embed --name faces --source data/afhq_v2 data/celeba_hq data/ffhq_256 --limit 2000 --model dinov2-base   # same items, DINOv2 (appearance, not concepts)
uv run projects/tagger/cli.py embed --name faces --source data/afhq_v2 data/celeba_hq data/ffhq_256 --limit 2000 --model dinov3-base   # DINOv3; gated on the Hub, accept its license first
uv run projects/tagger/cli.py project --name faces --n-neighbors 15                           # precompute a UMAP (optional)
uv run marimo edit projects/tagger/app.py -- --name faces                                     # explore and tag
```

Files:
- `config.py`: `EmbedConfig` and `ProjectConfig`, the two commands' flags.
- `data.py`: `load_source` turns the sources into one items table: `id`, `value` (image path or text), `source` (which
  source the item came from; named after its folder or dataset id, `--limit` applies to each), and the sources' other scalar columns (a folder's sub-folder names become `label`). Images of a Hugging Face dataset are saved
  under the dataset folder.
- `embedders.py`: `ImageEmbedder` (SigLIP/CLIP-style `get_image_features`) and `TextEmbedder` (CLS or mean pooling),
  both returning unit-norm float32 arrays. `--model` takes a registry key or any Hugging Face id. A new modality is
  a class with `embed(values, ...)` plus an entry in `make_embedder`.
- `projection.py`: PCA and a cached cosine UMAP.
- `probe.py`: tasks (named sets of tags) and the logistic-regression probe trained on them.
- `store.py`: dataset paths, listing, and reading/writing items, embeddings and tags.
- `cli.py`: `embed` and `project`.
- `app.py`: the marimo app.
  - **Views**: a dtour tour over the top 8 PCs linked to a UMAP; lasso (or click a legend entry) to select; hover a point
    for its picture, tags and prediction. *Compare embedding* adds a UMAP of a second embedding of the same items,
    linked to the others. *Zoom in* refits PCA and UMAP on the selection alone. *Find similar* lists the items
    nearest the selection's centroid and can select them.
  - **Tagging**: add or remove a tag on the selection. Color by any low-cardinality column, by tag, or by the probe's
    prediction; press Recolor to pick up new tags and predictions (rebuilding the plots drops the selection).
  - **Probes**: a task is a named set of tags used as classes (`species` = lion, tiger, jaguar), saved in
    `tasks.json`. *Train* fits a logistic regression per task on the current embedding, over the items carrying
    exactly one of the task's tags, and reports cross-validated accuracy and per-class precision/recall. Then it can
    select the most uncertain unlabeled items (the best ones to label next), select confident predictions of a class,
    or accept its predictions for the current selection as tags. Cross-validated accuracy is not an independent test
    if the tags came from lassoing clusters of the same embedding.

## Tags

`tags.parquet` holds one row per `(id, tag)`, so an item can have several tags. It is rewritten atomically on every
change. Re-embedding a dataset with `--overwrite` renumbers the items, so it deletes the tags.

## Ideas not built yet

Text search over image embeddings (SigLIP's text tower shares the space), crop embeddings for local attributes (eye\ncolor), a vision-language pre-labeler, tag provenance (human vs. accepted prediction), clustering, export of tagged items.
