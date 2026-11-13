"""Export embeddings + projections (+ k-means clusters) to JSON for the Bun explorer.

Run from anywhere, after embed.py and visualize.py:
    uv run projects/latent_analysis/export.py --dataset afhq_v2
    uv run projects/latent_analysis/export.py --dataset stargan   # AFHQv2 + CelebA-HQ as one dataset
then start the explorer:
    bun run projects/latent_analysis/explorer/server.ts
Only needs re-running when the embeddings or projections change: k-means results
(k*/clusters_<split>.json from kmeans.py) are read by the server on every page load.

Writes:
  <repo>/data/thumbs/<dataset>/<size>/...jpg   small JPEG thumbnails (made once, then reused;
      a combined dataset uses its members' thumbnail folders)
  <repo>/runs/latent_analysis/<dataset>/<model name>_<size>/explorer/<split>.json
      one record per image: stable image id, path, thumbnail, class, and 2-D coordinates
      for every projection in viz/<split>/coords.pt
  k*/clusters_<split>.json for k-means runs made before kmeans.py wrote them
  <repo>/runs/latent_analysis/<dataset>/index.json   image path -> id (see image_index.py)
Tags added in the explorer are saved to <repo>/runs/latent_analysis/<dataset>/labels.json
(and rejections.json), keyed by image id, so they are shared across models, image sizes,
splits and k. For a combined dataset, the members' tags are copied in once, on the first run.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed import DATA_ROOT, DEFAULT_MODEL, REPO_ROOT, latent_dir, load_embeddings  # noqa: E402
from image_index import image_ids, image_root, labels_file, sync_index  # noqa: E402
from kmeans import RUNS_ROOT, paths_key, write_cluster_json  # noqa: E402
from zoo.data import stargan  # noqa: E402


def make_thumb(src: Path, dst: Path, size: int) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img = img.convert("RGB")
        img.thumbnail((size, size), Image.Resampling.LANCZOS)
        img.save(dst, "JPEG", quality=85)


def rel(p: Path) -> str:
    return str(p.resolve().relative_to(REPO_ROOT))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=stargan.ALL, default="afhq_v2")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--split", default=None, help="defaults to the train split")
    p.add_argument("--thumb-size", type=int, default=160)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    args = p.parse_args()

    split = args.split or stargan.splits(args.dataset)[0]
    src = latent_dir(args.dataset, args.model, args.image_size, args.data_root)
    run = args.runs_root / args.dataset / src.name

    coords_path = run / "viz" / split / "coords.pt"
    e = load_embeddings(args.dataset, args.model, args.image_size, split, args.data_root)
    if not coords_path.exists():
        sys.exit(f"Missing {coords_path}; run visualize.py with the same arguments first.")
    coords = torch.load(coords_path, weights_only=False)
    n = len(e["paths"])
    projections = {m: v for m, v in coords.items() if not m.startswith("_")}
    if any(len(v) != n for v in projections.values()):
        sys.exit(f"{coords_path} doesn't match the {split} embeddings; re-run visualize.py.")

    # Backfill explorer cluster files for k-means runs that predate them.
    for km_dir in sorted(run.glob("k*")):
        if (km_dir / "centroids.pt").exists() and not (km_dir / f"clusters_{split}.json").exists():
            centroids = torch.load(km_dir / "centroids.pt")
            write_cluster_json(km_dir, split, len(centroids), e["paths"], e["features"], centroids)
            print(f"wrote {km_dir / f'clusters_{split}.json'}")

    # Thumbnails mirror each real dataset's tree, so a combined dataset shares its members':
    # data/<dataset>/train/cat/x.png -> data/thumbs/<dataset>/<size>/train/cat/x.jpg
    images = [Path(p) if Path(p).is_absolute() else REPO_ROOT / p for p in e["paths"]]
    index = sync_index(args.dataset, args.data_root, args.runs_root)
    ids = image_ids(e["paths"], index, image_root(args.dataset, args.data_root))
    data_root = args.data_root.resolve()
    thumbs = []
    for img in images:
        member, *rest = img.resolve().relative_to(data_root).parts
        thumbs.append((data_root / "thumbs" / member / str(args.thumb_size)).joinpath(*rest).with_suffix(".jpg"))
    with ThreadPoolExecutor(args.workers) as pool:
        list(tqdm(pool.map(make_thumb, images, thumbs, [args.thumb_size] * n), total=n, desc="thumbnails"))

    out = run / "explorer" / f"{split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": args.dataset,
        "model": src.name,
        "split": split,
        "classes": e["classes"],
        "ids": ids,
        "paths": [rel(p) for p in images],
        "thumbs": [rel(t) for t in thumbs],
        "label": e["labels"].tolist(),
        "pathsKey": paths_key(e["paths"]),  # cluster files must match this to be shown
        # flat [x0, y0, x1, y1, ...] per projection, to keep the file small
        "projections": {m: [round(float(c), 4) for c in v.reshape(-1)] for m, v in projections.items()},
        "labelsFile": rel(labels_file(args.dataset, args.runs_root)),
    }
    out.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {out}")
    print("start the explorer with:\n  bun run projects/latent_analysis/explorer/server.ts")


if __name__ == "__main__":
    main()
