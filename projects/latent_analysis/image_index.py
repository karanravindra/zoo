"""Stable integer ids for a dataset's images, and the tags attached to them.

These files live at the dataset level, so tags are shared across models, image sizes,
splits and k:

  runs/latent_analysis/<dataset>/index.json
      {"dataset": name, "paths": [...]}; an image's id is its position in `paths`, and each
      path is relative to the dataset's image root (see `image_root`): data/<dataset> for a
      single dataset (e.g. "train/cat/flickr_cat_000002.png"), data/ for a combined one like
      stargan, so its paths start with the member (e.g. "afhq_v2/train/cat/flickr_cat_000002.png").
      The list is append-only: new images get new ids at the end, existing ids never change.
  runs/latent_analysis/<dataset>/labels.json
      {"<id>": [tag, ...]}, written by the explorer.
  runs/latent_analysis/<dataset>/rejections.json
      {"<id>": [tag, ...]}: tags a human confirmed the image does NOT have, also written by
      the explorer. Untagged is not the same as rejected; autotag.py only trains on
      rejections (and --exclusive-group) as negatives. A tag is never in both files for one
      image: accepting it drops the rejection and rejecting it drops the label.
  runs/latent_analysis/<dataset>/migrated.json   (combined datasets only)
      {"<member>/labels.json": time, ...}: member tag files already copied in (see
      `migrate_member_tags`), so each is copied once.

A combined dataset has its own ids and tags; the members' files are left as they are.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from embed import DATA_ROOT, REPO_ROOT
from kmeans import RUNS_ROOT
from zoo.data import stargan

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def index_file(dataset: str, runs_root: Path = RUNS_ROOT) -> Path:
    return runs_root / dataset / "index.json"


def labels_file(dataset: str, runs_root: Path = RUNS_ROOT) -> Path:
    return runs_root / dataset / "labels.json"


def rejections_file(dataset: str, runs_root: Path = RUNS_ROOT) -> Path:
    return runs_root / dataset / "rejections.json"


def read_tags(path: Path) -> dict[str, list[str]]:
    """A labels/rejections file as {"<id>": [tag, ...]}; empty if it doesn't exist yet."""
    return json.loads(path.read_text()) if path.exists() else {}


def image_root(dataset: str, data_root: Path = DATA_ROOT) -> Path:
    """What index.json paths are relative to: data/<dataset>, or data/ for a combined dataset."""
    return (data_root if dataset in stargan.COMBINED else data_root / dataset).resolve()


def image_ids(paths: list[str], index: dict[str, int], root: Path) -> list[int]:
    """Ids for image paths as stored in embedding files (repo-relative or absolute)."""
    return [index[(Path(p) if Path(p).is_absolute() else REPO_ROOT / p).resolve().relative_to(root).as_posix()]
            for p in paths]


def _write_json(path: Path, obj: object, **kw) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, **kw))
    tmp.rename(path)


def update_index(dataset: str, data_root: Path = DATA_ROOT, runs_root: Path = RUNS_ROOT) -> dict[str, int]:
    """Add any image of the dataset (every member's, if combined) that has no id yet;
    return {path relative to image_root: id}.

    New images are appended in sorted path order, so a fresh index is deterministic."""
    f = index_file(dataset, runs_root)
    root = image_root(dataset, data_root)
    paths: list[str] = json.loads(f.read_text())["paths"] if f.exists() else []
    ids = {p: i for i, p in enumerate(paths)}
    found = sorted(
        p.relative_to(root).as_posix()
        for m in stargan.members(dataset)
        for p in (data_root / m).resolve().rglob("*")
        if p.suffix.lower() in IMAGE_EXTS
    )
    new = [p for p in found if p not in ids]
    if new:
        for p in new:
            ids[p] = len(paths)
            paths.append(p)
        _write_json(f, {"dataset": dataset, "paths": paths}, indent=0)
        print(f"{f}: +{len(new)} images ({len(paths)} total)")
    return ids


def sync_index(dataset: str, data_root: Path = DATA_ROOT, runs_root: Path = RUNS_ROOT) -> dict[str, int]:
    """update_index, then the one-off tag migrations below. Returns the index."""
    index = update_index(dataset, data_root, runs_root)
    migrate_path_labels(dataset, image_root(dataset, data_root), index, runs_root)
    migrate_member_tags(dataset, index, runs_root)
    return index


def migrate_path_labels(dataset: str, ds_root: Path, ids: dict[str, int], runs_root: Path = RUNS_ROOT) -> None:
    """Fold old per-model labels files (runs/.../<dataset>/<model>/labels.json, keyed by
    repo-relative image path) into the dataset-level id-keyed file, then rename them."""
    olds = sorted((runs_root / dataset).glob("*/labels.json"))
    if not olds:
        return
    dst = labels_file(dataset, runs_root)
    tags: dict[str, list[str]] = json.loads(dst.read_text()) if dst.exists() else {}
    for old in olds:
        unknown = 0
        for path, ts in json.loads(old.read_text()).items():
            p = Path(path) if Path(path).is_absolute() else REPO_ROOT / path
            try:
                key = str(ids[p.resolve().relative_to(ds_root.resolve()).as_posix()])
            except (KeyError, ValueError):
                unknown += 1
                continue
            tags[key] = list(dict.fromkeys(tags.get(key, []) + ts))
        _write_json(dst, tags, indent=1)
        old.rename(old.with_name("labels.path-keyed.json.bak"))
        print(f"migrated {old} -> {dst}" + (f" ({unknown} paths not in the index, dropped)" if unknown else ""))


def migrate_member_tags(dataset: str, ids: dict[str, int], runs_root: Path = RUNS_ROOT) -> None:
    """For a combined dataset, copy each member's labels.json and rejections.json into the
    combined files, mapping member ids to combined ids through the image paths.

    Each member file is copied once (recorded in migrated.json), so later edits in the
    combined explorer aren't undone by re-running this. The member files are left in place;
    one that appears later (e.g. after tagging CelebA-HQ on its own) is copied on the next run."""
    if dataset not in stargan.COMBINED:
        return
    marker = runs_root / dataset / "migrated.json"
    done: dict[str, str] = json.loads(marker.read_text()) if marker.exists() else {}
    dsts = {"labels.json": labels_file(dataset, runs_root), "rejections.json": rejections_file(dataset, runs_root)}
    merged = {name: read_tags(dst) for name, dst in dsts.items()}
    copied = []
    for m in stargan.members(dataset):
        m_index = index_file(m, runs_root)
        for name in dsts:
            src = runs_root / m / name
            if f"{m}/{name}" in done or not src.exists():
                continue
            if not m_index.exists():
                print(f"can't migrate {src}: no {m_index} to map its ids (run export.py --dataset {m})")
                continue
            m_paths = json.loads(m_index.read_text())["paths"]
            unknown = 0
            for key, ts in read_tags(src).items():
                new = ids.get(f"{m}/{m_paths[int(key)]}") if int(key) < len(m_paths) else None
                if new is None:
                    unknown += 1
                    continue
                cur = merged[name].setdefault(str(new), [])
                cur += [t for t in ts if t not in cur]
            copied.append(f"{m}/{name}")
            print(f"migrated {src} -> {dsts[name]}" + (f" ({unknown} ids not found, dropped)" if unknown else ""))
    if not copied:
        return
    # Keep the invariant that a tag is never both accepted and rejected for one image.
    for key, ts in list(merged["rejections.json"].items()):
        rest = [t for t in ts if t not in merged["labels.json"].get(key, [])]
        if rest:
            merged["rejections.json"][key] = rest
        else:
            del merged["rejections.json"][key]
    for name, dst in dsts.items():
        if merged[name] or dst.exists():
            _write_json(dst, merged[name], indent=1)
    now = datetime.now().isoformat(timespec="seconds")
    _write_json(marker, done | {c: now for c in copied}, indent=1)
