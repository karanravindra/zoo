"""Suggest tags for every image with linear probes on DINOv3 embeddings, trained on the
tags people have accepted and rejected in the explorer.

Run from anywhere (after embed.py and export.py, with the same --dataset/--model/--image-size),
then again after each review round in the explorer:
    uv run projects/latent_analysis/autotag.py --dataset afhq_v2
    uv run projects/latent_analysis/autotag.py --dataset afhq_v2 --exclusive-group cat,dog,wild
    uv run projects/latent_analysis/autotag.py --dataset stargan   # AFHQv2 + CelebA-HQ, with their shared tags

For each tag with at least --min-pos positives (images tagged with it in labels.json) and
--min-neg negatives (images where it was rejected, in rejections.json), fits a logistic
regression on the L2-normalized embeddings of every split. Untagged images are never used as
negatives. Under --exclusive-group a,b,c (repeatable), an image tagged with one tag of the
group also counts as a negative for the others. Each tag's threshold is the lowest score whose
cross-validated precision reaches --target-precision (null if none does).

Writes to <repo>/runs/latent_analysis/<dataset>/<model name>_<size>/autotag/:
  predictions.json  {"version", "command", "tags": {tag: {"threshold", "metrics", "n_pos",
                    "n_neg"}}, "skipped": {tag: {"n_pos", "n_neg"}}, "scores": {"<id>": {tag:
                    score}}}, scores rounded to 3 decimals and only kept when >= --min-score;
                    read by the explorer server on every page load for its review mode
  probes.pt         {"tags", "weight" (D, T), "bias" (T,), "thresholds", "version"}; scores are
                    sigmoid(normalize(features) @ weight + bias)
Never writes labels.json or rejections.json.
"""

from __future__ import annotations

import json
import shlex
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
import tyro
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from tyro.conf import UseAppendAction

from embed import DEFAULT_MODEL, latent_dir, load_embeddings
from image_index import image_ids, image_root, labels_file, read_tags, rejections_file, sync_index
from kmeans import RUNS_ROOT

from zoo.data import stargan
from zoo.paths import DATA_ROOT


def pick_threshold(y: np.ndarray, p: np.ndarray, target: float) -> tuple[float | None, float, float]:
    """Lowest threshold t with precision(p >= t) >= target, as (t, precision, recall).

    The lowest one keeps the most recall. With no such threshold, returns (None, precision,
    recall) at 0.5 so there is still something to report."""
    prec, rec, thr = precision_recall_curve(y, p)  # thr ascending; prec[i], rec[i] are for p >= thr[i]
    ok = np.flatnonzero(prec[:-1] >= target)
    if len(ok):
        i = ok[0]
        return float(thr[i]), float(prec[i]), float(rec[i])
    pred = p >= 0.5
    tp = int((pred & (y == 1)).sum())
    return None, tp / max(int(pred.sum()), 1), tp / max(int(y.sum()), 1)


def make_probe(C: float) -> LogisticRegression:
    return LogisticRegression(C=C, class_weight="balanced", max_iter=2000)


@dataclass
class Args:
    dataset: Literal[tuple(stargan.ALL)] = "afhq_v2"
    model: str = DEFAULT_MODEL  # Hugging Face model id
    image_size: int = 256
    min_pos: int = 5  # positives a tag needs before it gets a probe
    min_neg: int = 5  # negatives (rejections) a tag needs
    # tags no image has two of, as "A,B,..."; a positive for one is a negative for the others (repeatable)
    exclusive_group: UseAppendAction[list[str]] = field(default_factory=list)
    target_precision: float = 0.95
    folds: int = 5  # CV folds, fewer when a class has fewer examples
    C: float = 1.0  # inverse L2 strength of the logistic regression
    min_score: float = 0.05  # scores below this aren't written
    seed: int = 0
    data_root: Path = DATA_ROOT
    runs_root: Path = RUNS_ROOT


def main() -> None:
    args = tyro.cli(Args, description=__doc__)
    if args.min_pos < 2 or args.min_neg < 2:
        sys.exit("--min-pos and --min-neg must be at least 2 for cross-validation")

    # Embeddings for every split, each row mapped to its stable image id (as export.py does).
    src = latent_dir(args.dataset, args.model, args.image_size, args.data_root)
    split_names = stargan.splits(args.dataset)
    index = sync_index(args.dataset, args.data_root, args.runs_root)
    root = image_root(args.dataset, args.data_root)
    feats, ids = [], []
    for split in split_names:
        e = load_embeddings(args.dataset, args.model, args.image_size, split, args.data_root)
        feats.append(e["features"])
        ids += image_ids(e["paths"], index, root)
    x = F.normalize(torch.cat(feats).float(), dim=1)
    row = {i: r for r, i in enumerate(ids)}
    print(f"{len(ids)} images from {split_names}, {x.shape[1]}-d embeddings from {src.name}")

    # Human tags, restricted to images we have embeddings for, as {tag: set of rows}.
    def by_tag(path: Path) -> dict[str, set[int]]:
        out: dict[str, set[int]] = {}
        for key, ts in read_tags(path).items():
            if int(key) in row:
                for t in ts:
                    out.setdefault(t, set()).add(row[int(key)])
        return out

    pos = by_tag(labels_file(args.dataset, args.runs_root))
    rej = by_tag(rejections_file(args.dataset, args.runs_root))
    groups = [[t.strip() for t in g.split(",") if t.strip()] for g in args.exclusive_group]

    xn = x.numpy()
    fitted: dict[str, dict] = {}
    skipped: dict[str, dict] = {}
    weights, biases = [], []
    print(f"{'tag':<24} {'pos':>5} {'neg':>5} {'folds':>5} {'AP':>6} {'thresh':>7} {'prec':>6} {'rec':>6}")
    for tag in sorted(set(pos) | set(rej)):
        p_rows = pos.get(tag, set())
        n_rows = set(rej.get(tag, set()))
        for g in groups:
            if tag in g:
                for other in g:
                    if other != tag:
                        n_rows |= pos.get(other, set())
        # An image that is both (tagged twice in an exclusive group, or a stale file) is ambiguous: drop it.
        both = p_rows & n_rows
        if both:
            print(f"  {tag}: {len(both)} images are both positive and negative; leaving them out")
        p_rows, n_rows = sorted(p_rows - both), sorted(n_rows - both)
        if len(p_rows) < args.min_pos or len(n_rows) < args.min_neg:
            skipped[tag] = {"n_pos": len(p_rows), "n_neg": len(n_rows)}
            continue

        rows = np.array(p_rows + n_rows)
        y = np.r_[np.ones(len(p_rows)), np.zeros(len(n_rows))].astype(int)
        k = min(args.folds, len(p_rows), len(n_rows))
        cv = StratifiedKFold(k, shuffle=True, random_state=args.seed)
        p_cv = cross_val_predict(make_probe(args.C), xn[rows], y, cv=cv, method="predict_proba")[:, 1]
        thr, prec, rec = pick_threshold(y, p_cv, args.target_precision)
        ap = float(average_precision_score(y, p_cv))

        probe = make_probe(args.C).fit(xn[rows], y)
        weights.append(torch.from_numpy(probe.coef_[0]).float())
        biases.append(float(probe.intercept_[0]))
        # Floored so it compares correctly against the 3-decimal scores in predictions.json.
        thr = None if thr is None else float(np.floor(thr * 1000) / 1000)
        fitted[tag] = {
            "threshold": thr,
            # CV precision/recall at the threshold, or at 0.5 when there is none ("at")
            "metrics": {"ap": round(ap, 4), "precision": round(prec, 4), "recall": round(rec, 4),
                        "at": 0.5 if thr is None else thr, "folds": k,
                        "target_precision": args.target_precision},
            "n_pos": len(p_rows),
            "n_neg": len(n_rows),
        }
        shown = "none" if thr is None else f"{thr:.3f}"
        print(f"{tag:<24} {len(p_rows):>5} {len(n_rows):>5} {k:>5} {ap:>6.3f} {shown:>7} {prec:>6.3f} {rec:>6.3f}")
        if thr is None:
            print(f"  {tag}: no threshold reaches precision {args.target_precision}; "
                  f"threshold is null (precision/recall shown at 0.5). Review its uncertain images.")

    for tag, c in skipped.items():
        print(f"skipped {tag}: {c['n_pos']} positives, {c['n_neg']} negatives "
              f"(needs {args.min_pos} and {args.min_neg}; reject some images in the explorer)")

    tags = list(fitted)
    version = datetime.now().isoformat(timespec="seconds")
    w = torch.stack(weights, 1) if tags else torch.zeros(x.shape[1], 0)
    b = torch.tensor(biases, dtype=torch.float32)
    scores = torch.sigmoid(x @ w + b)  # (N, T), every image against every probe in one matmul

    out_scores: dict[str, dict[str, float]] = {}
    keep = scores >= args.min_score
    for r, c in keep.nonzero().tolist():
        out_scores.setdefault(str(ids[r]), {})[tags[c]] = round(float(scores[r, c]), 3)
    for j, tag in enumerate(tags):
        thr = fitted[tag]["threshold"]
        n_above = int((scores[:, j] >= thr).sum()) if thr is not None else 0
        print(f"{tag}: {n_above} images at or above the threshold ({len(pos[tag])} already tagged)")

    out_dir = args.runs_root / args.dataset / src.name / "autotag"
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions = {
        "version": version,
        "dataset": args.dataset,
        "model": src.name,
        "command": shlex.join(["uv", "run", "projects/latent_analysis/autotag.py", *sys.argv[1:]]),
        "tags": fitted,
        "skipped": skipped,
        "scores": out_scores,
    }
    tmp = out_dir / ".predictions.json.tmp"
    tmp.write_text(json.dumps(predictions, separators=(",", ":")))
    tmp.rename(out_dir / "predictions.json")
    tmp = out_dir / ".probes.pt.tmp"
    torch.save({"tags": tags, "weight": w, "bias": b, "version": version,
                "thresholds": [fitted[t]["threshold"] for t in tags],
                "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}, tmp)
    tmp.rename(out_dir / "probes.pt")
    print(f"wrote {out_dir / 'predictions.json'} ({len(tags)} tags, {len(out_scores)} images with a score)")
    print("review the suggestions in the explorer (Review tab), then re-run this command:\n  " + predictions["command"])


if __name__ == "__main__":
    main()
