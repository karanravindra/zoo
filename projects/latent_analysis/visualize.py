"""2-D projections (PCA, t-SNE at several perplexities, UMAP) of embeddings from `embed.py`.

Run from anywhere (after embed.py, with the same --dataset/--model/--image-size):
    uv run projects/latent_analysis/visualize.py --dataset afhq_v2
    uv run projects/latent_analysis/visualize.py --dataset afhq_v2 --k 8   # also color by k-means cluster
    uv run projects/latent_analysis/visualize.py --dataset stargan         # AFHQv2 + CelebA-HQ together

Features are L2-normalized (matching the cosine k-means), then reduced to `--pca-dims`
with PCA before t-SNE / UMAP. Outputs go to
<repo>/runs/latent_analysis/<dataset>/<model name>_<size>/viz/<split>/:
  coords.pt                    {method: (N, 2) coordinates}; reused on later runs
  pca_variance.png             cumulative explained variance of the PCA
  <method>_classes.png         scatter colored by dataset class (cat/dog/wild, female/male;
                               afhq_v2/cat ... celeba_hq/male for stargan)
  <method>_clusters_k<k>.png   one panel per k-means cluster, that cluster highlighted (needs --k)
where <method> is pca, tsne_p<perplexity>, umap.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed import DATA_ROOT, DEFAULT_MODEL, latent_dir, load_embeddings  # noqa: E402
from kmeans import RUNS_ROOT  # noqa: E402

from zoo.data import stargan  # noqa: E402

# Categorical slots that stay distinguishable (incl. colorblind) for all pairs in a scatter.
# Datasets here have <= 5 classes (3 + 2 for stargan); clusters use small multiples instead of k hues.
CLASS_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6")
HIGHLIGHT, BACKGROUND = "#2a78d6", "#d3d1c7"
TEXT, MUTED = "#2b2b29", "#6f6e69"


def compute_coords(x: np.ndarray, methods: list[str], pca_dims: int, seed: int, cached: dict) -> dict:
    coords = dict(cached)
    pca = PCA(n_components=min(pca_dims, *x.shape), random_state=seed).fit(x)
    reduced = pca.transform(x)
    coords["pca"] = reduced[:, :2]
    coords["_pca_explained"] = pca.explained_variance_ratio_

    for m in methods:
        if m in coords:
            print(f"{m}: cached")
            continue
        print(f"{m}: fitting on {reduced.shape}")
        if m.startswith("tsne_p"):
            perplexity = float(m.removeprefix("tsne_p"))
            coords[m] = TSNE(perplexity=perplexity, init="pca", random_state=seed, n_jobs=-1).fit_transform(reduced)
        elif m == "umap":
            import umap  # slow import (numba), so only when needed

            coords[m] = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=seed).fit_transform(reduced)
    return coords


def style(ax: plt.Axes, title: str | None = None) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, color=TEXT, fontsize=10, loc="left")


def point_size(n: int) -> float:
    return float(np.clip(20_000 / n, 0.5, 6))


def plot_classes(xy: np.ndarray, labels: np.ndarray, classes: list[str], title: str, path: Path) -> None:
    if len(classes) > len(CLASS_COLORS):
        raise ValueError(f"{len(classes)} classes; only {len(CLASS_COLORS)} scatter colors are defined")
    fig, ax = plt.subplots(figsize=(7, 7))
    order = np.random.default_rng(0).permutation(len(xy))  # no class painted entirely on top
    colors = np.array(CLASS_COLORS)[labels[order]]
    ax.scatter(*xy[order].T, c=colors, s=point_size(len(xy)), linewidths=0, alpha=0.7, rasterized=True)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, markersize=8, label=f"{name} ({(labels == i).sum()})")
               for i, (name, c) in enumerate(zip(classes, CLASS_COLORS))]
    ax.legend(handles=handles, frameon=False, labelcolor=TEXT, loc="best")
    style(ax, title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_clusters(xy: np.ndarray, assign: np.ndarray, k: int, title: str, path: Path) -> None:
    cols = math.ceil(math.sqrt(k))
    rows = math.ceil(k / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.6 + 0.4), squeeze=False)
    s = point_size(len(xy))
    for i, ax in enumerate(axes.flat):
        style(ax)
        if i >= k:
            ax.set_visible(False)
            continue
        m = assign == i
        ax.scatter(*xy[~m].T, c=BACKGROUND, s=s, linewidths=0, rasterized=True)
        ax.scatter(*xy[m].T, c=HIGHLIGHT, s=s, linewidths=0, rasterized=True)
        ax.set_title(f"cluster {i}  (n={m.sum()})", color=MUTED, fontsize=8, loc="left")
    fig.suptitle(title, color=TEXT, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_variance(explained: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(np.arange(1, len(explained) + 1), np.cumsum(explained), color=HIGHLIGHT, lw=2)
    ax.set(xlabel="principal components", ylabel="cumulative explained variance", ylim=(0, 1))
    ax.set_title("PCA explained variance", color=TEXT, fontsize=10, loc="left")
    ax.grid(axis="y", color="#e8e6df", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@dataclass
class Args:
    dataset: Literal[tuple(stargan.ALL)] = "afhq_v2"
    model: str = DEFAULT_MODEL  # Hugging Face model id
    image_size: int = 256
    split: str | None = None  # defaults to the train split
    perplexities: list[float] = field(default_factory=lambda: [5, 30, 100])
    no_umap: bool = False
    pca_dims: int = 50  # PCA dims fed to t-SNE / UMAP
    k: int | None = None  # also color by this k-means run (from kmeans.py)
    seed: int = 0
    data_root: Path = DATA_ROOT
    runs_root: Path = RUNS_ROOT


def main() -> None:
    args = tyro.cli(Args, description=__doc__)

    split = args.split or stargan.splits(args.dataset)[0]
    src = latent_dir(args.dataset, args.model, args.image_size, args.data_root)
    e = load_embeddings(args.dataset, args.model, args.image_size, split, args.data_root)
    x = F.normalize(e["features"], dim=1).numpy()
    labels = e["labels"].numpy()

    assign = None
    if args.k is not None:
        km_path = args.runs_root / args.dataset / src.name / f"k{args.k}" / "assignments.pt"
        if not km_path.exists():
            sys.exit(f"No k-means results at {km_path}; run kmeans.py --k {args.k} first.")
        km = torch.load(km_path)[split]
        if km["paths"] != e["paths"]:
            sys.exit(f"{km_path} was made from different embeddings; re-run kmeans.py.")
        assign = km["assign"].numpy()

    out = args.runs_root / args.dataset / src.name / "viz" / split
    out.mkdir(parents=True, exist_ok=True)
    coords_path = out / "coords.pt"
    cached = {}
    if coords_path.exists():
        saved = torch.load(coords_path, weights_only=False)
        # Only reuse coordinates computed with the same settings.
        if saved.get("_settings") == {"pca_dims": args.pca_dims, "seed": args.seed, "n": len(x)}:
            cached = {m: v for m, v in saved.items() if not m.startswith("_")}

    methods = [f"tsne_p{p:g}" for p in args.perplexities] + ([] if args.no_umap else ["umap"])
    coords = compute_coords(x, methods, args.pca_dims, args.seed, cached)
    coords["_settings"] = {"pca_dims": args.pca_dims, "seed": args.seed, "n": len(x)}
    torch.save(coords, coords_path)

    plot_variance(coords["_pca_explained"], out / "pca_variance.png")
    titles = {"pca": "PCA"} | {m: f"t-SNE (perplexity {m.removeprefix('tsne_p')})" for m in methods if m != "umap"}
    titles |= {"umap": "UMAP"} if "umap" in methods else {}
    for m, name in titles.items():
        title = f"{name} · {args.dataset}/{split} · {src.name}"
        plot_classes(coords[m], labels, e["classes"], title, out / f"{m}_classes.png")
        if assign is not None:
            plot_clusters(coords[m], assign, args.k, f"{title} · k-means k={args.k}", out / f"{m}_clusters_k{args.k}.png")
    print(f"Outputs in {out}")


if __name__ == "__main__":
    main()
