"""Run k-means on DINOv3 embeddings written by `embed.py`.

Run from anywhere (after embed.py, with the same --dataset/--model/--image-size):
    uv run projects/latent_analysis/kmeans.py --dataset afhq_v2 --k 8
    uv run projects/latent_analysis/kmeans.py --dataset stargan --k 16   # AFHQv2 + CelebA-HQ together

Clusters the unit-normalized train embeddings with e-means (emeans.py, the default) or
full-batch spherical k-means (--method spherical), normalizes the centroids, then assigns
every split to the most cosine-similar centroid. Outputs go to <repo>/runs/latent_analysis/<dataset>/<model name>_<size>/k<k>/:
  centroids.pt     (k, D) unit-norm centroids
  assignments.pt   {split: {"assign": cluster id per image, "paths": image paths}}
  summary.json     inertia, cluster sizes, cluster x class counts and purity per split
  clusters_<split>.json   {"k", "pathsKey", "cluster", "dist" (1 - cos to its centroid)},
                   picked up by the explorer server on every page load
  clusters.png     the images nearest to each centroid, one row per cluster
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import tyro
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed import DATA_ROOT, DEFAULT_MODEL, REPO_ROOT, latent_dir, load_embeddings  # noqa: E402
from emeans import EMeans  # noqa: E402

from zoo.data import stargan  # noqa: E402
from zoo.utils.device import seed_everything  # noqa: E402

RUNS_ROOT = REPO_ROOT / "runs" / "latent_analysis"


def kmeans_pp_init(x: torch.Tensor, k: int, gen: torch.Generator) -> torch.Tensor:
    """k-means++ seeding on rows of `x`."""
    n = x.shape[0]
    centroids = [x[torch.randint(n, (1,), generator=gen, device=x.device)]]
    d2 = torch.cdist(x, centroids[0]).squeeze(1).square()
    for _ in range(1, k):
        idx = torch.multinomial(d2 / d2.sum(), 1, generator=gen)
        centroids.append(x[idx])
        d2 = torch.minimum(d2, torch.cdist(x, centroids[-1]).squeeze(1).square())
    return torch.cat(centroids)


def spherical_kmeans(
    x: torch.Tensor, k: int, n_init: int = 5, max_iter: int = 300, tol: float = 1e-6, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """K-means on unit-normalized rows (i.e. cosine similarity), best of `n_init` restarts.

    Returns (centroids (k, D), assignments (N,), inertia = sum of (1 - cos) to assigned centroid).
    """
    x = F.normalize(x, dim=1)
    gen = torch.Generator(device=x.device).manual_seed(seed)
    best = (None, None, float("inf"))
    for _ in range(n_init):
        c = F.normalize(kmeans_pp_init(x, k, gen), dim=1)
        prev = float("inf")
        for _ in range(max_iter):
            sim = x @ c.T
            assign = sim.argmax(1)
            inertia = (1 - sim.gather(1, assign[:, None])).sum().item()
            sums = torch.zeros_like(c).index_add_(0, assign, x)
            empty = torch.bincount(assign, minlength=k) == 0
            if empty.any():  # re-seed empty clusters with the worst-fit points
                worst = sim.max(1).values.argsort()[: int(empty.sum())]
                sums[empty] = x[worst]
            c = F.normalize(sums, dim=1)
            if prev - inertia <= tol * max(prev, 1.0):
                break
            prev = inertia
        if inertia < best[2]:
            best = (c, assign, inertia)
    return best


def emeans_cosine(
    x: torch.Tensor, k: int, nepochs: int = 5, n_init: int = 5, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """e-means on unit-normalized rows, best of `n_init` seeds by cosine inertia.

    e-means minimizes squared Euclidean distance; on unit vectors that is 2 - 2cos, so it
    clusters by cosine, but its centroids are means with norm < 1. They are normalized
    afterwards and points reassigned by cosine, matching spherical_kmeans' outputs.
    Returns (centroids (k, D), assignments (N,), inertia = sum of (1 - cos)).
    """
    x = F.normalize(x, dim=1)
    best = (None, None, float("inf"))
    for i in range(n_init):
        em = EMeans(k, nepochs=nepochs, seed=seed + i, device=x.device)
        c, _ = em.fit(x)
        c = F.normalize(c, dim=1)
        sim = (x @ c.T).max(1)
        inertia = (1 - sim.values).sum().item()
        if inertia < best[2]:
            best = (c, sim.indices, inertia)
    print(f"e-means: B={em.batch_size_} alpha0={em.alpha0_:.3f} epochs={nepochs} x {n_init} seeds")
    return best


def paths_key(paths: list[str]) -> str:
    """Fingerprint of an embeddings file's image order, so the explorer only combines
    cluster files with points that are in the same order."""
    return hashlib.sha1("\n".join(paths).encode()).hexdigest()[:16]


def write_cluster_json(run_dir: Path, split: str, k: int, paths: list[str], features: torch.Tensor,
                       centroids: torch.Tensor) -> torch.Tensor:
    """Assign rows to the most cosine-similar centroid; save cluster + distance for the
    explorer to run_dir/clusters_<split>.json. Returns the assignments."""
    sim = (F.normalize(features, dim=1) @ centroids.T).max(1)
    out = {"k": k, "pathsKey": paths_key(paths), "cluster": sim.indices.tolist(),
           "dist": [round(1 - s, 5) for s in sim.values.tolist()]}
    tmp = run_dir / f".clusters_{split}.json.tmp"
    tmp.write_text(json.dumps(out, separators=(",", ":")))
    tmp.rename(run_dir / f"clusters_{split}.json")
    return sim.indices


def cluster_table(assign: torch.Tensor, labels: torch.Tensor, k: int, classes: list[str]) -> dict:
    counts = torch.zeros(k, len(classes), dtype=torch.long)
    counts.index_put_((assign, labels), torch.ones_like(assign), accumulate=True)
    return {
        "sizes": counts.sum(1).tolist(),
        "class_counts": {f"cluster_{i}": dict(zip(classes, row.tolist())) for i, row in enumerate(counts)},
        "purity": counts.max(1).values.sum().item() / len(assign),
    }


def plot_clusters(x: torch.Tensor, centroids: torch.Tensor, paths: list[str], path: Path, per_row: int = 10) -> None:
    """One row per cluster: the `per_row` images closest to that centroid."""
    sim = F.normalize(x, dim=1) @ centroids.T
    assign = sim.argmax(1)
    k = len(centroids)
    fig, axes = plt.subplots(k, per_row, figsize=(per_row * 1.2, k * 1.2), squeeze=False)
    for i in range(k):
        members = (assign == i).nonzero().squeeze(1)
        top = members[sim[members, i].argsort(descending=True)[:per_row]].tolist()
        for j, ax in enumerate(axes[i]):
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if j < len(top):
                p = Path(paths[top[j]])
                with Image.open(p if p.is_absolute() else REPO_ROOT / p) as img:
                    ax.imshow(img.convert("RGB").resize((96, 96)))
        axes[i, 0].set_ylabel(f"{i}\n(n={len(members)})", rotation=0, ha="right", va="center", fontsize=8)
    fig.tight_layout(pad=0.2)
    fig.savefig(path, dpi=100)
    plt.close(fig)


@dataclass
class Args:
    dataset: Literal[tuple(stargan.ALL)] = "afhq_v2"
    model: str = DEFAULT_MODEL  # Hugging Face model id
    image_size: int = 256
    k: int = 32
    method: Literal["emeans", "spherical"] = "emeans"
    n_init: int = 5  # restarts; the lowest-inertia one is kept
    epochs: int = 5  # e-means passes over the data
    seed: int = 0
    data_root: Path = DATA_ROOT
    runs_root: Path = RUNS_ROOT


def main() -> None:
    args = tyro.cli(Args, description=__doc__)

    seed_everything(args.seed)
    src = latent_dir(args.dataset, args.model, args.image_size, args.data_root)
    split_names = stargan.splits(args.dataset)
    embeddings = {s: load_embeddings(args.dataset, args.model, args.image_size, s, args.data_root) for s in split_names}

    fit_split = split_names[0]
    fit = embeddings[fit_split]
    # Tens of thousands of points x ~768 dims is quick on CPU; use CUDA if present.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = fit["features"].to(device)
    if args.method == "emeans":
        centroids, _, inertia = emeans_cosine(x, args.k, args.epochs, n_init=args.n_init, seed=args.seed)
    else:
        centroids, _, inertia = spherical_kmeans(x, args.k, n_init=args.n_init, seed=args.seed)
    centroids = centroids.cpu()
    print(f"{args.method} on {fit_split} ({tuple(fit['features'].shape)}): inertia={inertia:.2f}")

    run_dir = args.runs_root / args.dataset / src.name / f"k{args.k}"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "fit_split": fit_split,
        "inertia": inertia,
        "splits": {},
    }
    assignments = {}
    for split, e in embeddings.items():
        assign = write_cluster_json(run_dir, split, args.k, e["paths"], e["features"], centroids)
        assignments[split] = {"assign": assign, "paths": e["paths"]}
        table = summary["splits"][split] = cluster_table(assign, e["labels"], args.k, e["classes"])
        print(f"{split}: sizes={table['sizes']} class purity={table['purity']:.3f}")

    torch.save(centroids, run_dir / "centroids.pt")
    torch.save(assignments, run_dir / "assignments.pt")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_clusters(fit["features"], centroids, fit["paths"], run_dir / "clusters.png")
    print(f"Outputs in {run_dir}")


if __name__ == "__main__":
    main()
