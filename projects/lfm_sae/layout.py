"""Maps of a train_sae.py run's SAEs, for the feature viewer: a 2-D UMAP of each SAE's decoder directions, and each
latent's nearest decoder directions in the other SAEs of the run (how features split as the dictionary grows).

Writes to each <run>/sae_<n>/features/:
  umap.json        {"x": [...], "y": [...]} per latent (cosine UMAP of the unit-norm decoder rows, after PCA to 50)
  neighbors.json   {"<other n>": {"ids": [[...]], "cos": [[...]]}}: the `--neighbors` most similar decoder rows in
                   each other SAE, per latent

    uv run projects/lfm_sae/layout.py --run runs/lfm_sae/train/<timestamp>
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro


@dataclass
class Config:
    run: Path  # train_sae.py run directory (holds sae_<n>/)
    checkpoint: str = "best.pt"  # checkpoint name in each sae_<n>/
    neighbors: int = 5  # nearest decoder directions kept per latent and other SAE
    umap_neighbors: int = 15  # UMAP n_neighbors
    seed: int = 0  # UMAP seed
    device: str = "cuda"  # torch device for the similarity search


def decoders(cfg: Config) -> dict[int, torch.Tensor]:
    out = {}
    for d in sorted(Path(cfg.run).glob("sae_*")):
        w = torch.load(d / cfg.checkpoint, map_location="cpu", weights_only=False)["sae"]["W_dec"]
        out[w.shape[0]] = w / w.norm(dim=-1, keepdim=True)
    return dict(sorted(out.items()))


@torch.no_grad()
def nearest(a: torch.Tensor, b: torch.Tensor, k: int, device, chunk: int = 4096) -> tuple[list, list]:
    b = b.to(device)
    ids, cos = [], []
    for i in range(0, len(a), chunk):
        sim = a[i : i + chunk].to(device) @ b.T
        v, j = sim.topk(k, dim=-1)
        ids.append(j.cpu())
        cos.append(v.cpu())
    return torch.cat(ids).tolist(), torch.cat(cos).mul(1000).round().div(1000).tolist()


def main(cfg: Config):
    import umap
    from sklearn.decomposition import PCA

    decs = decoders(cfg)
    if not decs:
        raise SystemExit(f"no sae_*/{cfg.checkpoint} under {cfg.run}")
    for n, w in decs.items():
        out = Path(cfg.run) / f"sae_{n}" / "features"
        out.mkdir(parents=True, exist_ok=True)
        t = time.time()
        neigh = {str(m): dict(zip(("ids", "cos"), nearest(w, other, cfg.neighbors, cfg.device)))
                 for m, other in decs.items() if m != n}
        (out / "neighbors.json").write_text(json.dumps(neigh))
        z = PCA(50, random_state=cfg.seed).fit_transform(w.numpy())
        xy = umap.UMAP(n_neighbors=cfg.umap_neighbors, metric="cosine", random_state=cfg.seed).fit_transform(z)
        (out / "umap.json").write_text(json.dumps({"x": [round(float(v), 3) for v in xy[:, 0]],
                                                    "y": [round(float(v), 3) for v in xy[:, 1]]}))
        print(f"sae_{n}: neighbors in {', '.join(neigh)} and UMAP in {time.time() - t:.0f}s -> {out}")


if __name__ == "__main__":
    main(tyro.cli(Config))
