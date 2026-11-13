"""Embed a StarGAN v2 dataset (AFHQv2 / CelebA-HQ) with a DINOv3 model from Hugging Face.

Run from anywhere:
    uv run projects/latent_analysis/embed.py --dataset afhq_v2
    uv run projects/latent_analysis/embed.py --dataset celeba_hq --model facebook/dinov3-vitb16-pretrain-lvd1689m
    uv run projects/latent_analysis/embed.py --dataset stargan   # every member of a combined dataset

The DINOv3 checkpoints are gated: accept the license on the model page and log in
(`hf auth login`) before the first run.

Downloads the dataset if needed (`zoo.data.stargan`), then writes one file per split:
  <repo>/data/latents/<dataset>/<model name>_<size>/<split>.pt
      {"features": (N, D) float32 pooled embeddings (not normalized),
       "labels": (N,) class index, "classes": class names, "paths": image paths}
Splits that already have a file are skipped. A combined dataset (stargan.COMBINED) has no
files of its own: each member is embedded into its own directory, and `load_embeddings`
concatenates them for the other scripts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from zoo.data import stargan
from zoo.utils.device import get_device

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
DEFAULT_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"


def latent_dir(dataset: str, model: str, image_size: int, data_root: Path = DATA_ROOT) -> Path:
    """e.g. data/latents/afhq_v2/dinov3-convnext-tiny-pretrain-lvd1689m_256"""
    return data_root / "latents" / dataset / f"{model.split('/')[-1]}_{image_size}"


def load_embeddings(dataset: str, model: str, image_size: int, split: str, data_root: Path = DATA_ROOT) -> dict:
    """One split's embeddings in embed_split's format. For a combined dataset, the members'
    files for that split role concatenated in member order, with class names prefixed by the
    member ("afhq_v2/cat") and labels renumbered to match. Exits if a file is missing."""
    names = stargan.members(dataset)
    files = [latent_dir(m, model, image_size, data_root) / f"{stargan.member_split(dataset, m, split)}.pt" for m in names]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        sys.exit(f"No embeddings at {', '.join(missing)}; run embed.py --dataset {dataset} with the same --model/--image-size first.")
    parts = [torch.load(f) for f in files]
    if len(parts) == 1:
        return parts[0]
    classes, labels = [], []
    for name, e in zip(names, parts):
        labels.append(e["labels"] + len(classes))
        classes += [f"{name}/{c}" for c in e["classes"]]
    return {
        "features": torch.cat([e["features"] for e in parts]),
        "labels": torch.cat(labels),
        "classes": classes,
        "paths": [p for e in parts for p in e["paths"]],
    }


def make_transform(model_id: str, image_size: int) -> v2.Compose:
    """Resize the short side to `image_size` and center-crop to a square, normalizing with
    the model's own mean/std (read from its Hugging Face image processor config)."""
    proc = AutoImageProcessor.from_pretrained(model_id)
    return v2.Compose([
        v2.PILToTensor(),
        v2.Resize(image_size, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
        v2.CenterCrop(image_size),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(proc.image_mean, proc.image_std),
    ])


@torch.no_grad()
def embed_split(
    model: torch.nn.Module,
    split_dir: Path,
    out_path: Path,
    device: torch.device,
    transform: v2.Compose,
    batch_size: int,
    num_workers: int,
) -> None:
    """Embed every image under `split_dir` (ImageFolder layout) and save to `out_path`."""
    ds = datasets.ImageFolder(str(split_dir), transform=transform)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=device.type == "cuda")
    feats = []
    for x, _ in tqdm(dl, desc=f"embedding {split_dir.parent.name}/{split_dir.name}"):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(pixel_values=x.to(device, non_blocking=True))
        feats.append(out.pooler_output.float().cpu())

    paths = [Path(p) for p, _ in ds.samples]
    result = {
        "features": torch.cat(feats),
        "labels": torch.tensor(ds.targets),
        "classes": ds.classes,
        "paths": [str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p) for p in paths],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    torch.save(result, tmp)
    tmp.rename(out_path)
    print(f"wrote {out_path} {tuple(result['features'].shape)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=stargan.ALL, default="afhq_v2")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    args = p.parse_args()

    todo: list[tuple[Path, Path]] = []  # (split dir, output file)
    for name in stargan.members(args.dataset):
        ds_root = stargan.download(name, args.data_root)
        out_dir = latent_dir(name, args.model, args.image_size, args.data_root)
        todo += [(ds_root / s, out_dir / f"{s}.pt") for s in stargan.splits(name) if not (out_dir / f"{s}.pt").exists()]
    if not todo:
        print(f"All splits of {args.dataset} already embedded with {args.model} at {args.image_size}px")
        return

    device = get_device()
    print(f"Loading {args.model} on {device}")
    model = AutoModel.from_pretrained(args.model).eval().to(device)
    transform = make_transform(args.model, args.image_size)
    for split_dir, out_path in todo:
        embed_split(model, split_dir, out_path, device, transform, args.batch_size, args.num_workers)


if __name__ == "__main__":
    main()
