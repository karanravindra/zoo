"""Embed CelebA-HQ's images and class-name prompts with a vision-language encoder chosen by `--model`.

Run from anywhere:
    uv run projects/celeba_gen/embed.py
    uv run projects/celeba_gen/embed.py --model siglip2-so400m

Files that already exist are skipped, including text.pt, so delete text.pt before changing --prompt-template.

Each model has one image tower and one text tower that map into a shared space, so an image and a text prompt can be
compared with a dot product. Features are L2-normalized, so the dot product is cosine similarity.

Downloads CelebA-HQ if needed (`zoo.data.stargan`), then writes:
  <data>/latents/celeba_hq/vl-<model>/<split>.pt
      {"features": (N, D) float32 unit-norm image embeddings, "labels": (N,), "classes": class names, "paths": image paths}
  <data>/latents/celeba_hq/vl-<model>/text.pt
      {"features": (C, D) float32 unit-norm embeddings of `prompt_template` filled with each class name,
       "classes": class names, "prompts": the strings embedded}
Files that already exist are skipped. The layout matches `projects/latent_analysis/embed_vl.py`, so both projects read
the same files.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
import tyro
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from zoo.data import stargan
from zoo.paths import DATA_ROOT, REPO_ROOT
from zoo.utils.device import get_device

NAME = "celeba_hq"

# flag name -> Hugging Face model id. All load with AutoModel / AutoProcessor and expose
# get_image_features / get_text_features. Check each id on the Hub before the first run.
MODELS: dict[str, str] = {
    "siglip2-base-256": "google/siglip2-base-patch32-256",
    "siglip2-base": "google/siglip2-base-patch16-224",
    "siglip2-so400m": "google/siglip2-so400m-patch14-384",
    "clip-l": "openai/clip-vit-large-patch14",
}


def out_dir(model: str, data_root: Path = DATA_ROOT) -> Path:
    """e.g. data/latents/celeba_hq/vl-siglip2-base-256"""
    return data_root / "latents" / NAME / f"vl-{model}"


def _open_rgb(path: str) -> Image.Image:
    """Module-level so DataLoader workers can pickle it (macOS spawns workers, which rules out lambdas)."""
    return Image.open(path).convert("RGB")


def _collate(batch, processor):
    """Run the model's own image processor on a batch of PIL images (resize, normalize, any padding it needs)."""
    images = [img for img, _ in batch]
    labels = torch.tensor([y for _, y in batch])
    return processor(images=images, return_tensors="pt"), labels


def _as_tensor(out) -> torch.Tensor:
    """Some transformers versions return a tensor from get_*_features, others a ModelOutput whose pooled vector is the features."""
    if isinstance(out, torch.Tensor):
        return out
    return out.pooler_output


def _to_device(inputs: dict, device: torch.device) -> dict:
    """Move a processor's outputs to the device. Float tensors (pixel values) become bf16 to match the weights; token ids stay integer."""
    return {
        k: v.to(device=device, dtype=torch.bfloat16) if v.is_floating_point() else v.to(device)
        for k, v in inputs.items()
    }


def _save(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(result, tmp)
    tmp.rename(path)


@torch.no_grad()
def embed_images(model, processor, split_dir: Path, device: torch.device, batch_size: int, num_workers: int) -> dict:
    ds = datasets.ImageFolder(str(split_dir), loader=_open_rgb)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=partial(_collate, processor=processor),
        pin_memory=device.type == "cuda",
    )
    feats = []
    for inputs, _ in tqdm(dl, desc=f"embedding {split_dir.name}"):
        out = _as_tensor(model.get_image_features(**_to_device(inputs, device)))
        feats.append(F.normalize(out.float(), dim=-1).cpu())

    return {
        "features": torch.cat(feats),
        "labels": torch.tensor(ds.targets),
        "classes": ds.classes,
        "paths": [str(Path(p).relative_to(REPO_ROOT)) if Path(p).is_relative_to(REPO_ROOT) else p for p, _ in ds.samples],
    }


@torch.no_grad()
def embed_text(model, processor, prompts: list[str], device: torch.device) -> torch.Tensor:
    inputs = processor(text=prompts, padding="max_length", truncation=True, return_tensors="pt")
    out = _as_tensor(model.get_text_features(**_to_device(inputs, device)))
    return F.normalize(out.float(), dim=-1).cpu()


@dataclass
class EmbedConfig:
    model: Literal[tuple(MODELS)] = "siglip2-base-256"  # vision-language encoder (see MODELS)
    prompt_template: str = "a photo of a {}"  # text prompt per class; "{}" is replaced by the class name
    batch_size: int = 64  # images per forward pass
    num_workers: int = 2  # DataLoader workers for decoding images
    data_root: Path = DATA_ROOT  # directory holding celeba_hq/ and latents/
    compile: bool = False  # compile both towers in place with mode="reduce-overhead" (CUDA graphs on GPU)


def main(cfg: EmbedConfig) -> None:
    hf_id = MODELS[cfg.model]

    # Work out what is missing before loading the model, so a finished run costs nothing.
    root = stargan.download(NAME, cfg.data_root)
    out = out_dir(cfg.model, cfg.data_root)
    todo: list[tuple[Path, Path]] = [
        (root / s, out / f"{s}.pt") for s in stargan.splits(NAME) if not (out / f"{s}.pt").exists()
    ]
    if not (out / "text.pt").exists():
        todo.append((root / stargan.splits(NAME)[0], out / "text.pt"))
    if not todo:
        print(f"All of {NAME} already embedded with {cfg.model}")
        return

    device = get_device()
    print(f"Loading {hf_id} on {device}")
    processor = AutoProcessor.from_pretrained(hf_id)
    model = AutoModel.from_pretrained(hf_id, dtype=torch.bfloat16).eval().to(device)
    if cfg.compile:
        # get_image_features / get_text_features call the towers, so compiling the towers is what takes effect.
        model.vision_model.compile(mode="reduce-overhead", dynamic=False)
        model.text_model.compile(mode="reduce-overhead", dynamic=False)

    for split_dir, out_path in todo:
        if out_path.name == "text.pt":
            classes = datasets.ImageFolder(str(split_dir)).classes
            prompts = [cfg.prompt_template.format(c.replace("_", " ")) for c in classes]
            feats = embed_text(model, processor, prompts, device)
            _save({"features": feats, "classes": classes, "prompts": prompts}, out_path)
            print(f"wrote {out_path} {tuple(feats.shape)}")
        else:
            result = embed_images(model, processor, split_dir, device, cfg.batch_size, cfg.num_workers)
            _save(result, out_path)
            print(f"wrote {out_path} {tuple(result['features'].shape)}")


if __name__ == "__main__":
    main(tyro.cli(EmbedConfig))
