"""Embedders: a list of image paths or texts in, an (N, D) float32 array of unit-norm vectors out.

One class per modality, each loaded from a Hugging Face id. `resolve` maps a `--model` flag (a registry key or any
Hugging Face id) to its id and, for text, its pooling. Add a modality by adding a class with `embed(values, ...)` and an
entry in `EMBEDDERS`.
"""

from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from zoo.utils.device import get_device

# key -> Hugging Face id. Image models expose get_image_features (SigLIP / CLIP style).
IMAGE_MODELS = {
    "siglip2-base-256": "google/siglip2-base-patch32-256",
    "siglip2-base": "google/siglip2-base-patch16-224",
    "siglip2-so400m": "google/siglip2-so400m-patch14-384",
    "clip-l": "openai/clip-vit-large-patch14",
    # No text tower: the image vector is the pooled CLS token (appearance and layout, rather than concepts).
    "dinov2-small": "facebook/dinov2-small",
    "dinov2-base": "facebook/dinov2-base",
    # Gated on the Hub: accept the license on the model page with the account behind HF_TOKEN.
    "dinov3-small": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "dinov3-base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3-large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}
# key -> (Hugging Face id, pooling). "cls" takes the first token, "mean" averages the tokens under the attention mask.
TEXT_MODELS = {
    "bge-small": ("BAAI/bge-small-en-v1.5", "cls"),
    "bge-base": ("BAAI/bge-base-en-v1.5", "cls"),
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "mean"),
}
DEFAULTS = {"image": "siglip2-base-256", "text": "bge-small"}


def resolve(modality: str, model: str) -> tuple[str, str, str]:
    """(file-name key, Hugging Face id, pooling) for a `--model` flag. An unknown key is taken as a Hugging Face id."""
    key = model or DEFAULTS[modality]
    if modality == "image":
        return key, IMAGE_MODELS.get(key, key), ""
    hf_id, pooling = TEXT_MODELS.get(key, (key, "mean"))
    return key, hf_id, pooling


class _Images(Dataset):
    """Module-level so DataLoader workers can pickle it."""

    def __init__(self, paths: list[str]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> Image.Image:
        return Image.open(self.paths[i]).convert("RGB")


def _collate(batch: list[Image.Image], processor):
    return processor(images=batch, return_tensors="pt")


def _features(out) -> torch.Tensor:
    """get_*_features returns a tensor in some transformers versions and a ModelOutput in others."""
    return out if isinstance(out, torch.Tensor) else out.pooler_output


class ImageEmbedder:
    def __init__(self, hf_id: str):
        self.device = get_device()
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(hf_id, dtype=self.dtype).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(hf_id)

    @torch.no_grad()
    def embed(self, paths: list[str], batch_size: int = 64, num_workers: int = 4) -> np.ndarray:
        dl = DataLoader(
            _Images(paths), batch_size=batch_size, num_workers=num_workers,
            collate_fn=partial(_collate, processor=self.processor),
        )  # fmt: skip
        feats = []
        for inputs in tqdm(dl, desc="embedding images"):
            inputs = {
                k: v.to(self.device, self.dtype) if v.is_floating_point() else v.to(self.device)
                for k, v in inputs.items()
            }
            if hasattr(self.model, "get_image_features"):
                out = _features(self.model.get_image_features(**inputs))
            else:  # a vision-only model (DINO): the pooled CLS token
                out = self.model(**inputs).pooler_output
            feats.append(F.normalize(out.float(), dim=-1).cpu())
        return torch.cat(feats).numpy()


class TextEmbedder:
    def __init__(self, hf_id: str, pooling: str, max_length: int = 512):
        self.device = get_device()
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(hf_id, dtype=self.dtype).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        self.pooling, self.max_length = pooling, max_length

    @torch.no_grad()
    def embed(self, texts: list[str], batch_size: int = 64, num_workers: int = 0) -> np.ndarray:
        # Longest first wastes the least padding; `inverse` restores the input order.
        order = np.argsort([-len(t) for t in texts], kind="stable")
        out = np.zeros((len(texts), self.model.config.hidden_size), dtype=np.float32)
        for start in tqdm(range(0, len(texts), batch_size), desc="embedding text"):
            idx = order[start : start + batch_size]
            enc = self.tokenizer(
                [texts[i] for i in idx], padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
            ).to(self.device)
            hidden = self.model(**enc).last_hidden_state.float()
            if self.pooling == "cls":
                pooled = hidden[:, 0]
            else:
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            out[idx] = F.normalize(pooled, dim=-1).cpu().numpy()
        return out


def make_embedder(modality: str, model: str):
    """(file-name key, embedder) for a modality and `--model` flag."""
    key, hf_id, pooling = resolve(modality, model)
    return key, (ImageEmbedder(hf_id) if modality == "image" else TextEmbedder(hf_id, pooling))
