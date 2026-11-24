"""fastdocling data: the target's own greedy DocTags over page images (written by `prepare`), collated for the frozen target."""

from __future__ import annotations

import io

import lightning as L
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor

from config import PROMPT, TrainConfig
from pages import distill_shards


def chat_prompt(processor) -> str:
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    return processor.apply_chat_template(messages, add_generation_prompt=True)


class Pages(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        r = self.ds[i]
        return r["image"], r["tokens"]


class DistillData(L.LightningDataModule):
    """Pages split by index: the first `val_pages` are validation. The collate builds image+prompt with the target's processor
    and appends the stored response tokens, so what the drafter sees is exactly what the target generated."""

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: str | None = None) -> None:
        files = distill_shards(self.cfg.data, target=self.cfg.target)
        ds = load_dataset("parquet", data_files=files, split="train")
        n = self.cfg.val_pages
        self.val_ds, self.train_ds = Pages(ds.select(range(n))), Pages(ds.select(range(n, len(ds))))
        self.processor = AutoProcessor.from_pretrained(self.cfg.target)
        self.prompt = chat_prompt(self.processor)

    def collate(self, rows):
        images = [[Image.open(io.BytesIO(b)).convert("RGB")] for b, _ in rows]
        enc = self.processor(text=[self.prompt] * len(rows), images=images, padding=True, return_tensors="pt")
        seqs, masks, kept = [], [], []
        for i, (_, toks) in enumerate(rows):
            prompt_ids = enc["input_ids"][i, enc["attention_mask"][i].bool()]
            prompt_len = len(prompt_ids)
            resp = torch.tensor(toks[: max(self.cfg.max_len - prompt_len, 0)], dtype=torch.long)
            if len(resp) < 2:  # one target-provided token followed by at least one draft target
                continue
            kept.append(i)
            seqs.append(torch.cat([prompt_ids, resp]))
            masks.append(torch.cat([torch.zeros(prompt_len, dtype=torch.bool), torch.ones(len(resp), dtype=torch.bool)]))
        if not kept:
            return None
        T = max(len(s) for s in seqs)
        pad = self.processor.tokenizer.pad_token_id
        ids = torch.full((len(seqs), T), pad, dtype=torch.long)
        resp_mask = torch.zeros(len(seqs), T, dtype=torch.bool)
        attn = torch.zeros(len(seqs), T, dtype=torch.long)
        for i, (s, m) in enumerate(zip(seqs, masks)):
            ids[i, : len(s)], resp_mask[i, : len(s)], attn[i, : len(s)] = s, m, 1
        pixel_mask = enc.get("pixel_attention_mask")
        return dict(input_ids=ids, attention_mask=attn, mask=resp_mask, pixel_values=enc["pixel_values"][kept],
                    pixel_attention_mask=pixel_mask[kept] if pixel_mask is not None else None)

    def _loader(self, ds, train: bool) -> DataLoader:
        n = self.cfg.num_workers
        return DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=train, drop_last=train, num_workers=n,
                          persistent_workers=n > 0 and train, collate_fn=self.collate)

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, False)
