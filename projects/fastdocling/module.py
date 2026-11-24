"""LitDrafter: a drafter trained against the frozen target's hidden states.

The target is deliberately *not* a registered submodule (it sits in a one-element list): the EMA callback, the param-stat
callback and checkpoints all walk `module.parameters()`, and none of them should touch 500 MB of frozen weights. So a checkpoint
holds only the drafter, and the target is rebuilt from `cfg.target`.
"""

from __future__ import annotations

import lightning as L
import torch
from transformers import AutoConfig, AutoModelForImageTextToText

from config import TrainConfig
from drafters.base import Features, make_drafter

from zoo.training import compile_, make_scheduler


def load_target(name: str, device, dtype=torch.bfloat16) -> torch.nn.Module:
    model = AutoModelForImageTextToText.from_pretrained(name, dtype=dtype)
    return model.requires_grad_(False).eval().to(device)


class LitDrafter(L.LightningModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = TrainConfig.from_dict(cfg)
        text = AutoConfig.from_pretrained(self.cfg.target).text_config
        self.hidden_size = text.hidden_size
        self.drafter = make_drafter(self.cfg, text.hidden_size, len(self.cfg.feature_layers))
        self._target: list[torch.nn.Module] = []  # lazily loaded, see the module docstring

    @property
    def target(self) -> torch.nn.Module:
        if not self._target:
            self._target.append(load_target(self.cfg.target, self.device))
        if next(self._target[0].parameters()).device != self.device:
            self._target[0].to(self.device)
        return self._target[0]

    def setup(self, stage: str) -> None:
        if self.cfg.compile and stage == "fit":
            compile_(self.drafter)

    @torch.no_grad()
    def features(self, batch: dict) -> Features:
        target = self.target
        out = target.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                           pixel_values=batch["pixel_values"].to(torch.bfloat16),
                           pixel_attention_mask=batch["pixel_attention_mask"], output_hidden_states=True)
        hs = out.hidden_states
        hidden = torch.cat([hs[i] for i in self.cfg.feature_layers], dim=-1)
        return Features(hidden=hidden, last=out.last_hidden_state, input_ids=batch["input_ids"], mask=batch["mask"],
                        embed=target.get_input_embeddings())

    def _step(self, batch, prefix: str, on_step: bool):
        if batch is not None:
            out = self.drafter(self.features(batch))
            n = int(out.valid.any(-1).sum())
            if n:
                return self._log_drafts(out, prefix, on_step, n)
        if not on_step:
            return None
        if self.trainer.world_size == 1:
            return None
        # A rank with no usable targets still participates in distributed gradient reduction.
        return torch.stack([p.reshape(-1)[0] for p in self.drafter.parameters() if p.requires_grad]).sum() * 0

    def _log_drafts(self, out, prefix: str, on_step: bool, n: int):
        chain = out.match.long().cumprod(dim=1)
        self.log(f"{prefix}/loss", out.loss, on_step=on_step, on_epoch=True, batch_size=n)
        # Expected extra tokens accepted per verification pass under greedy verification (the bonus token comes on top).
        self.log(f"{prefix}/accept_len", chain.sum(1).float().mean(), on_step=on_step, on_epoch=True, batch_size=n)
        for name, value in (out.logs or {}).items():
            self.log(f"{prefix}/{name}", value, on_step=on_step, on_epoch=True, batch_size=n)
        for j in range(out.match.shape[1]):
            denom = out.valid[:, j].sum().clamp(min=1)
            self.log(f"{prefix}/acc@{j + 1}", out.match[:, j].sum() / denom, on_epoch=True, batch_size=n)
        return out.loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train", True)

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val", False)

    def test_step(self, batch, batch_idx):
        self._step(batch, "test", False)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.drafter.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        return {"optimizer": opt, "lr_scheduler": make_scheduler(opt, self.cfg, int(self.trainer.estimated_stepping_batches))}
