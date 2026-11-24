"""Pretraining: the next-token LightningModule over `model.Llama`."""

from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
from config import Config
from cut_cross_entropy import linear_cross_entropy
from data import checkpoint_corpus, load_tokenizer
from lightning.pytorch.loggers import WandbLogger
from model import Llama

from zoo.metrics.text import bits_per_byte
from zoo.training import compile_, make_scheduler


class LitGPT(L.LightningModule):
    """Training wrapper. Built from the config as a plain dict, so `load_from_checkpoint(path)` rebuilds it.

    Logs `train/loss`, `train/tokens_seen` and `train/flops` (cumulative, over all devices; see
    `Config.train_flops_per_token` for what a FLOP counts), and `val/loss`, `val/bpb` on the validation windows."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = Config.from_dict(cfg)
        self.net = Llama(self.cfg)
        self.tokens_seen = 0
        self.loss_impl = "torch"

    def setup(self, stage: str) -> None:
        on_cuda_bf16 = self.device.type == "cuda" and "bf16" in self.cfg.precision
        self.loss_impl = ("cce" if on_cuda_bf16 else "torch_compile") if self.cfg.loss == "auto" else self.cfg.loss
        if self.cfg.compile and stage == "fit":
            compile_(self.net, cudagraphs=self.cfg.grad_accum == 1)  # CUDA graphs cannot accumulate gradients over micro-batches

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.net.logits(self.net(idx))

    def lm_loss(self, window: torch.Tensor) -> torch.Tensor:
        """Mean next-token cross-entropy over a batch of (B, block_size + 1) windows. The tied embedding is the
        classifier, so with a cut cross-entropy the (B, block_size, vocab) logits never exist."""
        x, y = window[:, :-1], window[:, 1:]
        h, w = self.net(x), self.net.classifier_weight
        if self.loss_impl == "torch":
            return F.cross_entropy(F.linear(h, w).float().flatten(0, 1), y.flatten())
        if "bf16" in self.cfg.precision:
            h, w = h.bfloat16(), w.bfloat16()  # the kernel wants 16-bit inputs; the residual stream under autocast is fp32
        return linear_cross_entropy(h, w, y, impl=self.loss_impl)

    def token_losses(self, window: torch.Tensor) -> torch.Tensor:
        """The next-token cross-entropy of every position, (B, block_size): `lm_loss` without the mean."""
        x, y = window[:, :-1], window[:, 1:]
        h, w = self.net(x), self.net.classifier_weight
        if self.loss_impl == "torch":
            return F.cross_entropy(F.linear(h, w).float().flatten(0, 1), y.flatten(), reduction="none").view_as(y)
        if "bf16" in self.cfg.precision:
            h, w = h.bfloat16(), w.bfloat16()
        return linear_cross_entropy(h, w, y, impl=self.loss_impl, reduction="none")

    def position_buckets(self) -> list[tuple[int, int]]:
        """Position ranges (from, to) of a window that `val_by_pos` reports separately: 0-32, 32-64, 64-128, ... up to block_size."""
        t = self.cfg.block_size
        edges = [0] + [e for e in (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192) if e < t] + [t]
        return list(zip(edges[:-1], edges[1:]))

    def training_step(self, window: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss = self.lm_loss(window)
        self.tokens_seen += (window.shape[1] - 1) * window.shape[0] * self.trainer.world_size
        self.log("train/loss", loss, on_step=True, on_epoch=False, batch_size=len(window))
        self.log("train/tokens_seen", float(self.tokens_seen), on_step=True, on_epoch=False)
        self.log("train/flops", self.tokens_seen * self.cfg.train_flops_per_token, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, window: torch.Tensor, batch_idx: int) -> None:
        if self.cfg.val_by_pos:  # the same mean as lm_loss, plus the mean over each range of positions
            per_token = self.token_losses(window)
            loss = per_token.mean()
            for a, b in self.position_buckets():
                self.log(f"val/loss_pos_{a}_{b}", per_token[:, a:b].mean(), on_step=False, on_epoch=True, batch_size=len(window), sync_dist=True)
        else:
            loss = self.lm_loss(window)
        self.log("val/loss", loss, on_step=False, on_epoch=True, batch_size=len(window), sync_dist=True)
        if self.cfg.tokens_per_byte:
            bpb = bits_per_byte(loss, self.cfg.tokens_per_byte, 1)  # tokens per byte, so one byte's worth of tokens
            self.log("val/bpb", bpb, on_step=False, on_epoch=True, batch_size=len(window), sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        """Logs the running totals with the validation metrics, so W&B has `val/loss` and its x-axis in one row (a
        validation step is not a multiple of `log_every`, so the `train/` totals are never logged with it), then a text sample."""
        cfg = self.cfg
        self.log("val/tokens_seen", float(self.tokens_seen))
        self.log("val/flops", self.tokens_seen * cfg.train_flops_per_token)
        if self.trainer.sanity_checking or not cfg.sample_tokens or not self.trainer.is_global_zero:
            return
        loggers = [lg for lg in self.trainer.loggers if isinstance(lg, WandbLogger)]
        if not loggers:
            return
        tok = load_tokenizer(cfg.dataset, root=checkpoint_corpus(cfg))
        text = cfg.sample_prompt
        prompt = torch.tensor([[int(t) for t in tok.encode(text)]], device=self.device)
        g = torch.Generator(device=self.device).manual_seed(1234)  # the same seed at every validation, so samples are comparable
        with torch.autocast(self.device.type, torch.bfloat16, enabled="bf16" in cfg.precision):
            out = self.net.generate(prompt, cfg.sample_tokens, cfg.sample_temperature, cfg.sample_top_k, eos_id=cfg.eot_id, generator=g)
        decoded = tok.decode(out[0].tolist()).decode("utf-8", errors="replace")
        for lg in loggers:
            lg.log_text("samples", columns=["step", "text"], data=[[self.global_step, decoded]])

    def on_fit_start(self) -> None:
        """Adds what the config implies (and a W&B chart cannot compute) to the logged config: model size, FLOPs per token, peak LR."""
        cfg = self.cfg
        derived = {"n_params": cfg.n_params, "n_params_body": cfg.n_params_body, "flops_per_token": cfg.train_flops_per_token, "peak_lr": cfg.peak_lr}
        for lg in self.loggers:
            lg.log_hyperparams(derived)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["tokens_seen"] = self.tokens_seen

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self.tokens_seen = checkpoint.get("tokens_seen", 0)

    def configure_optimizers(self):
        decay = [p for p in self.net.parameters() if p.ndim >= 2]
        no_decay = [p for p in self.net.parameters() if p.ndim < 2]
        groups = [{"params": decay, "weight_decay": self.cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
        opt = torch.optim.AdamW(groups, lr=self.cfg.peak_lr, betas=(0.9, self.cfg.beta2))  # not fused: Lightning's AMP plugin cannot clip the gradients of a fused optimizer
        return {"optimizer": opt, "lr_scheduler": make_scheduler(opt, self.cfg, int(self.trainer.estimated_stepping_batches))}
