"""TopK sparse autoencoder (Gao et al. 2024, "Scaling and evaluating sparse autoencoders").

    pre   = (x - b_dec) @ W_enc + b_enc
    z     = TopK(ReLU(pre), k)                 # k active latents per token
    x_hat = z @ W_dec + b_dec                  # rows of W_dec are unit-norm feature directions

The SAE works on scaled inputs: `input_scale` (a buffer, set once from the training data) scales activations so that
E||x||^2 = d_model, which makes the losses and learning rate independent of the layer's norm. `reconstruct` and
`latents` take raw activations.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TopKSAE(nn.Module):
    def __init__(self, d_model: int, n_latents: int, k: int):
        super().__init__()
        self.d_model, self.n_latents, self.k = d_model, n_latents, k
        w_dec = torch.randn(n_latents, d_model)
        w_dec /= w_dec.norm(dim=-1, keepdim=True)
        self.W_dec = nn.Parameter(w_dec)
        self.W_enc = nn.Parameter(w_dec.T.clone())  # tied init
        self.b_enc = nn.Parameter(torch.zeros(n_latents))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.register_buffer("input_scale", torch.tensor(1.0))

    # ------------------------------------------------------------ scaled space (what training sees)

    def pre_acts(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def topk(self, pre: torch.Tensor, k: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """(values, indices) of the k largest ReLU'd pre-activations per row."""
        return torch.relu(pre).topk(k or self.k, dim=-1, sorted=False)

    def decode(self, vals: torch.Tensor, idx: torch.Tensor, bias: bool = True) -> torch.Tensor:
        # sum of the k selected decoder rows, weighted: no dense [n, n_latents] code is built
        out = F.embedding_bag(idx, self.W_dec, per_sample_weights=vals.to(self.W_dec.dtype), mode="sum")
        return out + self.b_dec if bias else out

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """(x_hat, values, indices, pre-activations) for scaled inputs x [n, d]."""
        pre = self.pre_acts(x)
        vals, idx = self.topk(pre)
        return self.decode(vals, idx), vals, idx, pre

    # ------------------------------------------------------------ raw activations

    def latents(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(values, indices) of the active latents for raw activations x [n, d]."""
        return self.topk(self.pre_acts(x.float() * self.input_scale))

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Raw activations -> reconstruction in the same (raw) space and dtype."""
        x_hat, *_ = self(x.float() * self.input_scale)
        return (x_hat / self.input_scale).to(x.dtype)

    # ------------------------------------------------------------ decoder constraint

    @torch.no_grad()
    def remove_parallel_grads(self):
        """Project out the gradient component along each decoder row (it would only change the row's norm)."""
        if self.W_dec.grad is not None:
            w = self.W_dec
            self.W_dec.grad -= (self.W_dec.grad * w).sum(-1, keepdim=True) * w

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec /= self.W_dec.norm(dim=-1, keepdim=True)
