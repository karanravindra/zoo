import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 4096
    context_length: int = 512
    n_layers: int = 5
    n_heads: int = 4
    d_model: int = 256
    dropout: float = 0.0

    def __post_init__(self):
        if (
            min(
                self.vocab_size,
                self.context_length,
                self.n_layers,
                self.n_heads,
                self.d_model,
            )
            < 1
        ):
            raise ValueError("Model dimensions must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.dropout = config.dropout
        self.attn_norm = nn.LayerNorm(config.d_model)
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.mlp_norm = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model, bias=False),
        )
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.attn_norm(x)).chunk(3, dim=-1)
        q, k, v = [
            t.view(batch, length, self.n_heads, width // self.n_heads).transpose(1, 2)
            for t in (q, k, v)
        ]
        attention = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attention = attention.transpose(1, 2).contiguous().view(batch, length, width)
        x = x + self.residual_dropout(self.proj(attention))
        return x + self.residual_dropout(self.mlp(self.mlp_norm(x)))


class GPT(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or GPTConfig()
        config = self.config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.d_model)
        self.apply(self._init_weights)
        for block in self.blocks:
            for weight in (block.proj.weight, block.mlp[2].weight):
                nn.init.normal_(weight, std=0.02 / math.sqrt(2 * config.n_layers))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, tokens, targets=None):
        length = tokens.size(1)
        if not 1 <= length <= self.config.context_length:
            raise ValueError("Input length must be between 1 and context_length")
        positions = torch.arange(length, device=tokens.device)
        x = self.dropout(
            self.token_embedding(tokens) + self.position_embedding(positions)
        )
        x = self.norm(self.blocks(x))
        # Reuse the embedding weights for the output projection.
        logits = F.linear(x, self.token_embedding.weight)
        loss = (
            None
            if targets is None
            else F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
            )
        )
        return logits, loss

    @torch.inference_mode()
    def generate(
        self, tokens, max_new_tokens=200, temperature=0.8, top_k=40, eos_id=None
    ):
        if max_new_tokens < 0 or temperature <= 0 or top_k < 0:
            raise ValueError("Require max_new_tokens >= 0, temperature > 0, top_k >= 0")
        if tokens.size(0) != 1:
            raise ValueError("Generation supports one prompt at a time")
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(tokens[:, -self.config.context_length :])
            logits = logits[:, -1] / temperature
            if top_k:
                threshold = logits.topk(min(top_k, logits.size(-1))).values[:, -1:]
                logits = logits.masked_fill(logits < threshold, float("-inf"))
            next_token = torch.multinomial(logits.softmax(dim=-1), num_samples=1)
            tokens = torch.cat((tokens, next_token), dim=1)
            if eos_id is not None and next_token.item() == eos_id:
                break
        return tokens
