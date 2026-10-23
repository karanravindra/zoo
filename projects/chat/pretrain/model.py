import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torchinfo import summary


@dataclass
class GPTConfig:
    vocab_size: int = 8192
    context_length: int = 512
    n_layers: int = 8
    n_heads: int = 4
    n_kv_heads: int = 1
    d_model: int = 256
    m: int = int(256 * 2.5)
    dropout: float = 0.0
    rope_theta: float = 10_000.0

    def __post_init__(self):
        if (
            min(
                self.vocab_size,
                self.context_length,
                self.n_layers,
                self.n_heads,
                self.n_kv_heads,
                self.d_model,
                self.m,
            )
            < 1
        ):
            raise ValueError("Model dimensions must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("head dimension must be even for RoPE")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout
        self.attn_norm = nn.LayerNorm(config.d_model)
        self.qkv = nn.Linear(
            config.d_model,
            config.d_model + 2 * config.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.mlp_norm = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.m, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.m, config.d_model, bias=False),
        )
        self.residual_dropout = nn.Dropout(config.dropout)
        frequencies = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        positions = torch.arange(config.context_length, dtype=torch.float32)
        angles = torch.outer(positions, frequencies)
        self.register_buffer("rope_cos", angles.cos(), persistent=False)
        self.register_buffer("rope_sin", angles.sin(), persistent=False)

    def apply_rope(self, x, length):
        cos = self.rope_cos[:length].to(dtype=x.dtype)[None, None, :, :, None]
        sin = self.rope_sin[:length].to(dtype=x.dtype)[None, None, :, :, None]
        x = x.unflatten(-1, (-1, 2))
        rotated = torch.stack((-x[..., 1], x[..., 0]), dim=-1)
        return (x * cos + rotated * sin).flatten(-2)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.attn_norm(x)).split(
            (width, self.n_kv_heads * self.head_dim, self.n_kv_heads * self.head_dim),
            dim=-1,
        )
        q = q.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k, v = [
            t.view(batch, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
            for t in (k, v)
        ]
        q = self.apply_rope(q, length)
        k = self.apply_rope(k, length)
        attention = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
            enable_gqa=self.n_kv_heads != self.n_heads,
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
        x = self.dropout(self.token_embedding(tokens))
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


if __name__ == "__main__":
    summary(GPT(), input_data=torch.zeros((1, 512), dtype=torch.long), depth=2)
