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
    # DeepSeek-V3 style mixture of experts; the defaults match the dense MLP's
    # active width: (n_shared_experts + moe_top_k) * moe_m == m.
    moe: bool = False
    n_routed_experts: int = 16
    n_shared_experts: int = 1
    moe_top_k: int = 3
    moe_m: int = 160
    moe_routed_scale: float = 1.0

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
        if self.moe:
            if min(self.n_routed_experts, self.moe_top_k, self.moe_m) < 1:
                raise ValueError("MoE sizes must be positive")
            if self.n_shared_experts < 0 or self.moe_routed_scale <= 0:
                raise ValueError("Require n_shared_experts >= 0, moe_routed_scale > 0")
            if self.moe_top_k > self.n_routed_experts:
                raise ValueError("moe_top_k must not exceed n_routed_experts")


class PermuteRows(torch.autograd.Function):
    """Row gather by a permutation whose backward is the inverse gather.

    Plain indexing would backpropagate through an atomic scatter-add, which is
    much slower than a second gather when the indices are a permutation.
    """

    @staticmethod
    def forward(ctx, x, order, inverse):
        ctx.save_for_backward(inverse, order)
        return x[order]

    @staticmethod
    def backward(ctx, grad):
        inverse, order = ctx.saved_tensors
        return grad[inverse], None, None


class MoE(nn.Module):
    """DeepSeek-V3 MoE: shared experts plus sigmoid-routed fine-grained experts.

    Load balancing is auxiliary-loss-free: a per-expert bias steers top-k selection
    only (not the gate weights) and is nudged by `update_bias` after each step.
    """

    def __init__(self, config):
        super().__init__()
        self.n_experts = config.n_routed_experts
        self.top_k = config.moe_top_k
        self.routed_scale = config.moe_routed_scale
        self.router = nn.Linear(config.d_model, self.n_experts, bias=False)
        self.up = nn.Parameter(torch.empty(self.n_experts, config.d_model, config.moe_m))
        self.down = nn.Parameter(
            torch.empty(self.n_experts, config.moe_m, config.d_model)
        )
        shared_m = config.n_shared_experts * config.moe_m
        self.shared = (
            nn.Sequential(
                nn.Linear(config.d_model, shared_m, bias=False),
                nn.GELU(approximate="tanh"),
                nn.Linear(shared_m, config.d_model, bias=False),
            )
            if shared_m
            else None
        )
        self.register_buffer("bias", torch.zeros(self.n_experts))
        self.register_buffer("load", torch.zeros(self.n_experts), persistent=False)

    def output_weights(self):
        return (self.down, *([self.shared[2].weight] if self.shared else []))

    @torch.no_grad()
    def update_bias(self, rate):
        # Raise the bias of underloaded experts and lower it for overloaded ones.
        self.bias += rate * (self.load.mean() - self.load).sign()
        self.load.zero_()

    @staticmethod
    def grouped_mm(x, weight, counts, dtype):
        x, weight = x.to(dtype), weight.to(dtype)
        if x.is_cuda:
            return torch._grouped_mm(x, weight, offs=counts.cumsum(0).to(torch.int32))
        # Reference path for devices without grouped GEMM support.
        chunks = x.split(counts.tolist())
        return torch.cat([c @ w for c, w in zip(chunks, weight)])

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        scores = torch.sigmoid(F.linear(x.float(), self.router.weight.float()))
        _, chosen = (scores + self.bias).topk(self.top_k, dim=-1)
        gates = scores.gather(1, chosen)
        gates = gates / gates.sum(dim=-1, keepdim=True) * self.routed_scale
        flat = chosen.reshape(-1)
        counts = torch.zeros(
            self.n_experts, dtype=torch.long, device=x.device
        ).scatter_add_(0, flat, torch.ones_like(flat))
        if self.training:
            self.load += counts
        # Sort token-expert pairs by expert so each expert sees a contiguous block.
        order = flat.argsort(stable=True)
        inverse = order.argsort()
        dtype = (
            torch.get_autocast_dtype(x.device.type)
            if torch.is_autocast_enabled(x.device.type)
            else x.dtype
        )
        # Row i of the repeated input is token i // top_k's copy for expert flat[i].
        repeated = x.to(dtype).repeat_interleave(self.top_k, dim=0)
        hidden = F.gelu(
            self.grouped_mm(
                PermuteRows.apply(repeated, order, inverse), self.up, counts, dtype
            ),
            approximate="tanh",
        )
        out = PermuteRows.apply(
            self.grouped_mm(hidden, self.down, counts, dtype), inverse, order
        )
        y = (out.view(-1, self.top_k, out.size(-1)) * gates.unsqueeze(-1).to(dtype)).sum(
            dim=1
        )
        if self.shared is not None:
            y = y + self.shared(x).to(y.dtype)
        return y.view(shape)


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
        self.mlp = (
            MoE(config)
            if config.moe
            else nn.Sequential(
                nn.Linear(config.d_model, config.m, bias=False),
                nn.GELU(approximate="tanh"),
                nn.Linear(config.m, config.d_model, bias=False),
            )
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
            output_weights = (
                block.mlp.output_weights() if config.moe else (block.mlp[2].weight,)
            )
            for weight in (block.proj.weight, *output_weights):
                nn.init.normal_(weight, std=0.02 / math.sqrt(2 * config.n_layers))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, MoE):
            nn.init.normal_(module.up, std=0.02)
            nn.init.normal_(module.down, std=0.02)

    def update_router_bias(self, rate=1e-3):
        """Apply the aux-loss-free balancing update; call once per optimizer step."""
        for block in self.blocks:
            if isinstance(block.mlp, MoE):
                block.mlp.update_bias(rate)

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
