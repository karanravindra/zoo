"""GPT settings: the project's fields on top of the shared `RunConfig` (precision, W&B, ...), one config per command."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from zoo.training import RunConfig

LR_REF_DIM = 256  # the width that `lr` is the peak for when `lr_width_exp` is set
VAL_ALIGN = 64  # the validation set is a multiple of this many windows, so every micro-batch size up to it divides it


@dataclass
class Config(RunConfig):
    """Pretraining. The model fields rebuild the network from a checkpoint, so later stages reuse this class."""

    dataset: str = "tinystories"  # a corpus written by `prepare` (see SOURCES in prepare.py)
    # model (about 5.9M parameters with the defaults)
    vocab_size: int = 4096  # set from the corpus by cli.py
    eot_id: int = 256  # end-of-text id, set from the corpus by cli.py
    tokens_per_byte: float = 0.0  # of the validation split, set from the corpus by cli.py; turns the loss into bits per byte
    dim: int = 256  # width
    n_layers: int = 6
    aspect: int = 0  # > 0: dim = aspect * n_layers (overrides dim), so n_layers alone sizes the model: a ladder for scaling-law sweeps
    head_dim: int = 64  # attention heads = dim / head_dim
    block_size: int = 256  # context length
    multiple_of: int = 32  # the SwiGLU hidden size (about 8/3 * dim) is rounded up to a multiple of this
    ff_mult: float = 0.0  # > 0: the SwiGLU hidden size is ff_mult * dim (before rounding) instead of Llama's 8/3 * dim; 4 = a 4x MLP
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    doc_mask: bool = False  # attention stays inside a document (FlexAttention); off = plain causal attention (flash)
    # optimization (steps, not epochs: a pretraining run is usually less than one pass over the corpus)
    epochs: int = -1  # no epoch limit, `max_steps` ends the run
    max_steps: int = 5000  # optimizer steps (set from max_flops when that is)
    max_flops: float = 0.0  # > 0: train to this many FLOPs: max_steps = max_flops / (flops per token * global_batch), at most one pass over the corpus, so every size of a scaling-law sweep gets the same compute
    global_batch: int = 16_384  # tokens per optimizer step, over all devices; gradient accumulation fills what batch_size does not
    batch_size: int = 64  # sequences per device in one micro-batch (lower it if the GPU runs out of memory)
    micro_tokens: int = 0  # > 0: sets batch_size so that a micro-batch is this many tokens (at most global_batch / devices), whatever block_size is: a sweep over block_size or global_batch then needs no gradient accumulation
    lr: float = 1e-3  # peak of the schedule (the peak at dim 256 when lr_width_exp is set)
    lr_width_exp: float = 0.0  # the peak is lr * (256 / dim) ** lr_width_exp: one learning-rate rule for a ladder of widths; 0 = lr as given
    weight_decay: float = 0.1  # AdamW, 2D parameters only
    beta2: float = 0.95  # AdamW
    lr_schedule: Literal["constant", "cosine", "linear", "onecycle"] = "constant"  # no decay
    ema: float = 0.999  # validation, checkpoints and samples use the weight average
    warmup: int = 200
    grad_clip: float = 1.0
    log_every: int = 20  # steps between logged train metrics
    # evaluation
    val_every: int = 250  # steps between validations (and text samples)
    val_tokens: int = 2_000_000  # tokens in the validation set: the same evenly spaced windows in every run, whatever the micro-batch, so runs are comparable
    val_by_pos: bool = False  # also log the validation loss by position in the window (val/loss_pos_<from>_<to>), to compare context lengths
    sample_prompt: str = "Once upon a time"  # continued at every validation and logged to W&B as text
    sample_tokens: int = 128  # 0 = no samples
    sample_temperature: float = 0.8
    sample_top_k: int = 40
    # runtime
    loss: Literal["auto", "cce", "torch_compile", "torch"] = "auto"  # auto = Apple's cut cross-entropy on CUDA with bf16, else its torch.compile version; torch = plain F.cross_entropy
    devices: int = 1  # GPUs; more than one trains with DDP (supported, not yet exercised)

    def __post_init__(self) -> None:
        if self.aspect:
            self.dim = self.aspect * self.n_layers
        if self.micro_tokens:
            self.batch_size = max(min(self.micro_tokens, self.global_batch // self.devices) // self.block_size, 1)

    @property
    def hidden_dim(self) -> int:
        h = int(self.ff_mult * self.dim) if self.ff_mult else int(2 * (4 * self.dim) / 3)
        return self.multiple_of * ((h + self.multiple_of - 1) // self.multiple_of)

    @property
    def peak_lr(self) -> float:
        """The learning rate the optimizer starts from: `lr`, scaled to this width when `lr_width_exp` is set."""
        return self.lr * (LR_REF_DIM / self.dim) ** self.lr_width_exp

    @property
    def n_params_body(self) -> int:
        """Parameters outside the (tied) token embedding: the blocks and the final norm."""
        d, h = self.dim, self.hidden_dim
        return self.n_layers * (4 * d * d + 3 * d * h + 2 * d) + d

    @property
    def n_params(self) -> int:
        """All parameters; the tied embedding counts once."""
        return self.n_params_body + self.vocab_size * self.dim

    @property
    def n_heads(self) -> int:
        if self.dim % self.head_dim:
            raise ValueError(f"dim {self.dim} is not a multiple of head_dim {self.head_dim}")
        return self.dim // self.head_dim

    @property
    def grad_accum(self) -> int:
        """Micro-batches per optimizer step: `global_batch` tokens split over the devices and `batch_size` sequences each."""
        micro = self.batch_size * self.devices * self.block_size  # tokens in one micro-batch across devices
        if self.global_batch % micro:
            raise ValueError(f"global_batch {self.global_batch} is not a multiple of batch_size * devices * block_size = {micro}")
        return self.global_batch // micro

    @property
    def train_flops_per_token(self) -> float:
        """Forward + backward FLOPs per token at the full context: matmuls only, 2 FLOPs per multiply-add, the
        backward pass costing two forwards (the "6 * params * tokens" rule, plus attention). Norms, RoPE and
        activations are left out; causal masking is not credited, so attention counts every position."""
        d, t = self.dim, self.block_size
        attn = 8 * d * d + 4 * t * d  # q, k, v, o projections; QK^T and the weighted sum over t positions
        mlp = 6 * d * self.hidden_dim  # the three SwiGLU matrices
        forward = self.n_layers * (attn + mlp) + 2 * self.vocab_size * d  # plus the output projection
        return 3 * forward

    def budget_steps(self, train_tokens: int) -> int:
        """`max_steps` for the `max_flops` budget: the steps that spend it, at most one pass over `train_tokens` (so no
        window repeats), rounded down to a multiple of `val_every` so that the last step is validated."""
        by_flops = int(self.max_flops / (self.train_flops_per_token * self.global_batch))
        windows = (train_tokens - 1) // self.block_size
        per_epoch = windows // (self.batch_size * self.devices) // self.grad_accum
        steps = min(by_flops, per_epoch) // self.val_every * self.val_every
        if steps < 1:
            raise ValueError(f"max_flops {self.max_flops:.1e} is less than one validation interval ({self.val_every} steps) of this model")
        return steps


@dataclass
class PrepareConfig:
    """Download a corpus, train its tokenizer and write the token shards, so `pretrain` starts straight away."""

    dataset: str = "tinystories"  # a key of SOURCES in prepare.py
    vocab_size: int | None = None  # BPE vocabulary size, special tokens included (default: the dataset's)
    vocab_docs: int = 300_000  # training documents used to learn the BPE merges
    max_docs: int = 0  # keep only this many training documents (0 = all)
    shard_mb: int = 256  # megabytes of text per temporary text shard (bounds RAM while encoding)
    shard_tokens: int = 1 << 28  # tokens per token shard on disk
    force: bool = False  # rebuild even if the corpus exists


@dataclass
class LRFitConfig:
    """The best peak learning rate at each width of the learning-rate sweep, and one rule for all of them (`scaling.py`)."""

    group: str = "tinystories-lr"  # W&B group of the sweep (sweeps/scaling_lr.yaml); its finished runs are read from <runs>/gpt/
    near: float = 0.12  # a run joins the rule's regression when its loss is within this of its width's best
    out: Path | None = None  # figures and results.json (default <runs>/gpt/scaling/<group>)


@dataclass
class ScalingConfig:
    """Fit the loss as a function of model size and tokens on a ladder sweep (`scaling.py`): IsoFLOP profiles, the parametric law, held-out checks."""

    group: str = "tinystories-ladder"  # W&B group of the ladder sweeps (sweeps/scaling_ladder.yaml and its bracket/seed siblings)
    params: Literal["total", "body"] = "total"  # N: every parameter (the tied embedding once) or those outside the embedding
    lr: Literal["tuned", "rule"] = "tuned"  # the run taken for each size: the lowest final loss among the learning rates run at that size (the bracket's), or only the rule's (`rule_lr`)
    rule_lr: float = 2e-3  # the ladder's base LR (--lr of scaling_ladder.yaml; the peak at width 256, scaled by 256 / width)
    min_tokens: float = 1e8  # validation rows before this many tokens (the warmup and the early transient, where the loss is not yet a power law) are left out of every fit
    iso_min_tokens: float = 3e7  # the IsoFLOP profiles use the rows from this many tokens on: a lower cut than the fit's, since with one pass over the corpus a budget needs the small sizes too
    cuts: tuple[float, ...] = (3e7, 6e7, 1e8, 1.5e8)  # the fit is repeated with the rows from each of these token counts on, to show how much the exponents depend on the cut
    holdout_tokens: float = 2e8  # the token holdout fits the rows up to this many tokens and predicts those beyond
    bootstraps: int = 100  # resamples of the runs for the error bars
    out: Path | None = None  # figures and results.json (default <runs>/gpt/scaling/<group>)


@dataclass
class ShapeConfig:
    """How the loss depends on the shape of the model at equal compute (`scaling.py`): depth x width, MLP size, head size, context length."""

    group: str = "tinystories-shape"  # W&B group of the shape sweeps (sweeps/scaling_shape*.yaml, scaling_ff.yaml, ...)
    flops: tuple[float, ...] = (1e15, 2e15, 2.5e15)  # training budgets to compare the shapes at (read off each run's validation curve)
    min_tokens: float = 1e7  # a run is left out of a budget it reaches with fewer tokens than this (3x the 200-step warmup: the warmup would decide its loss); 1e7 keeps the largest cells of the grid (22.8M parameters is 1.7e7 tokens at 2.5e15 FLOPs), 3e7 drops everything above about 12M
    ladder: str = "tinystories-ladder"  # W&B group of the ladder, to check that its aspect-64 runs agree with the grid's ("" = skip)
    lr_group: str = "tinystories-shape-lr"  # W&B group of the depth x LR runs (sweeps/scaling_shape_lr.yaml), read with the grid's rule-LR runs of the same width ("" = skip)
    out: Path | None = None  # figures and results.json (default <runs>/gpt/scaling/<group>)


@dataclass
class OptimConfig:
    """Optimizer settings (weight decay, AdamW beta2, warmup, tokens per step with the learning rate, gradient clipping, document mask) of one model size, at equal compute (`scaling.py`)."""

    group: str = "tinystories-optim"  # W&B group of the optimizer sweeps (sweeps/scaling_wd.yaml, scaling_beta2.yaml, scaling_warmup.yaml, scaling_batch.yaml, and sweeps/scaling_misc.sh for the rest)
    base: str = "tinystories-shape"  # W&B group of the recipe they are compared with: its run of the same size and seed 0, at the ladder's settings
    flops: tuple[float, ...] = (1e15, 2e15, 2.5e15)  # training budgets to compare at (read off each run's validation curve)
    min_tokens: float = 3e7  # a run is left out of a budget it reaches with fewer tokens than this
    out: Path | None = None  # results.json (default <runs>/gpt/scaling/<group>)


@dataclass
class AnnealConfig:
    """A constant-LR run with EMA against a run that anneals its LR to zero over the same steps (`scaling.py`)."""

    group: str = "tinystories-anneal"  # W&B group of the annealed runs (sweeps/scaling_anneal.yaml)
    ladder: str = "tinystories-ladder"  # W&B group of the constant-LR runs they are compared with
    out: Path | None = None  # figures and results.json (default <runs>/gpt/scaling/<group>)


@dataclass
class SampleConfig:
    """Continue a prompt with a trained model."""

    ckpt: Path  # a pretrain checkpoint
    prompt: str = "Once upon a time"  # "" starts a new document from the end-of-text token
    max_tokens: int = 200  # stops early at end-of-text
    temperature: float = 0.8  # 0 = greedy
    top_k: int | None = 40
    num_samples: int = 1  # generated as one batch
    seed: int = 0
