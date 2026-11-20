"""Train a small Llama-style decoder on the story mix from prepare.py (step-based, next-token prediction).

Run from anywhere (after prepare.py): `uv run projects/pretrain/train.py`

Each run gets its own directory, `<repo>/runs/pretrain/<name>/`, named after the model shape and budget (e.g.
`d256-l6-ctx512-3e16`, `d384-l6-ctx512-20k`, `d256-l6-ctx512-600s`; `--run-name` overrides it, and `-2`, `-3`, ...
is appended rather than reusing an existing directory), containing:
  config.json    the full config plus git commit, device and the cut-cross-entropy implementation used
  metrics.csv    a row every `log_every` steps and at every eval: train loss and bits per byte (mean since the
                 previous row), lr, gradient norm (mean, max, fraction of steps clipped), steps/s and training
                 tokens/s, GPU utilisation / memory / power / temperature, and the cumulative training tokens,
                 FLOPs and seconds. Filled on some rows only: weight norm, update-to-weight ratio and gradient
                 direction (every `stats_every` steps), val loss and bits per byte (eval rows), and the same on
                 each source's held-out stories as `val_<source>_loss` / `val_<source>_bpb` (eval rows; `val_loss`
                 itself is the `validation/` set, one source, which also gets its own `val_<source>_*` columns)
  layers.csv     every `stats_every` steps, per layer group (emb, L<i>.attn, L<i>.mlp, norms): weight norm,
                 update-to-weight ratio, gradient norm, gradient cosine with the previous step
  best.pt        checkpoint with the lowest validation loss (includes its config)
  last.pt        checkpoint from the latest eval (the final step, when the run finishes)
  results.json   best step with its val loss / bpb, final val loss / bpb (and the final per-source ones), mean tokens/s
  curves.png     train / val loss;  bpb.png  train / val bits per byte;  grads.png  gradient norm and update ratio;
                 val_sources.png  val bits per byte on each source
  samples.txt    a text sample every `sample_every` steps (and at the final step), from a fixed seed so steps are comparable

Bits per byte = loss (nats/token) / ln 2 * tokens / UTF-8 bytes of the split, with the split's token and byte counts
from prepare.py's meta.json (see zoo.metrics.bits_per_byte). `best.pt` is picked on `val_loss`, the `validation/` set
alone; the per-source columns only report. They use `--val-source-batches` batches each, laid out like training. tokens_per_s counts training tokens only: eval, sampling
and checkpointing are excluded from the window, and the clock is read after a device sync. With --compile the first
window also contains compilation.

Documents: by default a window is a random slice of the concatenated stories with plain causal attention. `--doc-mask`
keeps attention inside each story. `--doc-align pack` starts every window at a story and lets it run on through the
following ones (no padding); `--doc-align pad` puts one story in each row and pads it, leaving the padding out of the
loss (padded positions still cost compute and count in `tokens` and `flops`). `val_loss` is measured the same way the
run trains, so it is not comparable between these modes; `val_doc_loss` / `val_doc_bpb` always score whole stories
from their start, each token from its own story only, and are the columns to compare such runs on.

Memory: the saved activations grow with batch_size * block_size * dim, so a wide model with a long context may not
fit at the default batch size. With `--auto-batch` (the default) a run that hits a CUDA out-of-memory error before its
first checkpoint is restarted in a fresh process with the micro-batch halved and `grad_accum` doubled, as often as
needed; tokens per optimizer step, and so the step count of a FLOP budget, do not change. `--grad-checkpoint`
recomputes each block's activations in the backward pass instead (less memory, about a third more compute);
`bench_memory.py` measures which of the two is faster for a config.

`flops` is an analytic count of training FLOPs (see `Config.train_flops_per_token` in llama.py), so runs of different
model sizes can be compared at equal compute. `--flop-budget 3e16` trains for the number of steps that fit that budget
(a fixed step count, so the LR schedule is the usual one); `--time-budget-s 600` trains until that much training time
has passed, with the cosine following elapsed time (with --compile, compilation counts against it).
"""

import copy
import itertools
import json
import math
import multiprocessing
import shutil
import sys

import gigatoken as gt
import torch
import torch.nn.functional as F
import tyro
from cut_cross_entropy import linear_cross_entropy
from llama import Config, build_model, param_group, save_checkpoint
from torchinfo import summary
from tqdm import tqdm

from zoo.data.text import TokenDataset
from zoo.metrics import bits_per_byte
from zoo.monitor import GPU_FIELDS, GpuMonitor
from zoo.paths import DATA_ROOT
from zoo.plot import plot_curves
from zoo.runs import MetricsLogger, new_run_dir
from zoo.train import STAT_FIELDS, ParamStats, Throughput, update_ema
from zoo.utils.device import get_device, seed_everything

CACHE_ROOT = DATA_ROOT / "cache" / "pretrain"
METRIC_FIELDS = [
    "step", "tokens", "flops", "elapsed_s", "lr", "train_loss", "train_bpb", *STAT_FIELDS, "steps_per_s", "tokens_per_s",
    *GPU_FIELDS, "val_loss", "val_bpb", "val_doc_loss", "val_doc_bpb",
]  # fmt: skip
SAMPLE_SEED = 1234
EVAL_SEED = 4321  # val batches are the same at every eval


def source_columns(sources: list[str]) -> list[str]:
    """metrics.csv columns for per-source validation: `val_<source>_loss` and `val_<source>_bpb` for each source."""
    return [f"val_{s}_{m}" for s in sources for m in ("loss", "bpb")]


def resolve_loss_impl(loss: str, device: torch.device, amp: bool) -> str:
    """auto: the Triton CCE kernel on cuda under bf16 autocast, the torch.compile implementation elsewhere."""
    if loss != "auto":
        return loss
    return "cce" if device.type == "cuda" and amp else "torch_compile"


def get_batch(ds: TokenDataset, cfg: Config, eot_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A batch laid out as `cfg.doc_align` says: random windows, or windows that start at a story."""
    if cfg.doc_align == "off":
        return ds.batch(cfg.batch_size)
    return ds.doc_batch(cfg.batch_size, eot_id, pad=cfg.doc_align == "pad")


def lm_loss(model, x: torch.Tensor, y: torch.Tensor, impl: str, mask_eot: int | None = None) -> torch.Tensor:
    """Mean next-token cross-entropy of `model` on (x, y) with y already shifted by the batcher; targets of -100
    (padding) are skipped. With `mask_eot` (the end-of-text id), attention is kept inside each story.

    The model returns hidden states and the tied embedding is the classifier, so with a cut cross-entropy `impl`
    the (batch, block, vocab) logits never exist."""
    h = model(x, return_hidden=True, eot_id=mask_eot)
    if impl == "torch":
        return F.cross_entropy(F.linear(h, model.classifier_weight).float().flatten(0, 1), y.flatten())
    return linear_cross_entropy(h, model.classifier_weight, y, impl=impl)


def require_bytes(ds: TokenDataset, split: str) -> tuple[int, int]:
    """(tokens, UTF-8 bytes) of a split, which bits per byte needs; fails clearly on caches from before it was recorded."""
    if "num_bytes" not in ds.meta:
        raise SystemExit(f"{CACHE_ROOT / split / 'meta.json'} has no byte count; re-run `uv run projects/pretrain/prepare.py --force`.")
    return ds.meta["length"], ds.meta["num_bytes"]


@torch.no_grad()
def evaluate(model, ds: TokenDataset, cfg: Config, impl: str, amp, eot_id: int, per_story: bool = False, batches: int | None = None) -> float:
    """Mean loss per token over `batches` (default `eval_batches`) fixed val batches.

    By default the batches are laid out, and attention masked, exactly as in training (`doc_align`, `doc_mask`),
    so the number means the same thing as the training loss. With `per_story` every row is one story from its
    start token, padded: each token is predicted from its own story only. That number does not depend on the
    document options, so it is the one to compare runs trained with different ones."""
    model.eval()
    ds.reset(EVAL_SEED + per_story)
    total = torch.zeros((), device=ds.device)
    count = torch.zeros((), device=ds.device)
    for _ in range(cfg.eval_batches if batches is None else batches):
        if per_story:
            x, y = ds.doc_batch(cfg.batch_size, eot_id, pad=True)
            mask_eot = None  # one story per row: nothing to mask
        else:
            x, y = get_batch(ds, cfg, eot_id)
            mask_eot = eot_id if cfg.doc_mask else None
        n = (y != -100).sum()
        with amp():
            total += lm_loss(model, x, y, impl, mask_eot).detach() * n  # consumed before the next call (CUDA graphs reuse buffers)
        count += n
    model.train()
    return (total / count).item()


def load_val_sources(cfg: Config, device: torch.device) -> dict[str, tuple[TokenDataset, int, int]]:
    """The held-out stories of each source in the mix, from prepare.py's `validation_<source>/` dirs, as
    {source: (dataset, tokens, UTF-8 bytes)}. Empty if `val_source_batches` is 0 or the cache has none."""
    if cfg.val_source_batches <= 0:
        return {}
    sets = {}
    for meta in sorted(CACHE_ROOT.glob("validation_*/meta.json")):
        split = meta.parent.name
        ds = TokenDataset(CACHE_ROOT, split, cfg.block_size, device, seed=EVAL_SEED)
        sets[split.removeprefix("validation_")] = (ds, *require_bytes(ds, split))
    return sets


def main_source(val_ds: TokenDataset) -> str | None:
    """The one source `validation/` holds, which prepare.py records in meta.json (None for caches without it)."""
    names = list(val_ds.meta.get("sources", {}))
    return names[0] if len(names) == 1 else None


def source_scores(model, sets: dict[str, tuple[TokenDataset, int, int]], cfg: Config, impl: str, amp, eot_id: int) -> dict[str, float]:
    """`val_<source>_loss` and `val_<source>_bpb` on every source's held-out stories, laid out like training."""
    scores = {}
    for name, (ds, n_tokens, n_bytes) in sets.items():
        loss = evaluate(model, ds, cfg, impl, amp, eot_id, batches=cfg.val_source_batches)
        scores |= {f"val_{name}_loss": loss, f"val_{name}_bpb": bits_per_byte(loss, n_tokens, n_bytes)}
    return scores


def make_sample(model, tok: gt.Tokenizer, cfg: Config, eot_id: int, device: torch.device, amp) -> str:
    ids = [int(t) for t in tok.encode(cfg.sample_prompt)]
    bos = cfg.doc_align != "off"  # a story-aligned model has only ever seen stories that begin with the end-of-text token
    prompt = torch.tensor([[eot_id] * bos + ids], device=device)
    g = torch.Generator(device=device).manual_seed(SAMPLE_SEED)
    with amp():
        out = model.generate(prompt, cfg.sample_tokens, cfg.sample_temperature, cfg.sample_top_k, eos_id=eot_id, generator=g)
    return tok.decode(out[0].tolist()[bos:]).decode("utf-8", errors="replace")


def train(cfg: Config, progress: dict):
    """One training run. `progress` is filled in as it goes (run_dir, checkpoints saved), for `run`'s OOM handling."""
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    device = get_device()
    print(f"Using device: {device}")
    amp_on = cfg.bf16 and device.type == "cuda"
    amp = lambda: torch.autocast(device.type, torch.bfloat16, enabled=amp_on)  # noqa: E731
    impl = resolve_loss_impl(cfg.loss, device, amp_on)
    print(f"Loss: {impl}")

    train_ds = TokenDataset(CACHE_ROOT, "train", cfg.block_size, device, seed=cfg.seed)
    val_ds = TokenDataset(CACHE_ROOT, "validation", cfg.block_size, device, seed=EVAL_SEED)
    if train_ds.meta["vocab_size"] != cfg.vocab_size:
        raise SystemExit(f"cache vocab is {train_ds.meta['vocab_size']} but --vocab-size is {cfg.vocab_size}")
    train_tokens, train_bytes = require_bytes(train_ds, "train")
    val_tokens, val_bytes = require_bytes(val_ds, "validation")
    val_sets = load_val_sources(cfg, device)  # held-out stories of the other sources in the mix
    primary = main_source(val_ds) if val_sets else None  # the source `validation/` holds: val_loss is its score
    val_sets.pop(primary, None)  # a stale validation_<primary>/ left from an earlier mix
    sources = ([primary] if primary else []) + list(val_sets)
    source_cols = source_columns(sources)
    if sources:
        print(f"per-source validation: {', '.join(sources)} ({cfg.val_source_batches} batches each)")
    tok = gt.Tokenizer(CACHE_ROOT / "tokenizer.json")
    eot_id = train_ds.meta["eot_id"]

    tokens_per_step = cfg.batch_size * cfg.block_size * cfg.grad_accum
    flops_per_step = cfg.train_flops_per_token * tokens_per_step
    time_mode = cfg.time_budget_s > 0
    if cfg.flop_budget > 0 and not time_mode:
        cfg.max_steps = max(1, round(cfg.flop_budget / flops_per_step))  # recorded in config.json and the checkpoints
    budget = f"{cfg.time_budget_s:,.0f} s of training" if time_mode else f"{cfg.max_steps:,} steps = {cfg.max_steps * flops_per_step:.3g} FLOPs"
    print(f"{flops_per_step:.3g} training FLOPs per step ({cfg.train_flops_per_token:.3g} per token); budget: {budget}")

    name = cfg.run_name or cfg.default_run_name
    run_dir = new_run_dir("pretrain", cfg.to_dict(), device, dev=cfg.dev, name=name, loss_impl=impl, train_flops_per_token=cfg.train_flops_per_token)
    progress["run_dir"] = run_dir

    model = build_model(cfg).to(device)
    summary(model, input_size=(1, cfg.block_size),  # batch 1: only the layer table is wanted, and uncompiled FlexAttention falls back to dense scores
        dtypes=[torch.long], device=device, depth=2)
    ema = copy.deepcopy(model).requires_grad_(False) if cfg.ema else None
    if cfg.compile:
        model.compile(mode="reduce-overhead", dynamic=False)  # in place, so state_dict keys and checkpoints are unchanged
        if ema is not None:
            ema.compile(mode="reduce-overhead", dynamic=False)
    scored = ema if ema is not None else model  # the weights that are evaluated, sampled and saved

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr, betas=(0.9, cfg.beta2), fused=device.type == "cuda",
    )

    def lr_at(taken: int, elapsed: float) -> float:
        """Linear warmup over `warmup_steps`, then cosine from lr down to lr * min_lr_ratio over the budget: the
        remaining steps, or with a time budget the fraction of it used. `taken` = optimizer steps taken so far."""
        if taken < cfg.warmup_steps:
            return cfg.lr * (taken + 1) / cfg.warmup_steps
        if time_mode:
            progress = elapsed / cfg.time_budget_s
        else:
            progress = (taken - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
        return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))

    # Gradient accumulation under compile: a .grad tensor first created by a CUDA-graph backward is overwritten by
    # the next micro-batch's run and autograd refuses to add to it. Giving every parameter a .grad buffer up front,
    # and zeroing it in place rather than dropping it, keeps the buffers outside the graphs.
    accumulate = cfg.grad_accum > 1
    if accumulate:
        for p in model.parameters():
            p.grad = torch.zeros_like(p)

    best_val, best_step, best_val_bpb = float("inf"), 0, float("nan")
    samples = (run_dir / "samples.txt").open("w")
    model.train()
    loss_sum = torch.zeros((), device=device)  # on device; synced only at log / eval steps
    n_window = 0  # steps since the last metrics row
    tp = Throughput()  # window since the previous metrics row, and the whole run; paused during eval and sampling
    stats = ParamStats(model, every=cfg.stats_every, group=param_group)
    gpu = GpuMonitor(device)
    layers = MetricsLogger(run_dir / "layers.csv", ["step", *stats.group_fields()]) if cfg.stats_every > 0 else None
    eval_row: dict = {}
    # With a time budget the number of steps is not known up front: train until the training clock passes it.
    pbar = tqdm(itertools.count(1) if time_mode else range(1, cfg.max_steps + 1), desc="train", total=None if time_mode else cfg.max_steps)
    with MetricsLogger(run_dir / "metrics.csv", [*METRIC_FIELDS, *source_cols]) as logger:
        rows = logger.rows
        for step in pbar:
            lr = lr_at(step - 1, tp.elapsed())
            for group in opt.param_groups:
                group["lr"] = lr
            opt.zero_grad(set_to_none=not accumulate)
            for _ in range(cfg.grad_accum):
                x, y = get_batch(train_ds, cfg, eot_id)
                with amp():
                    loss = lm_loss(model, x, y, impl, eot_id if cfg.doc_mask else None) / cfg.grad_accum
                loss.backward()
                loss_sum += loss.detach()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip) if cfg.grad_clip > 0 else None
            stats.grads(step, norm, cfg.grad_clip)
            opt.step()
            stats.stepped(step)
            if ema is not None:
                update_ema(ema, model, min(cfg.ema_decay, (1 + step) / (10 + step)))
            n_window += 1

            # The time check reads the clock without a device sync, so it can run a few queued steps late.
            last = tp.elapsed() >= cfg.time_budget_s if time_mode else step == cfg.max_steps
            do_eval = step % cfg.eval_every == 0 or last
            do_sample = cfg.sample_every > 0 and (step % cfg.sample_every == 0 or last)
            do_log = step % cfg.log_every == 0 or do_eval  # a metrics.csv row: every log_every steps, and at every eval
            if do_log:
                train_loss = loss_sum.item() / n_window  # syncs the device, so the clock read below is real
                tokens_per_s = tp.tick(step, tokens_per_step)
                overall, per_layer = stats.read()
                row = {
                    "step": step,
                    "tokens": step * tokens_per_step,
                    "flops": step * flops_per_step,
                    "elapsed_s": tp.elapsed(),
                    "lr": lr,
                    "train_loss": train_loss,  # mean since the previous row
                    "train_bpb": bits_per_byte(train_loss, train_tokens, train_bytes),
                    **overall,
                    "steps_per_s": tokens_per_s / tokens_per_step,
                    "tokens_per_s": tokens_per_s,
                    **gpu.read(),
                }
                if per_layer and layers:
                    layers.log({"step": step, **per_layer})
                loss_sum.zero_()
                n_window = 0
                pbar.set_postfix(
                    loss=f"{train_loss:.4f}", lr=f"{lr:.2e}", gnorm=f"{row['grad_norm']:.2f}", tok_s=f"{tokens_per_s:,.0f}",
                    gpu=f"{row.get('gpu_util', 0):.0f}%", mem=f"{row.get('gpu_mem_gb', row.get('gpu_alloc_gb', 0)):.1f}G",
                )
            elif do_sample:
                loss_sum.item()  # sync, so the pause below starts after all queued training work
            if do_eval or do_sample:
                tp.pause()
            if do_eval:
                val_loss = evaluate(scored, val_ds, cfg, impl, amp, eot_id)
                val_doc_loss = evaluate(scored, val_ds, cfg, impl, amp, eot_id, per_story=True)
                row |= {
                    "val_loss": val_loss,
                    "val_bpb": bits_per_byte(val_loss, val_tokens, val_bytes),
                    "val_doc_loss": val_doc_loss,
                    "val_doc_bpb": bits_per_byte(val_doc_loss, val_tokens, val_bytes),
                }
                if sources:
                    row |= source_scores(scored, val_sets, cfg, impl, amp, eot_id)
                    if primary:  # `validation/` is that source's held-out set, already scored as val_loss
                        row |= {f"val_{primary}_loss": val_loss, f"val_{primary}_bpb": row["val_bpb"]}
                eval_row = row
                tqdm.write(  # not print: that would land on the progress bar's line
                    f"step {step}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_bpb={row['val_bpb']:.4f} "
                    f"val_doc_loss={val_doc_loss:.4f} grad_norm={row['grad_norm']:.3f} ({row['steps_per_s']:.1f} steps/s, "
                    f"{tokens_per_s:,.0f} tokens/s, gpu {row.get('gpu_util', 0):.0f}% {row.get('gpu_mem_gb', row.get('gpu_alloc_gb', 0)):.1f} GiB)"
                )
                if sources:
                    tqdm.write("  val bpb by source: " + "  ".join(f"{s}={row[f'val_{s}_bpb']:.3f}" for s in sources))
                if val_loss < best_val:
                    best_val, best_step, best_val_bpb = val_loss, step, row["val_bpb"]
                    save_checkpoint(run_dir / "best.pt", scored, cfg, step, row)
                save_checkpoint(run_dir / "last.pt", scored, cfg, step, row)
                progress["checkpoints"] = progress.get("checkpoints", 0) + 1
            if do_log:
                logger.log(row)
            if do_sample:
                text = make_sample(scored, tok, cfg, eot_id, device, amp)
                samples.write(f"=== step {step} ===\n{text}\n\n")
                samples.flush()
                tqdm.write(f"--- sample (step {step}) ---\n{text}\n--------------")
            if do_eval or do_sample:
                gpu.read()  # drop the eval / sampling stretch, so the next row's GPU numbers are training only
                tp.resume()
            if last:
                break
    pbar.close()
    samples.close()
    gpu.close()
    if layers:
        layers.close()

    mean_tokens_per_s = tp.rate(step, tokens_per_step)  # training time only (eval etc. paused out)
    plot_curves(rows, ["train_loss", "val_loss"], run_dir / "curves.png", ylabel="loss (nats/token)", x="step")
    plot_curves(rows, ["train_bpb", "val_bpb"], run_dir / "bpb.png", ylabel="bits per byte", x="step")
    plot_curves(rows, ["grad_norm", "update_ratio"], run_dir / "grads.png", ylabel="gradient norm / update-to-weight ratio", log=True, x="step")
    if sources:
        plot_curves(rows, [f"val_{s}_bpb" for s in sources], run_dir / "val_sources.png", ylabel="val bits per byte", x="step")
    results = {
        "best_step": best_step,
        "best_val_loss": best_val,
        "best_val_bpb": best_val_bpb,
        "final_val_loss": eval_row["val_loss"],
        "final_val_bpb": eval_row["val_bpb"],
        "final_val_doc_loss": eval_row["val_doc_loss"],
        "final_val_doc_bpb": eval_row["val_doc_bpb"],
        **{f"final_{c}": eval_row[c] for c in source_cols},
        "final_grad_norm": eval_row["grad_norm"],
        "total_steps": step,
        "total_flops": step * flops_per_step,
        "mean_tokens_per_s": mean_tokens_per_s,
        "train_time_s": tp.elapsed(),
        **gpu.summary(),
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"best step {best_step}: val_loss={best_val:.4f} val_bpb={best_val_bpb:.4f} ({mean_tokens_per_s:,.0f} tokens/s mean)")
    print(f"Outputs in {run_dir}")


OOM_EXIT = 75  # exit code of a run that ran out of GPU memory before its first checkpoint


def split_batch(cfg: Config) -> bool:
    """Split the micro-batch by its smallest factor and accumulate that many more, in place; tokens per optimizer
    step (and so the FLOPs per step and the step count of a FLOP budget) do not change. False at micro-batch 1."""
    k = next((k for k in range(2, cfg.batch_size + 1) if cfg.batch_size % k == 0), None)
    if k is None:
        return False
    cfg.batch_size, cfg.grad_accum = cfg.batch_size // k, cfg.grad_accum * k
    return True


def run(cfg: Config) -> None:
    """`train`, exiting with OOM_EXIT if the GPU runs out of memory before the first checkpoint (and removing the
    run directory that attempt created). A later out-of-memory error is raised as usual."""
    progress: dict = {}
    try:
        train(cfg, progress)
    except (torch.OutOfMemoryError, RuntimeError) as e:
        oom = e if isinstance(e, torch.OutOfMemoryError) else e.__cause__  # torchinfo wraps errors from the forward pass
        if not isinstance(oom, torch.OutOfMemoryError):
            raise
        e = oom
        if not cfg.auto_batch or progress.get("checkpoints"):
            raise
        if progress.get("run_dir") and not cfg.dev:  # a dev run's directory is removed on exit anyway
            shutil.rmtree(progress["run_dir"], ignore_errors=True)
        print(f"\nOut of GPU memory at batch_size={cfg.batch_size}, grad_accum={cfg.grad_accum}: {str(e).split('. ')[0]}.", flush=True)
        sys.exit(OOM_EXIT)


def main(cfg: Config) -> None:
    """Run training in a child process, so that an out-of-memory failure leaves nothing behind on the GPU (compiled
    graphs keep their memory pools for the life of a process); on one, split the micro-batch and start again."""
    if not cfg.auto_batch:
        return train(cfg, {})
    ctx = multiprocessing.get_context("spawn")
    while True:
        child = ctx.Process(target=run, args=(cfg,))
        child.start()
        child.join()
        if child.exitcode != OOM_EXIT:
            sys.exit(child.exitcode)
        if not split_batch(cfg):
            sys.exit("Out of GPU memory even at batch_size=1; reduce block_size or the model size.")
        print(f"Restarting with --batch-size {cfg.batch_size} --grad-accum {cfg.grad_accum} (same tokens per step).\n", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Config, description=__doc__))
