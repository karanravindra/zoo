"""Benchmark activation checkpointing against a smaller micro-batch, for the two 1e18 / 3e18 run configs.

Run from anywhere (after prepare.py): `uv run projects/pretrain/bench_memory.py --compile`

Both knobs buy memory at a price. A smaller micro-batch (more `grad_accum`) keeps fewer activations alive at once but
launches more, smaller kernels; `--grad-checkpoint` keeps only each block's input and recomputes the rest in the
backward pass, about a third more compute. Each case keeps the sequences per optimizer step fixed (the preset's
`batch_size * grad_accum`), so every case trains the same thing and only the speed and the memory differ.

A case is a (preset, micro-batch, checkpointing) triple. Each runs in a fresh process (so an out-of-memory failure or
CUDA-graph pool of one case cannot affect the next) through the same step as `train.py`: real training batches,
bf16 autocast, cut cross-entropy, EMA, gradient clipping, fused AdamW. A few untimed steps come first (with
`--compile` they include compilation), then `--steps` timed ones. The table has tokens/s, peak memory, and the hours
the preset's whole FLOP budget would take at that speed; the best case per preset is marked. The table is also
written to `runs/pretrain/memory_bench.csv`.

Evals and samples are not measured; they run without gradients at the micro-batch size, so they need far less
memory than a training step.
"""

import copy
import csv
import multiprocessing
import time
from dataclasses import dataclass
from typing import Literal

import torch
import tyro
from llama import Config, build_model
from train import CACHE_ROOT, get_batch, lm_loss, resolve_loss_impl

from zoo.data.text import TokenDataset
from zoo.paths import runs_root
from zoo.train import update_ema
from zoo.utils.device import get_device, seed_everything

GIB = 1024**3
# The run configs from the training commands: the model, the sequences per optimizer step (batch_size * grad_accum), the budget.
COMMON = dict(block_size=2048, doc_mask=True, attn_window=512, attn_global_every=5, ema=True, min_lr_ratio=1.0)
PRESETS = {
    "1e18": dict(dim=768, n_heads=12, n_layers=10, lr=6e-4, flop_budget=1e18, seqs=64),  # ~74M
    "3e18": dict(dim=960, n_heads=15, n_layers=12, lr=5e-4, flop_budget=3e18, seqs=128),  # ~137M
}
FIELDS = ["preset", "micro_batch", "grad_accum", "checkpoint", "status", "tokens_per_s", "step_s", "peak_alloc_gib", "peak_reserved_gib", "budget_hours"]


@dataclass
class Bench:
    presets: tuple[Literal["1e18", "3e18"], ...] = ("1e18", "3e18")  # which run configs to measure
    batch_sizes: tuple[int, ...] = (32, 16, 8, 4)  # micro-batch sizes to try; must divide the preset's sequences per step (64 / 128)
    checkpoint: tuple[bool, ...] = (False, True)  # activation checkpointing settings to try, each at every micro-batch size
    steps: int = 6  # timed optimizer steps per case
    warmup_steps: int = 2  # untimed optimizer steps first (with --compile: compilation and CUDA graph capture)
    compile: bool = False  # model.compile(mode="reduce-overhead", dynamic=False), as the training runs do; FlexAttention is only fast compiled, so pass it
    seed: int = 0


def case_config(preset: str, micro_batch: int, checkpoint: bool, compile: bool, seed: int) -> Config:
    p = dict(PRESETS[preset])
    seqs = p.pop("seqs")
    if seqs % micro_batch:
        raise SystemExit(f"{preset}: micro-batch {micro_batch} does not divide {seqs} sequences per step")
    return Config(**COMMON, **p, batch_size=micro_batch, grad_accum=seqs // micro_batch, grad_checkpoint=checkpoint, compile=compile, seed=seed)


def measure(cfg: Config, steps: int, warmup_steps: int) -> dict:
    """Run `warmup_steps + steps` optimizer steps like `train.py` and time the last `steps`."""
    seed_everything(cfg.seed)
    device = get_device()
    amp_on = cfg.bf16 and device.type == "cuda"
    amp = lambda: torch.autocast(device.type, torch.bfloat16, enabled=amp_on)  # noqa: E731
    impl = resolve_loss_impl(cfg.loss, device, amp_on)
    ds = TokenDataset(CACHE_ROOT, "train", cfg.block_size, device, seed=cfg.seed)
    eot_id = ds.meta["eot_id"]

    model = build_model(cfg).to(device)
    ema = copy.deepcopy(model).requires_grad_(False) if cfg.ema else None
    if cfg.compile:  # the EMA is only ever read by update_ema here, so compiling it would do nothing
        model.compile(mode="reduce-overhead", dynamic=False)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr * 0.01, betas=(0.9, cfg.beta2), fused=device.type == "cuda",
    )  # fmt: skip
    accumulate = cfg.grad_accum > 1
    if accumulate:  # as train.py: persistent .grad buffers, zeroed in place, keep accumulation out of the CUDA graphs
        for p in model.parameters():
            p.grad = torch.zeros_like(p)
    model.train()

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    def step(i: int) -> None:
        opt.zero_grad(set_to_none=not accumulate)
        for _ in range(cfg.grad_accum):
            x, y = get_batch(ds, cfg, eot_id)
            with amp():
                loss = lm_loss(model, x, y, impl, eot_id if cfg.doc_mask else None) / cfg.grad_accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if ema is not None:
            update_ema(ema, model, min(cfg.ema_decay, (1 + i) / (10 + i)))

    for i in range(1, warmup_steps + 1):
        step(i)
    sync()
    t0 = time.perf_counter()
    for i in range(warmup_steps + 1, warmup_steps + steps + 1):
        step(i)
    sync()
    dt = (time.perf_counter() - t0) / steps
    tokens_per_step = cfg.batch_size * cfg.grad_accum * cfg.block_size
    tokens_per_s = tokens_per_step / dt
    budget_tokens = cfg.flop_budget / cfg.train_flops_per_token  # the useful work; recomputation does not count towards it
    cuda = device.type == "cuda"
    return {
        "status": "ok",
        "tokens_per_s": tokens_per_s,
        "step_s": dt,
        "peak_alloc_gib": torch.cuda.max_memory_allocated() / GIB if cuda else float("nan"),
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB if cuda else float("nan"),
        "budget_hours": budget_tokens / tokens_per_s / 3600,
    }


def child(cfg: Config, steps: int, warmup_steps: int, conn) -> None:
    try:
        conn.send(measure(cfg, steps, warmup_steps))
    except Exception as e:  # noqa: BLE001 - every failure is a result, not a crash of the whole benchmark
        lines = str(e).strip().splitlines()
        oom = isinstance(e, torch.OutOfMemoryError) or isinstance(e.__cause__, torch.OutOfMemoryError) or "out of memory" in str(e).lower()
        conn.send({"status": "OOM" if oom else f"error: {type(e).__name__}: {lines[0][:150] if lines else ''}"})


def run_case(cfg: Config, steps: int, warmup_steps: int) -> dict:
    ctx = multiprocessing.get_context("spawn")
    recv, send = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=child, args=(cfg, steps, warmup_steps, send))
    proc.start()
    send.close()
    try:
        result = recv.recv()
    except EOFError:  # the process died without reporting (e.g. a CUDA error that kills it)
        result = {"status": "crashed"}
    proc.join()
    return result


def fmt(row: dict) -> str:
    ok = row["status"] == "ok"
    num = lambda k, f: format(row[k], f) if ok else ""  # noqa: E731
    return (
        f"{row['preset']:>5} {row['micro_batch']:>6}x{row['grad_accum']:<3} {'on' if row['checkpoint'] else 'off':>4} "
        f"{num('tokens_per_s', ',.0f'):>9} {num('step_s', '.2f'):>7} {num('peak_alloc_gib', '.1f'):>8} {num('peak_reserved_gib', '.1f'):>9} "
        f"{num('budget_hours', '.2f'):>8}  {'' if ok else row['status']}{row.get('mark', '')}"
    )


def main(cfg: Bench) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("this benchmark measures GPU memory and speed; no CUDA device found")
    total = torch.cuda.get_device_properties(0).total_memory / GIB
    print(f"{torch.cuda.get_device_name(0)}, {total:.1f} GiB; {cfg.warmup_steps} warmup + {cfg.steps} timed steps per case, compile={cfg.compile}\n")
    header = f"{'run':>5} {'mb x acc':>10} {'ckpt':>4} {'tokens/s':>9} {'step s':>7} {'alloc GiB':>8} {'resv GiB':>9} {'budget h':>8}"
    print(header)
    rows: list[dict] = []
    for preset in cfg.presets:
        for micro_batch in cfg.batch_sizes:
            for checkpoint in cfg.checkpoint:
                run_cfg = case_config(preset, micro_batch, checkpoint, cfg.compile, cfg.seed)
                row = {"preset": preset, "micro_batch": micro_batch, "grad_accum": run_cfg.grad_accum, "checkpoint": checkpoint}
                row |= run_case(run_cfg, cfg.steps, cfg.warmup_steps)
                rows.append(row)
                print(fmt(row), flush=True)
    for preset in cfg.presets:  # mark the fastest working case of each preset
        ok = [r for r in rows if r["preset"] == preset and r["status"] == "ok"]
        if ok:
            ok_best = min(ok, key=lambda r: r["budget_hours"])
            ok_best["mark"] = "  <- fastest"
    print("\n" + header)
    for r in rows:
        print(fmt(r))
    path = runs_root("pretrain") / "memory_bench.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main(tyro.cli(Bench, description=__doc__))
