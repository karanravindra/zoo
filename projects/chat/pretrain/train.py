import argparse
import csv
import json
import math
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from model import GPT, GPTConfig

from data import DATA_DIR, TokenData
from zoo.progress import StepProgress

OUTPUT_DIR = Path(__file__).parent / "outputs"


def choose_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def learning_rate(step, steps, warmup, peak):
    if step <= warmup:
        return peak * step / max(1, warmup)
    progress = (step - warmup) / max(1, steps - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def precision_context(device):
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def make_optimizers(model, lr, weight_decay):
    # Muon operates on hidden-layer matrices, excluding the tied output embedding.
    hidden_weights = [p for p in model.blocks.parameters() if p.ndim == 2]
    hidden_ids = {id(p) for p in hidden_weights}
    remaining = [p for p in model.parameters() if id(p) not in hidden_ids]
    return {
        "muon": torch.optim.Muon(
            hidden_weights,
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.95,
            adjust_lr_fn="match_rms_adamw",
        ),
        "adamw": torch.optim.AdamW(
            [
                {
                    "params": [p for p in remaining if p.ndim >= 2],
                    "weight_decay": weight_decay,
                },
                {"params": [p for p in remaining if p.ndim < 2], "weight_decay": 0.0},
            ],
            lr=lr,
            betas=(0.9, 0.95),
        ),
    }


def initialize_gradients(model):
    """Allocate stable gradient buffers before CUDA graph capture."""
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.zeros_like(parameter)


@torch.inference_mode()
def evaluate(model, data, batch_size, batches, device):
    model.eval()
    # Reuse the same validation windows at each evaluation.
    generator = torch.Generator().manual_seed(1234)
    losses = []
    for _ in range(batches):
        x, y = data.batch("validation", batch_size, generator, device)
        with precision_context(device):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def plot_curves(history, output):
    if not history:
        return
    steps = [row["step"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric, title in zip(
        axes,
        (("train_loss", "val_loss"), ("val_perplexity",), ("lr",)),
        ("Loss", "Validation perplexity", "Learning rate"),
    ):
        for name in metric:
            ax.plot(
                steps, [row[name] for row in history], label=name.removeprefix("val_")
            )
        ax.set(xlabel="Optimizer step", ylabel=title, title=title)
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "curves.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Pretrain a GPT on the prepared data mix"
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    for name in (
        "steps",
        "batch_size",
        "grad_accum",
        "context_length",
        "eval_every",
        "eval_batches",
        "log_every",
        "threads",
    ):
        if getattr(args, name) < 1:
            parser.error(f"{name.replace('_', '-')} must be positive")
    if args.lr <= 0 or args.weight_decay < 0 or args.warmup_steps < 0:
        parser.error("Require lr > 0, weight-decay >= 0, warmup-steps >= 0")
    if not 0 <= args.dropout < 1:
        parser.error("dropout must be in [0, 1)")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    data = TokenData(args.data_dir, args.context_length)
    tokenizer_json = (args.data_dir / "tokenizer.json").read_text()
    config = GPTConfig(
        vocab_size=data.metadata["vocab_size"],
        context_length=args.context_length,
        dropout=args.dropout,
    )
    model = GPT(config).to(device)
    optimizers = make_optimizers(model, args.lr, args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed)
    start_step, tokens_seen, best_loss = 0, 0, float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint["config"] != asdict(config):
            parser.error(
                "Checkpoint model configuration does not match training arguments"
            )
        if (
            checkpoint["tokenizer"] != tokenizer_json
            or checkpoint["data"] != data.metadata
        ):
            parser.error(
                "Checkpoint tokenizer or dataset metadata does not match prepared data"
            )
        if set(checkpoint.get("optimizers", {})) != set(optimizers):
            parser.error(
                "Resuming requires a Muon + AdamW checkpoint; start a new run for older checkpoints"
            )
        model.load_state_dict(checkpoint["model"])
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(checkpoint["optimizers"][name])
        generator.set_state(checkpoint["sampler_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda" and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        if device.type == "mps" and checkpoint["mps_rng"] is not None:
            torch.mps.set_rng_state(checkpoint["mps_rng"])
        start_step, best_loss = checkpoint["step"], checkpoint["best_loss"]
        tokens_seen = checkpoint["tokens_seen"]
    if start_step >= args.steps:
        parser.error("steps must exceed the checkpoint step")
    compiled_model = torch.compile(model, mode="reduce-overhead")
    initialize_gradients(model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parameters = sum(p.numel() for p in model.parameters())
    tokens_per_step = args.batch_size * args.grad_accum * args.context_length
    print(
        f"Parameters: {parameters:,}; device: {device}; tokens/step: {tokens_per_step:,}"
    )
    metrics_path = args.output_dir / "metrics.csv"
    append = args.resume is not None and metrics_path.exists()
    history = []
    if append:
        with metrics_path.open(newline="") as file:
            for row in csv.DictReader(file):
                history.append(
                    {
                        "step": int(row["step"]),
                        "train_loss": float(row["train_loss"]),
                        "val_loss": float(row["val_loss"]),
                        "val_perplexity": float(row["val_perplexity"]),
                        "lr": float(row["lr"]),
                    }
                )
    with metrics_path.open("a" if append else "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "step",
                "tokens_seen",
                "train_loss",
                "val_loss",
                "val_perplexity",
                "lr",
            ],
        )
        if not append:
            writer.writeheader()
        model.train()
        progress = StepProgress(start_step)
        for step in range(start_step + 1, args.steps + 1):
            lr = learning_rate(
                step, args.steps, min(args.warmup_steps, args.steps), args.lr
            )
            for optimizer in optimizers.values():
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=False)
            train_loss = 0.0
            for _ in range(args.grad_accum):
                x, y = data.batch("train", args.batch_size, generator, device)
                with precision_context(device):
                    _, loss = compiled_model(x, y)
                train_loss += loss.detach().item() / args.grad_accum
                (loss / args.grad_accum).backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            for optimizer in optimizers.values():
                optimizer.step()
            tokens_seen += tokens_per_step
            progress.complete(step)
            if step % args.log_every == 0 or step == 1:
                print(
                    f"Step {step}/{args.steps}: loss={train_loss:.4f} lr={lr:.2e} "
                    f"{progress.status(step, args.steps)}"
                )
            if step % args.eval_every == 0 or step == args.steps:
                val_loss = evaluate(
                    compiled_model, data, args.batch_size, args.eval_batches, device
                )
                perplexity = math.exp(min(val_loss, 80))
                improved = val_loss < best_loss
                best_loss = min(best_loss, val_loss)
                row = {
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_perplexity": perplexity,
                    "lr": lr,
                }
                writer.writerow(row)
                file.flush()
                history.append(row)
                plot_curves(history, args.output_dir)
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizers": {
                        name: optimizer.state_dict()
                        for name, optimizer in optimizers.items()
                    },
                    "config": asdict(config),
                    "tokenizer": tokenizer_json,
                    "data": data.metadata,
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "best_loss": best_loss,
                    "sampler_rng": generator.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all()
                    if device.type == "cuda"
                    else None,
                    "mps_rng": torch.mps.get_rng_state()
                    if device.type == "mps"
                    else None,
                    "args": {
                        k: str(v) if isinstance(v, Path) else v
                        for k, v in vars(args).items()
                    },
                }
                # Replace only after serialization completes.
                temporary = args.output_dir / "checkpoint.tmp"
                torch.save(checkpoint, temporary)
                temporary.replace(args.output_dir / "checkpoint.pt")
                if improved:
                    torch.save(checkpoint, temporary)
                    temporary.replace(args.output_dir / "best.pt")
                summary = dict(
                    parameters=parameters,
                    device=str(device),
                    **row,
                    best_val_loss=best_loss,
                    config=asdict(config),
                )
                (args.output_dir / "summary.json").write_text(
                    json.dumps(summary, indent=2) + "\n"
                )
                print(
                    f"Validation: loss={val_loss:.4f} perplexity={perplexity:.2f}; "
                    f"checkpoint saved at step {step}"
                )


if __name__ == "__main__":
    main()
