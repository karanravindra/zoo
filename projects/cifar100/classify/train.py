"""Train a compact ConvNeXt on CIFAR-100 and evaluate its best checkpoint."""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
from model import ConvNeXt
from reports import write_reports
from torch import nn

from zoo.data import make_loader
from zoo.data.cifar100 import add_data_arguments, config_from_args, make_splits
from zoo.progress import StepProgress


def batch_metrics(logits, labels):
    top = logits.detach().topk(5, dim=1).indices
    return (
        (top[:, 0] == labels).sum().item(),
        (top == labels[:, None]).any(dim=1).sum().item(),
    )


def average_metrics(loss, top1, top5, count):
    return {
        "loss": loss / count,
        "top1": top1 / count,
        "top5": top5 / count,
        "samples": count,
    }


@torch.inference_mode()
def evaluate(model, loader, device, max_batches=0):
    model.eval()
    loss, top1, top5, count = 0.0, 0, 0, 0
    targets, predictions = [], []
    for index, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
        correct1, correct5 = batch_metrics(logits, labels)
        top1 += correct1
        top5 += correct5
        count += len(labels)
        targets.append(labels.cpu())
        predictions.append(logits.argmax(1).cpu())
        if max_batches and index + 1 >= max_batches:
            break
    return (
        average_metrics(loss, top1, top5, count),
        torch.cat(targets),
        torch.cat(predictions),
    )


def make_scheduler(optimizer, total_steps, warmup_steps):
    def multiplier(step):
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, max(0.0, progress))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def save_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--widths", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--depths", nargs="+", type=int, default=[2, 2, 4])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent / "outputs"
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Restore training configuration and epoch-boundary state",
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="Evaluate the best model from --resume"
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help="Smoke runs: batches per epoch; 0 means all",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=0,
        help="Smoke runs: batches per evaluation; 0 means all",
    )
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        default=0,
        help="Save and stop without test evaluation",
    )
    add_data_arguments(parser)
    args = parser.parse_args()
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        # Restore the optimization horizon as well as architecture and data settings.
        runtime = {
            "device",
            "threads",
            "log_every",
            "output_dir",
            "resume",
            "eval_only",
            "stop_after_epoch",
            "num_workers",
            "persistent_workers",
            "prefetch_factor",
            "pin_memory",
        }
        for key, value in checkpoint["config"].items():
            if key not in runtime:
                setattr(
                    args,
                    key,
                    Path(value) if key in ("data_dir", "cache_dir") else value,
                )
    if min(args.epochs, args.batch_size, args.lr, args.threads, args.log_every) <= 0:
        parser.error("epochs, batch-size, lr, threads, and log-every must be positive")
    if (
        min(
            args.weight_decay,
            args.warmup_epochs,
            args.max_train_batches,
            args.max_eval_batches,
            args.stop_after_epoch,
        )
        < 0
    ):
        parser.error("weight-decay, warmup, and batch/epoch limits must be nonnegative")
    if args.eval_only and checkpoint is None:
        parser.error("--eval-only requires --resume")
    if checkpoint is not None and not args.eval_only and "optimizer" not in checkpoint:
        parser.error(
            "Use checkpoint.pt for training resume; best.pt supports --eval-only"
        )
    if (
        len(args.widths) != len(args.depths)
        or not 1 <= len(args.widths) <= 5
        or min(*args.widths, *args.depths) < 1
    ):
        parser.error(
            "widths and depths must have matching lengths (1–5) and positive values"
        )
    try:
        config_from_args(args)
    except ValueError as error:
        parser.error(str(error))
    return args, checkpoint


def main():
    args, restored = parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    data_config = config_from_args(args)
    train_data, val_data, test_data = make_splits(data_config, args.seed)
    classes = test_data.classes
    train_loader = make_loader(
        train_data, data_config, args.batch_size, shuffle=True, seed=args.seed
    )
    val_loader = make_loader(val_data, data_config, args.batch_size)
    test_loader = make_loader(test_data, data_config, args.batch_size)
    model = ConvNeXt(widths=args.widths, depths=args.depths).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Device={device}, parameters={parameters:,}, train={len(train_data)}, val={len(val_data)}, test={len(test_data)}",
        flush=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    steps_per_epoch = min(
        len(train_loader), args.max_train_batches or len(train_loader)
    )
    total_steps = args.epochs * steps_per_epoch
    scheduler = make_scheduler(
        optimizer,
        total_steps,
        min(args.warmup_epochs * steps_per_epoch, total_steps - 1),
    )
    history, start_epoch, step, elapsed_before = [], 0, 0, 0.0
    best_top1, best_epoch, best_model = -1.0, 0, None
    if restored:
        if restored["classes"] != classes:
            raise ValueError("Checkpoint class names differ from the dataset")
        model.load_state_dict(restored["model"])
        if not args.eval_only:
            optimizer.load_state_dict(restored["optimizer"])
            scheduler.load_state_dict(restored["scheduler"])
        history, start_epoch, step = (
            restored["history"],
            restored["epoch"],
            restored["step"],
        )
        elapsed_before = restored["elapsed_seconds"]
        best_top1, best_epoch, best_model = (
            restored["best_top1"],
            restored["best_epoch"],
            restored["best_model"],
        )
    start = time.perf_counter()
    progress = StepProgress(step)

    def record(epoch, totals, lr):
        validation, _, _ = evaluate(model, val_loader, device, args.max_eval_batches)
        train = average_metrics(*totals)
        row = {
            "epoch": epoch,
            "step": step,
            "elapsed_seconds": elapsed_before + time.perf_counter() - start,
            "lr": lr,
        }
        for split, metrics in (("train", train), ("val", validation)):
            row.update({f"{split}_{key}": value for key, value in metrics.items()})
        history.append(row)
        with (output / "metrics.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(
            f"Epoch {epoch}/{args.epochs}, step {step}: train_loss={train['loss']:.4f} "
            f"train_top1={train['top1']:.2%} train_top5={train['top5']:.2%} "
            f"val_loss={validation['loss']:.4f} val_top1={validation['top1']:.2%} "
            f"val_top5={validation['top5']:.2%} elapsed={row['elapsed_seconds']:.1f}s "
            f"{progress.status(step, total_steps)}",
            flush=True,
        )
        return validation

    if not args.eval_only:
        for epoch in range(start_epoch + 1, args.epochs + 1):
            # Epoch-local seeds make zero-worker continuation reproducible.
            torch.manual_seed(args.seed + epoch)
            train_loader.generator.manual_seed(args.seed + epoch)
            totals = [0.0, 0, 0, 0]
            for batch, (images, labels) in enumerate(train_loader, 1):
                model.train()
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = nn.functional.cross_entropy(logits, labels)
                loss.backward()
                lr = optimizer.param_groups[0]["lr"]
                optimizer.step()
                scheduler.step()
                step += 1
                progress.complete(step)
                top1, top5 = batch_metrics(logits, labels)
                totals[0] += loss.item() * len(labels)
                totals[1] += top1
                totals[2] += top5
                totals[3] += len(labels)
                last_batch = batch == steps_per_epoch
                if step % args.log_every == 0 or last_batch:
                    validation = record(epoch, totals, lr)
                    if validation["top1"] > best_top1:
                        best_top1, best_epoch = validation["top1"], epoch
                        best_model = {
                            key: value.detach().cpu().clone()
                            for key, value in model.state_dict().items()
                        }
                if last_batch:
                    break
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": config,
                "classes": classes,
                "epoch": epoch,
                "step": step,
                "history": history,
                "elapsed_seconds": elapsed_before + time.perf_counter() - start,
                "best_top1": best_top1,
                "best_epoch": best_epoch,
                "best_model": best_model,
            }
            save_checkpoint(output / "checkpoint.pt", state)
            best_state = dict(state, model=best_model)
            # Optimizer state belongs to epoch-end model; best.pt is inference-only.
            best_state.pop("optimizer")
            best_state.pop("scheduler")
            save_checkpoint(output / "best.pt", best_state)
            if args.stop_after_epoch and epoch >= args.stop_after_epoch:
                print(
                    f"Stopped at epoch {epoch}. Resume {output / 'checkpoint.pt'}",
                    flush=True,
                )
                return
    if args.eval_only:
        with (output / "metrics.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
    model.load_state_dict(best_model)
    test, targets, predictions = evaluate(
        model, test_loader, device, args.max_eval_batches
    )
    write_reports(history, targets, predictions, test_data, classes, output)
    summary = {
        "config": config,
        "parameters": parameters,
        "completed_epochs": start_epoch if args.eval_only else args.epochs,
        "step": step,
        "best_epoch": best_epoch,
        "best_val_top1": best_top1,
        "train_samples": len(train_data),
        "validation_samples": len(val_data),
        "test_samples": len(test_data),
        "limited_run": bool(args.max_train_batches or args.max_eval_batches),
        "elapsed_seconds": elapsed_before + time.perf_counter() - start,
        "test": test,
        "classes": classes,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"Best epoch={best_epoch}; test loss={test['loss']:.4f}, top1={test['top1']:.2%}, "
        f"top5={test['top5']:.2%}, samples={test['samples']}. Outputs: {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
