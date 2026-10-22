import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from model import ConvNeXt
from torch import nn

from zoo.data import make_loader
from zoo.data.mnist import add_data_arguments, config_from_args, make_splits


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    loss, correct, count = 0.0, 0, 0
    targets, predictions = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        predicted = logits.argmax(dim=1)
        loss += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
        correct += (predicted == labels).sum().item()
        count += labels.numel()
        targets.append(labels.cpu())
        predictions.append(predicted.cpu())
    return loss / count, correct / count, torch.cat(targets), torch.cat(predictions)


def plot_results(history, targets, predictions, test_data, output):
    steps = [row["step"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ("loss", "accuracy")):
        for split in ("train", "val"):
            ax.plot(steps, [row[f"{split}_{metric}"] for row in history], label=split)
        ax.set(xlabel="Optimizer step", ylabel=metric.title(), title=metric.title())
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "curves.png", dpi=150)
    plt.close(fig)

    matrix = torch.bincount(10 * targets + predictions, minlength=100).reshape(10, 10)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix.numpy(), cmap="Blues")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(matrix[i, j].item()), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    ax.set(xticks=range(10), yticks=range(10), xlabel="Predicted digit",
           ylabel="True digit", title="MNIST test confusion matrix")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for index, ax in enumerate(axes.flat):
        image, label = test_data[index]
        predicted = predictions[index].item()
        ax.imshow(image.squeeze(), cmap="gray")
        ax.set_title(f"True: {label} / Pred: {predicted}",
                     color="green" if label == predicted else "red", fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output / "predictions.png", dpi=150)
    plt.close(fig)

    support = matrix.sum(dim=1)
    precision = matrix.diag() / matrix.sum(dim=0).clamp_min(1)
    recall = matrix.diag() / support.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
    with (output / "per_class.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["digit", "precision", "recall", "f1", "support"])
        for digit in range(10):
            writer.writerow([digit, precision[digit].item(), recall[digit].item(),
                             f1[digit].item(), support[digit].item()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    add_data_arguments(parser)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).parent / "outputs")
    args = parser.parse_args()
    if args.steps < 1 or args.log_every < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("steps, log-every, batch-size, and lr must be positive")
    try:
        data_config = config_from_args(args)
    except ValueError as error:
        parser.error(str(error))

    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    train_data, val_data, test_data = make_splits(data_config, args.seed)
    train_loader = make_loader(train_data, data_config, args.batch_size, shuffle=True, seed=args.seed)
    val_loader = make_loader(val_data, data_config, 256)
    test_loader = make_loader(test_data, data_config, 256)
    model = ConvNeXt().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    history = []
    batches = iter(train_loader)

    with (output / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "step", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "lr",
        ])
        writer.writeheader()
        for step in range(1, args.steps + 1):
            try:
                images, labels = next(batches)
            except StopIteration:
                batches = iter(train_loader)
                images, labels = next(batches)
            model.train()
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            loss.backward()
            lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            if step % args.log_every == 0 or step == args.steps:
                val_loss, val_accuracy, _, _ = evaluate(model, val_loader, device)
                row = dict(step=step, train_loss=loss.item(),
                           train_accuracy=(logits.argmax(1) == labels).float().mean().item(),
                           val_loss=val_loss, val_accuracy=val_accuracy, lr=lr)
                history.append(row)
                writer.writerow(row)
                file.flush()
                print(f"Step {step:3d}/{args.steps}: loss={loss.item():.4f} "
                      f"val_loss={val_loss:.4f} val_accuracy={val_accuracy:.1%} lr={lr:.2e}")

    test_loss, test_accuracy, targets, predictions = evaluate(model, test_loader, device)
    plot_results(history, targets, predictions, test_data, output)
    summary = dict(steps=args.steps, seed=args.seed, batch_size=args.batch_size,
                   lr=args.lr, weight_decay=0.01, device=str(device),
                   data_backend=args.backend, num_workers=args.num_workers,
                   parameters=sum(p.numel() for p in model.parameters()),
                   validation_samples=len(val_data), test_samples=len(test_data),
                   test_loss=test_loss, test_accuracy=test_accuracy)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "summary": summary}, output / "checkpoint.pt")
    print(f"Test loss={test_loss:.4f}, accuracy={test_accuracy:.1%}. Outputs: {output}")


if __name__ == "__main__":
    main()
