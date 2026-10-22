"""Plots and named per-class reports for CIFAR-100."""

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from zoo.data.cifar100 import MEAN, STD


def write_reports(history, targets, predictions, test_data, classes, output):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, ("loss", "top1", "top5")):
        for split in ("train", "val"):
            ax.plot([row["step"] for row in history], [row[f"{split}_{metric}"] for row in history], label=split)
        ax.set(xlabel="Optimizer step", ylabel=metric, title=metric)
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "curves.png", dpi=150)
    plt.close(fig)

    count = len(classes)
    matrix = torch.bincount(count * targets + predictions, minlength=count ** 2).reshape(count, count)
    fig, ax = plt.subplots(figsize=(22, 20))
    image = ax.imshow(matrix.numpy(), cmap="Blues")
    ax.set(xticks=range(count), yticks=range(count), xticklabels=classes, yticklabels=classes,
           xlabel="Predicted class", ylabel="True class", title="CIFAR-100 test confusion matrix")
    ax.tick_params(axis="both", labelsize=6)
    plt.setp(ax.get_xticklabels(), rotation=90)
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    with (output / "confusion_matrix.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["true/predicted", *classes])
        for name, row in zip(classes, matrix.tolist()):
            writer.writerow([name, *row])
    with (output / "per_class.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["class_id", "class_name", "accuracy", "correct", "support"])
        for index, name in enumerate(classes):
            support, correct = matrix[index].sum().item(), matrix[index, index].item()
            writer.writerow([index, name, correct / support if support else "", correct, support])

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for index, ax in enumerate(axes.flat):
        ax.axis("off")
        if index >= len(predictions):
            continue
        image, label = test_data[index]
        image = (image * torch.tensor(STD)[:, None, None] + torch.tensor(MEAN)[:, None, None]).clamp(0, 1)
        predicted = predictions[index].item()
        ax.imshow(image.permute(1, 2, 0).numpy(), interpolation="nearest")
        ax.set_title(f"True: {classes[label]}\nPred: {classes[predicted]}", fontsize=9,
                     color="green" if label == predicted else "red")
    fig.tight_layout()
    fig.savefig(output / "predictions.png", dpi=150)
    plt.close(fig)
