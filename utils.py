# utils.py  -  Evaluation metrics + plotting helpers.
# Plots are saved to PLOTS_DIR (defined in config.py).

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config import CLASS_NAMES, DEVICE, PLOTS_DIR


# Setup

def ensure_plots_dir() -> None:
    """Create the plots output directory if it does not already exist."""
    os.makedirs(PLOTS_DIR, exist_ok=True)


# Evaluation

@torch.no_grad()
def evaluate(model: torch.nn.Module, loader) -> tuple[list, list]:
    """
    Run inference over ``loader`` and collect predictions + ground-truth labels.

    Returns:
        all_preds  (list[int]): predicted class indices.
        all_labels (list[int]): ground-truth class indices.
    """
    model.eval()
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

    return all_preds, all_labels


def print_metrics(all_labels: list, all_preds: list) -> dict:
    """
    Compute and pretty-print accuracy, F1, precision, recall.

    Returns:
        metrics (dict): {"accuracy", "f1", "precision", "recall"} all in [0,100].
    """
    accuracy = accuracy_score(all_labels, all_preds) * 100
    f1 = f1_score(all_labels, all_preds, average="weighted") * 100
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0) * 100
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0) * 100

    sep = "=" * 45
    print(sep)
    print(f"  Accuracy  : {accuracy:.2f}%")
    print(f"  F1 Score  : {f1:.2f}%")
    print(f"  Precision : {precision:.2f}%")
    print(f"  Recall    : {recall:.2f}%")
    print(sep)
    print("\nPer-Class Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    return dict(accuracy=accuracy, f1=f1, precision=precision, recall=recall)


# Plots

def plot_confusion_matrix(all_labels: list, all_preds: list, accuracy: float, f1: float, filename: str = "confusion_matrix.png") -> None:
    """
    Save a normalised (percentage) confusion-matrix heatmap to PLOTS_DIR.
    """
    ensure_plots_dir()
    cm = confusion_matrix(all_labels, all_preds)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_pct,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(
        f"Confusion Matrix (%)  |  Acc: {accuracy:.2f}%  |  F1: {f1:.2f}%",
        fontsize=12,
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved → {save_path}")


def plot_training_curves(train_losses: list, val_losses: list, train_accs: list, val_accs: list, prefix: str = "") -> None:
    """
    Save loss and accuracy curves to PLOTS_DIR.

    Args:
        prefix: Optional filename prefix (e.g. "baseline_" vs "resnet50_").
    """
    ensure_plots_dir()
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="Train Loss", marker="o", markersize=3)
    ax.plot(epochs, val_losses,   label="Val Loss",   marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Validation Loss")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    loss_path = os.path.join(PLOTS_DIR, f"{prefix}loss_curve.png")
    plt.savefig(loss_path, dpi=150)
    plt.close()
    print(f"Loss curve saved -> {loss_path}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_accs, label="Train Accuracy", marker="o", markersize=3)
    ax.plot(epochs, val_accs,   label="Val Accuracy",   marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Training vs Validation Accuracy")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    acc_path = os.path.join(PLOTS_DIR, f"{prefix}accuracy_curve.png")
    plt.savefig(acc_path, dpi=150)
    plt.close()
    print(f"Accuracy curve saved -> {acc_path}")
