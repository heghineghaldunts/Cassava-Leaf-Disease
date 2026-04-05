# baseline.py  -  ResNet18 baseline.
#
# Intentionally simple: ImageNet pre-trained ResNet18, head-only fine-tuning,
# no CutMix, no progressive unfreezing, no early stopping (fixed epochs).
# This is the performance floor that the ResNet50 + CutMix model must beat.
#
# Usage:
#   python baseline.py

import torch
import torch.nn as nn

from config import (
    BASELINE_EPOCHS,
    BASELINE_LR,
    BASELINE_MODEL_PATH,
    DEVICE,
)
from dataset import get_dataloaders, load_dataframes
from model import build_resnet18
from utils import (
    evaluate,
    plot_confusion_matrix,
    plot_training_curves,
    print_metrics,
)


def train_one_epoch(model, loader, criterion, optimizer) -> tuple[float, float]:
    """Run a single training epoch. Returns (avg_loss, accuracy_pct)."""
    model.train()
    running_loss = correct = total = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    return running_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def val_one_epoch(model, loader, criterion) -> tuple[float, float]:
    """Run a single validation epoch. Returns (avg_loss, accuracy_pct)."""
    model.eval()
    running_loss = correct = total = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    return running_loss / len(loader), 100.0 * correct / total


def main() -> None:
    print(f"Using device: {DEVICE}")
    print("=" * 60)
    print("Baseline: ResNet18 - end-to-end fine-tuning")
    print("=" * 60)

    X_train, X_val, y_train, y_val = load_dataframes()
    train_loader, val_loader = get_dataloaders(X_train, X_val, y_train, y_val)

    criterion = nn.CrossEntropyLoss()

    model = build_resnet18(freeze_backbone=True)
    optimizer = torch.optim.Adam(
        model.fc.parameters(), lr=BASELINE_LR
    )

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_acc = 0.0

    for epoch in range(1, BASELINE_EPOCHS + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_acc = val_one_epoch(model, val_loader, criterion)

        train_losses.append(t_loss)
        val_losses.append(v_loss)
        train_accs.append(t_acc)
        val_accs.append(v_acc)

        print(
            f"  [Epoch {epoch:02d}/{BASELINE_EPOCHS}] "
            f"Train: {t_acc:.2f}% (loss {t_loss:.4f}) | "
            f"Val: {v_acc:.2f}% (loss {v_loss:.4f})"
        )

        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), BASELINE_MODEL_PATH)
            print(f"  =>  New best saved ({best_acc:.2f}%)")

    print(f"\nBaseline Best Val Accuracy: {best_acc:.2f}%")

    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs, prefix="baseline_"
    )

    print("\n--- Final Evaluation ---")
    model.load_state_dict(torch.load(BASELINE_MODEL_PATH, map_location=DEVICE))
    all_preds, all_labels = evaluate(model, val_loader)
    metrics = print_metrics(all_labels, all_preds)
    plot_confusion_matrix(
        all_labels, all_preds,
        accuracy=metrics["accuracy"],
        f1=metrics["f1"],
        filename="baseline_confusion_matrix.png",
    )


if __name__ == "__main__":
    main()
