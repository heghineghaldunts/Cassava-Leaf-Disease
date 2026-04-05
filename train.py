# train.py  -  Full training pipeline for the ResNet50 model.
#
# Strategy (mirrors the original notebook):
#   Phase 1  - train the classifier head only (backbone frozen).
#   Phase 2  - progressive unfreezing: layer4 -> layer3 -> layer2, each with its own learning rate and CosineAnnealingLR scheduler.
#              CutMix  augmentation is applied throughout Phase 2.
# Usage:
#   python train.py

import numpy as np
import torch
import torch.nn as nn

from config import (
    BEST_MODEL_PATH,
    CLASS_WEIGHT_POWER,
    CUTMIX_ALPHA,
    DEVICE,
    EARLY_STOPPING_PATIENCE,
    LABEL_SMOOTHING,
    MAX_GRAD_NORM,
    PHASE1_EPOCHS,
    PHASE1_LR,
    PHASE1_WD,
    PROGRESSIVE_STAGES,
)
from dataset import compute_class_weights, get_dataloaders, load_dataframes
from model import build_resnet50
from utils import (
    evaluate,
    plot_confusion_matrix,
    plot_training_curves,
    print_metrics,
)

# CutMix helper

def cutmix_batch(images: torch.Tensor, labels: torch.Tensor, alpha: float = CUTMIX_ALPHA):
    """
    Apply CutMix to a single batch.

    Returns:
        mixed_images, labels_a, labels_b, lam
    """
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(images.size(0)).to(DEVICE)

    W, H = images.size(3), images.size(2)
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(W * cut_ratio), int(H * cut_ratio)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = int(np.clip(cx - cut_w // 2, 0, W))
    y1 = int(np.clip(cy - cut_h // 2, 0, H))
    x2 = int(np.clip(cx + cut_w // 2, 0, W))
    y2 = int(np.clip(cy + cut_h // 2, 0, H))

    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]

    lam = 1.0 - (x2 - x1) * (y2 - y1) / (W * H)
    return mixed, labels, labels[index], lam

# Phase 1 - head only

def phase1(model, train_loader, criterion) -> None:
    """Train only the classifier head for PHASE1_EPOCHS epochs."""
    print("=" * 60)
    print("Phase 1: Training classifier head (backbone frozen)")
    print("=" * 60)

    optimizer = torch.optim.AdamW(
        model.fc.parameters(), lr=PHASE1_LR, weight_decay=PHASE1_WD
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=PHASE1_EPOCHS
    )

    for epoch in range(1, PHASE1_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = total = 0

        for images, labels in train_loader:
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

        scheduler.step()
        print(
            f"  [Epoch {epoch:02d}/{PHASE1_EPOCHS}] "
            f"Loss: {running_loss / len(train_loader):.4f} | "
            f"Train Acc: {100 * correct / total:.2f}%"
        )

# Phase 2 - progressive unfreezing with CutMix

def phase2(
    model,
    train_loader,
    val_loader,
    criterion,
) -> tuple[list, list, list, list]:
    """
    Progressive unfreezing of ResNet50 backbone layers.

    Tracking lists (loss / accuracy per epoch across ALL stages) are returned
    so they can be plotted with utils.plot_training_curves.
    """
    print("\n" + "=" * 60)
    print("Phase 2: Progressive fine-tuning with CutMix")
    print("=" * 60)

    best_acc         = 0.0
    epochs_no_improve = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for layer_name, description, lr, n_epochs in PROGRESSIVE_STAGES:
        layer = getattr(model, layer_name)
        for param in layer.parameters():
            param.requires_grad = True

        print(f"\n--- Unfreezing {description} | lr={lr} | {n_epochs} epochs ---")

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=PHASE1_WD,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs
        )

        for epoch in range(1, n_epochs + 1):
            model.train()
            t_loss = t_correct = t_total = 0

            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                mixed, labels_a, labels_b, lam = cutmix_batch(images, labels)

                optimizer.zero_grad()
                outputs = model(mixed)
                loss = lam * criterion(outputs, labels_a) + \
                       (1 - lam) * criterion(outputs, labels_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=MAX_GRAD_NORM
                )
                optimizer.step()

                t_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                t_total += labels.size(0)
                t_correct += (lam * (preds == labels_a).sum().item()
                              + (1 - lam) * (preds == labels_b).sum().item())

            train_acc = 100 * t_correct / t_total
            avg_t_loss = t_loss / len(train_loader)

            model.eval()
            v_loss = v_correct = v_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = model(images)
                    v_loss += criterion(outputs, labels).item()
                    _, preds = torch.max(outputs, 1)
                    v_total += labels.size(0)
                    v_correct += (preds == labels).sum().item()

            val_acc = 100 * v_correct / v_total
            avg_v_loss = v_loss / len(val_loader)

            train_losses.append(avg_t_loss)
            val_losses.append(avg_v_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            print(
                f"  [{description}][{epoch:02d}/{n_epochs}] "
                f"Train: {train_acc:.2f}% (loss {avg_t_loss:.4f}) | "
                f"Val: {val_acc:.2f}% (loss {avg_v_loss:.4f})"
            )

            if val_acc > best_acc:
                best_acc = val_acc
                epochs_no_improve = 0
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(f"  => ✓ New best saved ({best_acc:.2f}%)")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                    print(f"  => Early stopping triggered.")
                    return train_losses, val_losses, train_accs, val_accs

            scheduler.step()

    print(f"\nBest Validation Accuracy: {best_acc:.2f}%")
    return train_losses, val_losses, train_accs, val_accs



def main() -> None:
    print(f"Using device: {DEVICE}\n")

    X_train, X_val, y_train, y_val = load_dataframes()
    train_loader, val_loader = get_dataloaders(X_train, X_val, y_train, y_val)

    class_weights = compute_class_weights(y_train)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=LABEL_SMOOTHING
    )

    model = build_resnet50(freeze_backbone=True)

    phase1(model, train_loader, criterion)

    train_losses, val_losses, train_accs, val_accs = phase2(
        model, train_loader, val_loader, criterion
    )

    plot_training_curves(train_losses, val_losses, train_accs, val_accs, prefix="resnet50_")

    print("\n--- Final Evaluation ---")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    all_preds, all_labels = evaluate(model, val_loader)
    metrics = print_metrics(all_labels, all_preds)
    plot_confusion_matrix(
        all_labels, all_preds,
        accuracy=metrics["accuracy"],
        f1=metrics["f1"],
        filename="resnet50_confusion_matrix.png",
    )

if __name__ == "__main__":
    main()
