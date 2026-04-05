# model.py  -  Model factories for both the main model (ResNet50) and the baseline (ResNet18).

import torch
import torch.nn as nn
from torchvision import models

from config import BEST_MODEL_PATH, BASELINE_MODEL_PATH, DEVICE, NUM_CLASSES



# ResNet50 - main model

def build_resnet50(freeze_backbone: bool = True) -> nn.Module:
    """
    Load ImageNet-pre-trained ResNet50 and replace its classifier head.

    The head uses:
        Linear(2048 -> 512) -> ReLU -> Dropout(0.5) -> Linear(512 -> NUM_CLASSES)

    Args:
        freeze_backbone: If True, all backbone parameters are frozen so only
                         the new head trains in Phase 1.

    Returns:
        model (nn.Module) already moved to DEVICE.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, NUM_CLASSES),
    )

    return model.to(DEVICE)


def load_resnet50(path: str = BEST_MODEL_PATH) -> nn.Module:
    """
    Reconstruct the ResNet50 architecture and load saved weights.

    Args:
        path: Path to the ``.pth`` checkpoint produced by train.py.

    Returns:
        model set to ``eval()`` mode.
    """
    model = build_resnet50(freeze_backbone=False)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


# ResNet18 - baseline (used by baseline.py)

def build_resnet18(freeze_backbone: bool = False) -> nn.Module:
    """
    Load ImageNet-pre-trained ResNet18 and replace its classifier head.

    The baseline uses a simpler, single Linear layer to keep it comparable
    to a "minimal fine-tuning" scenario.  The full backbone is fine-tuned
    end-to-end from the start (no progressive unfreezing).

    Args:
        freeze_backbone: Set True to freeze the backbone (not used in the
                         default baseline run, but exposed for experiments).

    Returns:
        model (nn.Module) already moved to DEVICE.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features        
    model.fc = nn.Linear(in_features, NUM_CLASSES)

    return model.to(DEVICE)


def load_resnet18(path: str = BASELINE_MODEL_PATH) -> nn.Module:
    """
    Reconstruct the ResNet18 architecture and load saved weights.

    Args:
        path: Path to the ``.pth`` checkpoint produced by baseline.py.

    Returns:
        model set to ``eval()`` mode.
    """
    model = build_resnet18(freeze_backbone=False)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model
