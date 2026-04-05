#!/usr/bin/env python3
# predict.py  -  Standalone inference script.
#
# Loads the saved best model (ResNet50 by default) and classifies a single
# cassava leaf image, printing a human-readable prediction + confidence.
#
# Usage:
#   python predict.py path/to/leaf.jpg
#   python predict.py path/to/leaf.jpg --model baseline   # use ResNet18
#
# Output example:
#   Prediction : Cassava Bacterial Blight
#   Confidence : 94.2%

import argparse
import sys

import torch
import torch.nn.functional as F
from PIL import Image

from config import (
    BASELINE_MODEL_PATH,
    BEST_MODEL_PATH,
    CLASS_NAMES,
    DEVICE,
)
from dataset import get_val_transform
from model import load_resnet18, load_resnet50


def load_model(model_type: str) -> torch.nn.Module:
    """
    Load and return the requested model (all weights, eval mode).

    Args:
        model_type: "resnet50" (default) or "baseline" / "resnet18".
    """
    if model_type in ("baseline", "resnet18"):
        print(f"Loading baseline ResNet18 from '{BASELINE_MODEL_PATH}' ...")
        return load_resnet18(BASELINE_MODEL_PATH)
    else:
        print(f"Loading ResNet50 from '{BEST_MODEL_PATH}' ...")
        return load_resnet50(BEST_MODEL_PATH)


def predict(image_path: str, model: torch.nn.Module) -> tuple[str, float]:
    """
    Preprocess a single image and return (class_name, confidence_pct).

    The preprocessing is identical to the validation transform so the model
    sees exactly the same pixel distribution it was evaluated on.
    """
    transform = get_val_transform()

    with Image.open(image_path) as img:
        image = img.convert("RGB")

    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)                    
        probs = F.softmax(logits, dim=1)[0]     
        class_idx = probs.argmax().item()
        confidence = probs[class_idx].item() * 100.0

    return CLASS_NAMES[class_idx], confidence


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify a cassava leaf image using a trained model."
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the input image (JPEG, PNG, ...).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        choices=["resnet50", "baseline", "resnet18"],
        help="Which checkpoint to load (default: resnet50).",
    )
    args = parser.parse_args()

    try:
        model = load_model(args.model)
    except FileNotFoundError as e:
        print(f"\nERROR: Checkpoint not found - {e}")
        print("Run train.py (or baseline.py) first to generate the checkpoint.")
        sys.exit(1)

    try:
        class_name, confidence = predict(args.image_path, model)
    except FileNotFoundError:
        print(f"\nERROR: Image not found - '{args.image_path}'")
        sys.exit(1)

    print()
    print(f"Prediction : {class_name}")
    print(f"Confidence : {confidence:.1f}%")


if __name__ == "__main__":
    main()