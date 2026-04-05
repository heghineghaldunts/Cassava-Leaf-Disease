# dataset.py  -  CassavaDataset + augmentation transforms + data-split helpers.

import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import (
    BATCH_SIZE,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_WORKERS,
    RANDOM_STATE,
    RESIZE_TO,
    TEST_SIZE,
    TRAIN_CSV,
    TRAIN_IMG_DIR,
    CLASS_WEIGHT_POWER,
    DEVICE,
)

# Augmentation pipelines

def get_train_transform() -> transforms.Compose:
    """Heavy augmentation pipeline used during training."""
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=15,
                translate=(0.05, 0.05),
                scale=(0.9, 1.1),
            )
        ], p=0.5),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.05,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(
            p=0.3,
            scale=(0.02, 0.20),
            ratio=(0.3, 3.3),
            value="random",
        ),
    ])


def get_val_transform() -> transforms.Compose:
    """Deterministic pipeline used at validation / inference time."""
    return transforms.Compose([
        transforms.Resize(RESIZE_TO),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# Dataset

class CassavaDataset(Dataset):
    """
    PyTorch Dataset for the Cassava Leaf Disease classification task.

    Args:
        image_paths (array-like): Absolute paths to each image file.
        labels      (array-like): Integer class labels (0-4).
        transform   (callable):   torchvision transform applied to each image.
    """

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = np.array(image_paths)
        self.labels = np.array(labels, dtype=int)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        with Image.open(img_path) as img:
            image = img.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]


# Data-loading helpers

def load_dataframes(train_csv: str = TRAIN_CSV, train_img_dir: str = TRAIN_IMG_DIR):
    """
    Read the CSV, attach full image paths, and split into train / val.

    Returns:
        X_train, X_val, y_train, y_val - pandas DataFrames / Series.
    """
    df = pd.read_csv(train_csv)
    df["image_path"] = df["image_id"].apply(
        lambda x: os.path.join(train_img_dir, x)
    )

    X = df.drop(columns=["label"])
    y = df[["label"]]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify = y,
    )
    return X_train, X_val, y_train, y_val


def compute_class_weights(y_train: pd.DataFrame) -> torch.Tensor:
    """
    Compute inverse-frequency class weights (with optional power dampening).

    Weights are normalised so they sum to ``NUM_CLASSES``, which keeps the
    loss scale comparable to unweighted training.
    """
    counts = y_train["label"].value_counts().sort_index().values.astype(float)
    weights = (1.0 / counts) ** CLASS_WEIGHT_POWER
    n_classes = len(counts)
    weights = weights / weights.sum() * n_classes
    return torch.tensor(weights, dtype=torch.float).to(DEVICE)


def get_dataloaders(
    X_train, X_val, y_train, y_val
) -> tuple[DataLoader, DataLoader]:
    """
    Build and return (train_loader, val_loader).
    """
    train_ds = CassavaDataset(
        X_train["image_path"].values,
        y_train["label"].values,
        transform=get_train_transform(),
    )
    val_ds = CassavaDataset(
        X_val["image_path"].values,
        y_val["label"].values,
        transform=get_val_transform(),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=NUM_WORKERS,
    )
    return train_loader, val_loader
