"""
Dataset for the die-face ResNet classifier.

Reads from DATASET_DIR/<digit>/*.jpg where <digit> is one of "1".."6"
(the layout label_tool.py produces). The "discard" folder is deliberately
excluded — it's not a class, just rejected crops.
"""
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from torchvision import transforms
from config import DATASET_DIR

CLASSES = [str(i) for i in range(1, 7)]  # index -> label name
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
IMG_SIZE = 64  # crops are small; no need for large input resolution


def default_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


class DiceFaceDataset(Dataset):
    """Flat dataset over DATASET_DIR/<digit>/*.jpg, digit in CLASSES only."""

    def __init__(self, dataset_dir: str = DATASET_DIR, transform=None):
        self.transform = transform
        self.samples = []  # list of (path, class_idx)

        for class_idx, class_name in enumerate(CLASSES):
            class_dir = os.path.join(dataset_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if os.path.splitext(fname)[1].lower() in IMAGE_EXTS:
                    self.samples.append((os.path.join(class_dir, fname), class_idx))

        if not self.samples:
            raise RuntimeError(
                f"No labeled images found under {dataset_dir}/<1-6>/. "
                "Run label_tool.py first to build the dataset."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def class_counts(self):
        counts = {c: 0 for c in CLASSES}
        for _, label in self.samples:
            counts[CLASSES[label]] += 1
        return counts


def get_dataloaders(batch_size: int = 32, val_split: float = 0.2, seed: int = 42):
    """
    Builds train/val DataLoaders from DATASET_DIR.

    Note: train and val share one underlying DiceFaceDataset instance (for a
    simple, reproducible split), but each subset wraps a transform-specific
    view so augmentation only applies to the training split.
    """
    full = DiceFaceDataset(transform=None)

    n_val = max(1, int(len(full) * val_split)) if len(full) > 4 else 0
    n_train = len(full) - n_val

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(full, [n_train, n_val], generator=generator)

    train_subset.dataset = _TransformWrapper(full, default_transforms(train=True))
    val_subset.dataset = _TransformWrapper(full, default_transforms(train=False))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0) if n_val else None

    print(f"Dataset: {len(full)} images | class counts: {full.class_counts()}")
    print(f"Train: {n_train}  Val: {n_val}")

    return train_loader, val_loader


class _TransformWrapper(Dataset):
    """Applies a given transform to an underlying (transform=None) dataset."""

    def __init__(self, base: DiceFaceDataset, transform):
        self.base = base
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        path, label = self.base.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label