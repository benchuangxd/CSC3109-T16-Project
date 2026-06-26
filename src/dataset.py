from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_transforms(image_size: int = 224):
    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def get_dataloaders(
    root: Path,
    data_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    train_ratio: float = 0.7,
    seed: int = 42,
    num_workers: int = 2,
):
    """Create train/validation loaders with a deterministic split per class folder."""
    train_tf, val_tf = get_transforms(image_size)

    full_ds_train = datasets.ImageFolder(root / data_dir, transform=train_tf)
    full_ds_val   = datasets.ImageFolder(root / data_dir, transform=val_tf)

    # Stratified split inside each class folder under data/set 16, so
    # beach/ferry_terminal/harbor/river each keep the same train/val ratio
    # (e.g. 490 train / 210 val per class for TRAIN_RATIO = 0.7).
    targets = np.array(full_ds_train.targets)
    rng = np.random.RandomState(seed)
    train_idx, val_idx = [], []
    for c in np.unique(targets):
        cls_idx = np.where(targets == c)[0]
        rng.shuffle(cls_idx)
        split = int(round(train_ratio * len(cls_idx)))
        train_idx += cls_idx[:split].tolist()
        val_idx   += cls_idx[split:].tolist()

    train_ds = Subset(full_ds_train, train_idx)
    val_ds   = Subset(full_ds_val, val_idx)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_dl, val_dl, full_ds_train.classes
