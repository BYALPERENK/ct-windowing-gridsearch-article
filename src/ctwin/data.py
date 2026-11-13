"""Loading the curated E-NCCT arrays and wrapping them for PyTorch.

The curated data root (see scripts/data/) has the layout

    <data_root>/{train,val,test}/{0,1,2}/*.npy

with 256x256 int16 HU arrays. Each binary task keeps two of the three
classes and relabels them to 0/1 (paper Sec. 3.3).
"""

from __future__ import annotations

import gc
import os
import time
from typing import Callable

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

# original class folder -> label used for training
TASKS: dict[str, dict[int, int]] = {
    "ischemic": {0: 0, 1: 1},      # normal vs. ischemic stroke
    "hemorrhagic": {0: 0, 2: 1},   # normal vs. hemorrhagic stroke
}


def task_class_names(task: str) -> tuple[str, str]:
    """Original class folder names of a task, in label order, e.g. ("0", "2")."""
    label_map = TASKS[task]
    return tuple(str(c) for c, _ in sorted(label_map.items(), key=lambda kv: kv[1]))


def load_split(split_dir: str, label_map: dict[int, int], log: Callable[[str], None] = print
               ) -> tuple[np.ndarray, np.ndarray]:
    """Load every .npy slice of the classes in label_map from one split folder.

    Class folders are visited in sorted order and files in sorted order within
    each class, so the sample order is fixed (it matters for the seeded
    shuffling of the training loader). Folders that are not class numbers, or
    whose class is not part of label_map, are skipped.

    Returns (images int16 of shape (N, H, W), labels int64 of shape (N,)).
    """
    imgs, labs = [], []
    for cls_name in sorted(os.listdir(split_dir)):
        cls_path = os.path.join(split_dir, cls_name)
        if not os.path.isdir(cls_path) or not cls_name.isdigit():
            continue
        original_label = int(cls_name)
        if original_label not in label_map:
            log(f"    class {original_label}: skipped (not part of this task)")
            continue
        new_label = label_map[original_label]
        files = sorted(f for f in os.listdir(cls_path) if f.endswith(".npy"))
        t0 = time.time()
        for f in files:
            imgs.append(np.load(os.path.join(cls_path, f)))
            labs.append(new_label)
        log(f"    class {original_label} -> {new_label}: {len(files):>5} files ({time.time() - t0:.1f}s)")

    if not imgs:
        raise FileNotFoundError(f"no .npy slices of classes {sorted(label_map)} under {split_dir}")
    images = np.stack(imgs).astype(np.int16)
    labels = np.array(labs, dtype=np.int64)
    del imgs, labs
    gc.collect()
    missing = sorted(set(label_map.values()) - set(labels.tolist()))
    if missing:
        raise ValueError(f"{split_dir}: no slices for label(s) {missing} of this task")
    return images, labels


class CTDataset(Dataset):
    """Pre-normalized float32 slices; each item is a (1, H, W) tensor and its label."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.images[idx]).unsqueeze(0)
        if self.transform:
            x = self.transform(x)
        return x, self.labels[idx]


def train_transform() -> transforms.Compose:
    """Geometric augmentation only (paper Sec. 3.5): random horizontal and
    vertical flips, each with probability 0.5. No intensity augmentation."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ])


def make_loader(images: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool,
                transform=None) -> DataLoader:
    """Single-process loader, as used in every experiment."""
    return DataLoader(
        CTDataset(images, labels, transform),
        batch_size=batch_size, shuffle=shuffle,
        num_workers=0, pin_memory=False, drop_last=False,
    )
