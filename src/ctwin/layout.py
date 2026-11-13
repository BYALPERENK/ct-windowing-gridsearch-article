"""On-disk layout of the curated E-NCCT dataset.

Every preparation script writes 256x256 int16 HU arrays (.npy) into

    <data_root>/<split>/<class>/<source prefix><original id>.npy

with split in {train, val, test} and class in {0, 1, 2}:
0 = normal or non-acute findings, 1 = acute ischemic stroke, 2 = hemorrhagic stroke.
"""

from __future__ import annotations

import os

SPLITS = ("train", "val", "test")
CLASSES = ("0", "1", "2")
CLASS_NAMES = {"0": "normal", "1": "ischemic", "2": "hemorrhagic"}


def class_dir(data_root: str, split: str, cls: str) -> str:
    """Return <data_root>/<split>/<cls>, validating both names."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    if cls not in CLASSES:
        raise ValueError(f"unknown class {cls!r}; expected one of {CLASSES}")
    return os.path.join(data_root, split, cls)
