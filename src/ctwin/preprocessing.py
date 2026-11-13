"""Common curation pipeline shared by every E-NCCT source (paper Sec. 3.1.1).

Each 2D NCCT slice is converted to a 256x256 int16 array of Hounsfield Units:

1. MONOCHROME1 images are inverted to the MONOCHROME2 convention
   (max_val - pixel, with max_val = 2**BitsStored - 1), before rescaling.
2. Pixel values are converted to HU with RescaleSlope / RescaleIntercept.
3. The HU array is resized to 256x256 with area interpolation (cv2.INTER_AREA).
4. The result is rounded with np.rint and stored as int16.

No windowing is applied here; windowing happens at load time (see windowing.py).
"""

from __future__ import annotations

import glob
import os
import random
from typing import Sequence

import cv2
import numpy as np
import pydicom

TARGET_SIZE = (256, 256)


def dicom_to_hu(ds: pydicom.Dataset) -> np.ndarray:
    """Return the pixel data of a DICOM dataset in Hounsfield Units (float64)."""
    pixel_array = ds.pixel_array
    if ds.PhotometricInterpretation == "MONOCHROME1":
        bits_stored = getattr(ds, "BitsStored", 12)
        max_val = (2 ** bits_stored) - 1
        pixel_array = max_val - pixel_array
    image = pixel_array.astype(np.float64)
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    return image * slope + intercept


def resize_round_int16(image: np.ndarray, target_size: tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """Resize an HU array with INTER_AREA, round to nearest integer, cast to int16.

    target_size follows the OpenCV convention (width, height). The cast does
    not clip: values are assumed to lie within the int16 range, which holds
    for CT HU data.
    """
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    return np.rint(resized).astype(np.int16)


def dicom_file_to_array(path: str, target_size: tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """Read one DICOM file and return its curated 256x256 int16 HU array."""
    ds = pydicom.dcmread(path)
    return resize_round_int16(dicom_to_hu(ds), target_size)


def save_array(array: np.ndarray, dest_dir: str, name: str) -> str:
    """Save an array as <dest_dir>/<name>.npy and return the path."""
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, f"{name}.npy")
    np.save(path, array)
    return path


def convert_dicom_files(paths: Sequence[str], dest_dir: str, prefix: str) -> tuple[int, list[str]]:
    """Convert DICOM files and save each as <dest_dir>/<prefix><original stem>.npy.

    dest_dir is created up front, even if nothing ends up in it. A file that
    cannot be read, converted or saved is skipped, as in the original
    pipeline, and reported back. Two inputs that would map to the same output
    name (compared case-insensitively) raise a ValueError before anything is
    written, instead of silently overwriting each other.
    Returns (n_saved, failed_paths).
    """
    names: dict[str, str] = {}
    for path in paths:
        name = f"{prefix}{os.path.splitext(os.path.basename(path))[0]}"
        if name.lower() in names:
            raise ValueError(f"output name collision: {names[name.lower()]} and {path} -> {name}.npy")
        names[name.lower()] = path
    os.makedirs(dest_dir, exist_ok=True)
    ok = 0
    failed: list[str] = []
    for path in paths:
        name = f"{prefix}{os.path.splitext(os.path.basename(path))[0]}"
        try:
            save_array(dicom_file_to_array(path), dest_dir, name)
        except Exception:  # unreadable, malformed or unsavable: skipped
            failed.append(path)
            continue
        ok += 1
    return ok, failed


def find_dicoms(source_dir: str, recursive: bool = True) -> list[str]:
    """List the .dcm files under source_dir, sorted so that a given file set
    always comes back in the same order."""
    pattern = os.path.join(source_dir, "**", "*.dcm") if recursive else os.path.join(source_dir, "*.dcm")
    return sorted(glob.glob(pattern, recursive=recursive))


def seeded_sample(items: Sequence[str], n: int, seed: int = 42) -> list[str]:
    """Select n items at random with a fixed seed, as in the original scripts.

    The global `random` and `numpy.random` generators are seeded first (on
    purpose: the original scripts reset the global state this way), the items
    are sorted, and `random.sample` draws n of them. If fewer than n items are
    available, all of them are returned in a seeded random order.

    This is the selection rule of the original validation/test split code,
    where the order matters because the result is sliced afterwards. The
    original training-set code differed only when len(items) <= n (it kept
    every file in sorted order); the selected *set* is the same in that case,
    so the files written to disk are identical.
    """
    random.seed(seed)
    np.random.seed(seed)
    items = sorted(items)
    if len(items) >= n:
        return random.sample(items, n)
    random.shuffle(items)
    return items
