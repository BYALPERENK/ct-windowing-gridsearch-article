"""Prepare the TEKNOFEST 2021 Stroke Dataset (paper Sec. 3.1, Source 3).

The release is organised as a training set and two competition sessions and
has no patient identifiers, so the split follows that structure:

* Training set -> train split. A fixed-seed random sample of
  2,000 normal, 1,100 ischemic and 1,000 hemorrhagic images.
* Competition sessions 1 and 2 -> validation and test splits. Each image is
  labelled from its colour mask (blue = ischemia, green = hemorrhage). Images
  with both colours or with neither are excluded. The generator is seeded
  once (42), the ischemic list is shuffled, then the hemorrhagic list, and in
  each the first half (rounded down) goes to test and the rest to validation.
  The paper's release gives 79 ischemic and 82 hemorrhagic images, i.e.
  test 39 / val 40 and test 41 / val 41.

Output files are named t21_<original id>.npy (see ctwin.layout).

Expected inputs (folder names of the original release may differ):
    --normal-dir      DICOMs of the "no stroke" training set
    --ischemic-dir    DICOMs of the ischemia training set
    --hemorrhage-dir  DICOMs of the hemorrhage training set
    --session-dirs    the competition sessions IN ORDER (session 1, then
                      session 2), each folder holding DICOM/<id>.dcm and
                      MASKS/<id>.png

Session files are processed in sorted order. The original notebook used the
unsorted glob order, which on NTFS (where it was run) is already the sorted
order for these numeric file names; sorting keeps that order on every OS.
"""

from __future__ import annotations

import argparse
import glob
import os
import random

import cv2
import numpy as np

from ctwin.layout import class_dir
from ctwin.preprocessing import convert_dicom_files, find_dicoms, seeded_sample

PREFIX = "t21_"
PAPER_SESSION_COUNTS = {"1": 79, "2": 82}  # stroke-positive session images in the paper
SEED = 42
TRAIN_SAMPLES = {"0": 2000, "1": 1100, "2": 1000}
MASK_THRESHOLD = 50  # channel maximum above which a mask colour counts as present


def read_mask_bgr(path: str) -> np.ndarray | None:
    """Read a colour mask as BGR. np.fromfile + imdecode also handles
    non-ASCII paths on Windows, which cv2.imread does not."""
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def mask_label(mask: np.ndarray) -> str | None:
    """Map a BGR mask to a class: '1' blue only, '2' green only, None otherwise."""
    has_blue = mask[:, :, 0].max() > MASK_THRESHOLD
    has_green = mask[:, :, 1].max() > MASK_THRESHOLD
    if has_blue and not has_green:
        return "1"
    if has_green and not has_blue:
        return "2"
    return None  # both pathologies, or a normal (black) mask: excluded


def label_session_images(session_dirs: list[str]) -> dict[str, list[str]]:
    """Collect the stroke-positive DICOM paths of the competition sessions,
    sessions in the given order and files sorted within each session."""
    labelled: dict[str, list[str]] = {"1": [], "2": []}
    for session in session_dirs:
        for dcm_path in sorted(glob.glob(os.path.join(session, "DICOM", "*.dcm"))):
            file_id = os.path.splitext(os.path.basename(dcm_path))[0]
            mask_path = os.path.join(session, "MASKS", f"{file_id}.png")
            if not os.path.exists(mask_path):
                continue
            mask = read_mask_bgr(mask_path)
            if mask is None:
                continue
            label = mask_label(mask)
            if label is not None:
                labelled[label].append(dcm_path)
    return labelled


def prepare_train(source_dirs: dict[str, str], data_root: str) -> None:
    for cls, source in source_dirs.items():
        selected = seeded_sample(find_dicoms(source), TRAIN_SAMPLES[cls], SEED)
        ok, failed = convert_dicom_files(selected, class_dir(data_root, "train", cls), PREFIX)
        print(f"train/{cls}: {ok} saved, {len(failed)} failed (of {len(selected)} selected)")


def prepare_val_test(session_dirs: list[str], data_root: str) -> None:
    labelled = label_session_images(session_dirs)
    for cls, expected in PAPER_SESSION_COUNTS.items():
        if len(labelled[cls]) != expected:
            print(f"warning: {len(labelled[cls])} images of class {cls} found in the sessions; "
                  f"the paper's release has {expected}")
    # Seed once, shuffle ischemic then hemorrhagic, split, and only then write.
    random.seed(SEED)
    splits: list[tuple[str, str, list[str]]] = []
    for cls in ("1", "2"):
        paths = list(labelled[cls])
        random.shuffle(paths)
        half = len(paths) // 2
        splits += [("test", cls, paths[:half]), ("val", cls, paths[half:])]
    for split, cls, part in splits:
        ok, failed = convert_dicom_files(part, class_dir(data_root, split, cls), PREFIX)
        print(f"{split}/{cls}: {ok} saved, {len(failed)} failed (of {len(part)})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--normal-dir", required=True, help="training set: DICOMs without stroke (class 0)")
    p.add_argument("--ischemic-dir", required=True, help="training set: ischemic DICOMs (class 1)")
    p.add_argument("--hemorrhage-dir", required=True, help="training set: hemorrhagic DICOMs (class 2)")
    p.add_argument("--session-dirs", nargs="+", required=True,
                   help="competition sessions in order; each holds DICOM/ and MASKS/")
    p.add_argument("--data-root", required=True, help="root of the curated dataset (output)")
    a = p.parse_args()
    for d in (a.normal_dir, a.ischemic_dir, a.hemorrhage_dir):
        if not os.path.isdir(d):
            p.error(f"not a directory: {d}")
    for d in a.session_dirs:
        if not (os.path.isdir(os.path.join(d, "DICOM")) and os.path.isdir(os.path.join(d, "MASKS"))):
            p.error(f"session folder must contain DICOM/ and MASKS/: {d}")
    if len(set(map(os.path.abspath, a.session_dirs))) != len(a.session_dirs):
        p.error("the same session folder was given twice")
    prepare_train({"0": a.normal_dir, "1": a.ischemic_dir, "2": a.hemorrhage_dir}, a.data_root)
    prepare_val_test(a.session_dirs, a.data_root)


if __name__ == "__main__":
    main()
