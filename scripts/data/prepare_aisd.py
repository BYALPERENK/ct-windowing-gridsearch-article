"""Prepare AISD (paper Sec. 3.1, Source 1): class 1 slices.

1. For every mask slice (PNG) the pixels labelled 2 ("clear acute infarct")
   or 3 ("blurred acute infarct") are counted; lesion-bearing slices get a
   coverage value: acute pixels / mask size * 100.
2. Slices above the median coverage of the whole source are kept, and at most
   five per patient, those with the largest coverage (ctwin.slice_selection;
   sorted by coverage only, with the pandas default sort, as in the original).
3. Patients (not slices) are split 70/10/20: the unique patients, in their
   order of appearance in the selected table, are shuffled with
   np.random.seed(42); the first int(0.7 n) go to train, the next int(0.1 n)
   to validation and the rest to test.
4. Mask <i>.png is taken to belong to the i-th file (0-based) of the
   patient's sorted CT DICOM list (a file-order assumption of the original;
   InstanceNumber is not used); that DICOM goes through the common curation pipeline
   (ctwin.preprocessing).

Output: <data_root>/<split>/1/aisd_<patient>_<i>.npy

Inputs (layout of the AISD release):
    --dicom-dir   <dicom-dir>/<patient>/CT/*.dcm
    --mask-dir    <mask-dir>/<patient>/<i>.png

Patients and mask files are processed in sorted order. The original used
os.listdir / glob order; on NTFS (where it ran) that order is expected to
be the sorted order for these zero-padded numeric names, but the APIs do not
guarantee it. Because the split shuffles the patients in their order of
appearance after a non-stable sort, ties in coverage make the split depend on
that order, and in principle on the NumPy sort implementation.
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
from PIL import Image

from ctwin.layout import class_dir
from ctwin.preprocessing import dicom_file_to_array, save_array
from ctwin.slice_selection import MAX_SLICES_PER_PATIENT, above_median_top_k

SEED = 42
CLASS = "1"
ACUTE_LABELS = (2, 3)


def lesion_coverage_table(mask_dir: str) -> pd.DataFrame:
    """One row per mask slice with acute infarct: PatientID, MaskFilename, Coverage (in %)."""
    rows = []
    patients = sorted(d for d in os.listdir(mask_dir) if os.path.isdir(os.path.join(mask_dir, d)))
    for patient in patients:
        for mask_path in sorted(glob.glob(os.path.join(mask_dir, patient, "*.png"))):
            with Image.open(mask_path) as img:
                arr = np.array(img)
            acute = np.isin(arr, ACUTE_LABELS)
            lesion_pixels = np.count_nonzero(acute)
            if lesion_pixels > 0:
                rows.append({"PatientID": patient, "MaskFilename": os.path.basename(mask_path),
                             "Coverage": lesion_pixels / arr.size * 100.0})
    if not rows:
        raise ValueError("no mask slice with label 2 or 3 found")
    return pd.DataFrame(rows)


def split_patients(selected: pd.DataFrame) -> dict[str, str]:
    """Map patient -> 'train' | 'val' | 'test' (70/10/20 at patient level)."""
    # A plain object ndarray, as pandas < 3 returned: pandas 3 returns a
    # StringArray, which np.random.shuffle does not support reliably.
    patients = np.asarray(selected["PatientID"].unique(), dtype=object)
    np.random.seed(SEED)
    np.random.shuffle(patients)
    n = len(patients)
    n_train, n_val = int(n * 0.70), int(n * 0.10)
    split_map = {p: "train" for p in patients[:n_train]}
    split_map.update({p: "val" for p in patients[n_train:n_train + n_val]})
    split_map.update({p: "test" for p in patients[n_train + n_val:]})
    return split_map


def save_slices(selected: pd.DataFrame, split_map: dict[str, str], dicom_dir: str, data_root: str) -> None:
    for split in ("test", "train", "val"):  # processing order of the original
        part = selected[selected["PatientID"].map(split_map) == split]
        ok = failed = 0
        for _, row in part.iterrows():
            try:
                patient = row["PatientID"]
                index = int(os.path.splitext(row["MaskFilename"])[0])  # 005.png -> 5
                dicoms = sorted(glob.glob(os.path.join(dicom_dir, patient, "CT", "*.dcm")))
                if index >= len(dicoms):
                    raise IndexError(f"mask index {index} outside the {len(dicoms)} DICOMs of {patient}")
                save_array(dicom_file_to_array(dicoms[index]), class_dir(data_root, split, CLASS),
                           f"aisd_{patient}_{index}")
                ok += 1
            except Exception:
                failed += 1
        print(f"{split}: {ok} saved, {failed} failed (of {len(part)})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dicom-dir", required=True, help="AISD dicom/ folder (<patient>/CT/*.dcm)")
    p.add_argument("--mask-dir", required=True, help="AISD mask/ folder (<patient>/<i>.png)")
    p.add_argument("--data-root", required=True, help="root of the curated dataset (output)")
    a = p.parse_args()
    for d in (a.dicom_dir, a.mask_dir):
        if not os.path.isdir(d):
            p.error(f"not a directory: {d}")
    for split in ("train", "val", "test"):  # created up front, as in the original
        os.makedirs(class_dir(a.data_root, split, CLASS), exist_ok=True)
    table = lesion_coverage_table(a.mask_dir)
    selected, median = above_median_top_k(table, "Coverage", "PatientID",
                                          sort_by="Coverage", ascending=False,
                                          k=MAX_SLICES_PER_PATIENT)
    print(f"{len(table)} lesion slices, median coverage {median:.4f}%, {len(selected)} selected")
    split_map = split_patients(selected)
    for s in ("train", "val", "test"):
        print(f"{s}: {sum(v == s for v in split_map.values())} patients")
    save_slices(selected, split_map, a.dicom_dir, a.data_root)


if __name__ == "__main__":
    main()
