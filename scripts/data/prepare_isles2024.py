"""Prepare ISLES 2024 (paper Sec. 3.1, Source 2): class 1 slices.

1. Every lesion-bearing axial slice of the follow-up lesion mask
   (ses-02, released in admission-NCCT space) gets a coverage value:
   lesion pixels / (X * Y) * 100.
2. Slices above the median coverage of the whole source are kept, and at
   most five per subject, those with the largest coverage
   (ctwin.slice_selection).
3. Subjects (not slices) are split 70/10/20: train_test_split(test_size=0.20,
   random_state=42) gives the test subjects, and a second
   train_test_split(test_size=0.125, random_state=42) of the remainder gives
   the validation subjects (0.125 * 0.80 = 0.10).
4. The admission NCCT (ses-01) is read with nibabel (get_fdata, which applies
   the NIfTI scaling; the ISLES NCCT values are HU), each selected slice is
   rotated by the same fixed np.rot90 as in the original pipeline (the affine
   is not inspected), resized to 256x256 with INTER_AREA, rounded with
   np.rint and saved as int16.

Output: <data_root>/<split>/1/isles_<subject>_<slice index>.npy

Each --roots folder must contain
    raw_data/<subject>/ses-01/<subject>_ses-01_ncct.nii.gz
    derivatives/<subject>/ses-02/<subject>_ses-02_lesion-msk.nii.gz
(the layout of the ISLES 2024 training release; the original run used its
two parts, train_a and train_b, as two roots in that order).
"""

from __future__ import annotations

import argparse
import os

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ctwin.layout import class_dir
from ctwin.preprocessing import resize_round_int16, save_array
from ctwin.slice_selection import MAX_SLICES_PER_PATIENT, above_median_top_k

SEED = 42
CLASS = "1"


def find_file_paths(subject: str, roots: list[str]) -> tuple[str | None, str | None]:
    """Return (ncct, lesion mask) of the first root that has both, else (None, None)."""
    for root in roots:
        ncct = os.path.join(root, "raw_data", subject, "ses-01", f"{subject}_ses-01_ncct.nii.gz")
        mask = os.path.join(root, "derivatives", subject, "ses-02", f"{subject}_ses-02_lesion-msk.nii.gz")
        if os.path.exists(ncct) and os.path.exists(mask):
            return ncct, mask
    return None, None


def list_subjects(roots: list[str]) -> list[str]:
    subjects: list[str] = []
    for root in roots:
        raw = os.path.join(root, "raw_data")
        if os.path.exists(raw):
            subjects += [d for d in os.listdir(raw) if d.startswith("sub-stroke")]
    return sorted(set(subjects))


def lesion_coverage_table(roots: list[str]) -> pd.DataFrame:
    """One row per lesion-bearing slice: Subject, Slice_Index, Coverage (%)."""
    rows = []
    for subject in list_subjects(roots):
        _, mask_path = find_file_paths(subject, roots)
        if not mask_path:
            continue
        try:  # as in the original: any failure skips the subject
            data = nib.load(mask_path).get_fdata()  # (X, Y, Z)
            slice_area = data.shape[0] * data.shape[1]
            for z in range(data.shape[2]):
                lesion_pixels = np.count_nonzero(data[:, :, z])
                if lesion_pixels > 0:
                    rows.append({"Subject": subject, "Slice_Index": z,
                                 "Coverage": lesion_pixels / slice_area * 100.0})
        except Exception as e:
            print(f"skipped {subject}: {e}")
    if not rows:
        raise ValueError("no lesion-bearing slice found")
    return pd.DataFrame(rows)


def split_subjects(selected: pd.DataFrame) -> dict[str, str]:
    """Map subject -> 'train' | 'val' | 'test' (70/10/20 at subject level)."""
    subjects = selected["Subject"].unique()
    if len(subjects) < 3:
        raise ValueError(f"only {len(subjects)} subject(s) left after slice selection; the split needs at least 3")
    train_val, test = train_test_split(subjects, test_size=0.20, random_state=SEED)
    train, val = train_test_split(train_val, test_size=0.125, random_state=SEED)
    split_map = {s: "train" for s in train}
    split_map.update({s: "val" for s in val})
    split_map.update({s: "test" for s in test})
    return split_map


def save_slices(selected: pd.DataFrame, split_map: dict[str, str], roots: list[str], data_root: str) -> None:
    ok = failed = 0  # as in the original, a failing subject counts all its selected slices as failed
    for subject, group in selected.groupby("Subject"):
        split = split_map.get(subject)
        if not split:
            continue
        ncct_path, _ = find_file_paths(subject, roots)
        if not ncct_path:
            failed += len(group)
            continue
        try:
            volume = nib.load(ncct_path).get_fdata()  # HU, (X, Y, Z)
            for _, row in group.iterrows():
                z = row["Slice_Index"]
                image = resize_round_int16(np.rot90(volume[:, :, z]))
                save_array(image, class_dir(data_root, split, CLASS), f"isles_{subject}_{z}")
                ok += 1
        except Exception as e:
            print(f"failed {subject}: {e}")
            failed += len(group)
    print(f"saved {ok} slices, {failed} failed")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--roots", nargs="+", required=True,
                   help="ISLES 2024 release folders, in order (each holds raw_data/ and derivatives/)")
    p.add_argument("--data-root", required=True, help="root of the curated dataset (output)")
    a = p.parse_args()
    missing = [r for r in a.roots if not os.path.isdir(os.path.join(r, "raw_data"))]
    for r in missing:  # the original skipped such roots silently
        print(f"warning: no raw_data/ folder in {r}; root ignored")
    if len(missing) == len(a.roots):
        p.error("none of the roots has a raw_data/ folder")
    for split in ("train", "val", "test"):  # created up front, as in the original
        os.makedirs(class_dir(a.data_root, split, CLASS), exist_ok=True)
    table = lesion_coverage_table(a.roots)
    selected, median = above_median_top_k(table, "Coverage", "Subject",
                                          sort_by=["Subject", "Coverage"], ascending=[True, False],
                                          k=MAX_SLICES_PER_PATIENT)
    print(f"{len(table)} lesion slices, median coverage {median:.4f}%, {len(selected)} selected")
    split_map = split_subjects(selected)
    for s in ("train", "val", "test"):
        print(f"{s}: {sum(v == s for v in split_map.values())} subjects")
    save_slices(selected, split_map, a.roots, a.data_root)


if __name__ == "__main__":
    main()
