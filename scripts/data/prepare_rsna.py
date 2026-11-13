"""Prepare the RSNA 2019 Brain CT Hemorrhage data (paper Sec. 3.1, Source 4).

Class 0 (normal): patients whose metadata entry has has_hemorrhage == false.
    The patient IDs are sorted, random.sample draws 2,200 with seed 42, and
    the sample is sliced into train 1,600 / test 400 / val 200. One slice per
    patient is drawn with weights from a normal curve over the patient's slice
    list (mean 0.5, sd 0.13 on a 0-1 scale), which favours mid-level slices.
Class 2 (hemorrhagic): patients with has_hemorrhage == true and a non-empty
    "intraparenchymal" slice list. Sorted by ID; if there are more than 3,250,
    random.sample draws 3,250, otherwise the sorted list is used as it is
    (not shuffled). The result is sliced into train 2,500 / test 500 /
    val 250 (the last split takes the remainder). One slice is drawn from the
    intraparenchymal slices, in the order of that list, with a narrower
    curve (sd 0.10).

Each class re-seeds `random` and `numpy.random` (42). Slices are drawn with
np.random.choice patient by patient in the order train, test, val, before the
DICOM is read (a patient with a single candidate slice draws nothing), so that
order fixes which slice every patient gets. A patient whose DICOM then fails
is skipped, not replaced. The chosen
DICOM goes through the common curation pipeline (ctwin.preprocessing).

Output: <data_root>/<split>/0/kaggle_n_<patient>.npy and
        <data_root>/<split>/2/kaggle_k_<patient>.npy

Inputs
    --metadata   patient metadata JSON (format below)
    --dicom-dir  the stage_2_train folder of the RSNA release

Metadata JSON. The script that built it from the RSNA labels and DICOM
headers was not kept, so the file is not part of this repository. Its format,
as read here, is

    {
      "patient: <PatientID>": {
        "has_hemorrhage": true | false,
        "slices": {"<slice number>": "<DICOM file name>", ...},
        "diagnoses": {"intraparenchymal": [<slice number>, ...], ...}
      },
      ...
    }

The key order of "slices" is used as it is: the weights of class 0 are laid
over it, so the file is expected to list the slices in anatomical order (the
lost builder is assumed to have done so; this script does not check it). For
class 2 the order of the "intraparenchymal" list is used instead; its entries,
converted with str(), must match keys of "slices". has_hemorrhage is expected
to be false only when no slice of the patient carries any of the five subtype
labels or the "any" label.
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
from scipy.stats import norm

from ctwin.layout import class_dir
from ctwin.preprocessing import dicom_file_to_array, save_array

SEED = 42
NORMAL_SPLITS = (("train", 1600), ("test", 400), ("val", 200))
HEMORRHAGE_SPLITS = (("train", 2500), ("test", 500), ("val", 250))
NORMAL_SD = 0.13
HEMORRHAGE_SD = 0.10
PATIENT_KEY_PREFIX = "patient: "


def bell_curve_pick(candidates: list[str], sd: float) -> str:
    """Draw one item, weighting positions by a normal curve centred mid-list.

    With a single candidate no random number is drawn (as in the original).
    """
    n = len(candidates)
    if n == 1:
        return candidates[0]
    weights = norm.pdf(np.linspace(0, 1, n), loc=0.5, scale=sd)
    weights = weights / weights.sum()
    return candidates[np.random.choice(range(n), p=weights)]


def slice_split(items: list, plan: tuple[tuple[str, int], ...]) -> list[tuple[str, list]]:
    """Slice items in the order of plan; the last split takes the remainder."""
    parts, start = [], 0
    for i, (split, n) in enumerate(plan):
        end = len(items) if i == len(plan) - 1 else start + n
        parts.append((split, items[start:end]))
        start = end
    return parts


def save_one(dcm_name: str, dicom_dir: str, dest_dir: str, name: str) -> None:
    save_array(dicom_file_to_array(os.path.join(dicom_dir, dcm_name)), dest_dir, name)


def prepare_normal(metadata: dict, dicom_dir: str, data_root: str) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    patients = [key.replace(PATIENT_KEY_PREFIX, "") for key, info in metadata.items()
                if info.get("has_hemorrhage") is False]
    needed = sum(n for _, n in NORMAL_SPLITS)
    print(f"class 0: {len(patients)} hemorrhage-free patients")
    if len(patients) < needed:
        print(f"class 0 skipped: {needed} patients needed")  # the original stopped here too
        return
    patients.sort()
    selected = random.sample(patients, needed)
    for split, part in slice_split(selected, NORMAL_SPLITS):
        dest = class_dir(data_root, split, "0")
        os.makedirs(dest, exist_ok=True)
        ok = failed = 0
        for pid in part:
            try:
                slices = metadata[f"{PATIENT_KEY_PREFIX}{pid}"]["slices"]
                dcm_name = slices[bell_curve_pick(list(slices.keys()), NORMAL_SD)]
                save_one(dcm_name, dicom_dir, dest, f"kaggle_n_{pid}")
                ok += 1
            except Exception:
                failed += 1
        print(f"{split}/0: {ok} saved, {failed} failed (of {len(part)})")


def prepare_hemorrhage(metadata: dict, dicom_dir: str, data_root: str) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    patients = []
    for key, info in metadata.items():
        if info.get("has_hemorrhage") is True:
            ich_slices = info["diagnoses"].get("intraparenchymal", [])
            if ich_slices:
                patients.append({"id": key.replace(PATIENT_KEY_PREFIX, ""),
                                 "slices": ich_slices, "all_slices": info["slices"]})
    needed = sum(n for _, n in HEMORRHAGE_SPLITS)
    print(f"class 2: {len(patients)} patients with intraparenchymal hemorrhage")
    if len(patients) < needed:
        print(f"warning: {needed} wanted; all of them are used")  # the original went on as well
    patients.sort(key=lambda p: p["id"])
    selected = random.sample(patients, needed) if len(patients) > needed else patients
    for split, part in slice_split(selected, HEMORRHAGE_SPLITS):
        dest = class_dir(data_root, split, "2")
        os.makedirs(dest, exist_ok=True)
        ok = failed = 0
        for patient in part:
            try:
                valid = [str(s) for s in patient["slices"] if str(s) in patient["all_slices"]]
                if not valid:
                    raise ValueError("no intraparenchymal slice found in the slice list")
                dcm_name = patient["all_slices"][bell_curve_pick(valid, HEMORRHAGE_SD)]
                if not dcm_name:
                    raise ValueError("empty DICOM name in the metadata")
                save_one(dcm_name, dicom_dir, dest, f"kaggle_k_{patient['id']}")
                ok += 1
            except Exception:
                failed += 1
        print(f"{split}/2: {ok} saved, {failed} failed (of {len(part)})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata", required=True, help="patient metadata JSON (see the format above)")
    p.add_argument("--dicom-dir", required=True, help="RSNA stage_2_train folder")
    p.add_argument("--data-root", required=True, help="root of the curated dataset (output)")
    p.add_argument("--classes", nargs="+", choices=["0", "2"], default=["0", "2"],
                   help="which classes to prepare (default: both)")
    a = p.parse_args()
    if not os.path.isfile(a.metadata):
        p.error(f"metadata file not found: {a.metadata}")
    if not os.path.isdir(a.dicom_dir):
        p.error(f"not a directory: {a.dicom_dir}")
    with open(a.metadata, "r") as f:
        metadata = json.load(f)
    if "0" in a.classes:
        prepare_normal(metadata, a.dicom_dir, a.data_root)
    if "2" in a.classes:
        prepare_hemorrhage(metadata, a.dicom_dir, a.data_root)


if __name__ == "__main__":
    main()
