"""Prepare the private clinical dataset (paper Sec. 3.1, Source 5).

The dataset is not publicly available; this script documents how it
was split. It never contributes to training. For each class a fixed-seed
random sample is drawn and sliced into the validation and test splits:

    class  folder                          sample  first part    second part
    0      normal / non-acute findings     1,001   test 667      val 334
    1      acute ischemic stroke             660   val 220       test 440
    2      hemorrhagic stroke                750   val 250       test 500

The slicing order (test first for class 0, validation first for classes 1 and
2) is that of the original scripts and decides which images go where. The
generator is re-seeded (42) before each class.

Output files are named pcd_<original id>.npy (see ctwin.layout).
"""

from __future__ import annotations

import argparse
import os

from ctwin.layout import class_dir
from ctwin.preprocessing import convert_dicom_files, find_dicoms, seeded_sample

PREFIX = "pcd_"
SEED = 42
# class -> [(split, count), ...] in slicing order
PLAN = {
    "0": [("test", 667), ("val", 334)],
    "1": [("val", 220), ("test", 440)],
    "2": [("val", 250), ("test", 500)],
}


def prepare_class(source_dir: str, cls: str, data_root: str) -> None:
    plan = PLAN[cls]
    total = sum(n for _, n in plan)
    selected = seeded_sample(find_dicoms(source_dir), total, SEED)
    if len(selected) < total:
        print(f"warning: class {cls}: only {len(selected)} DICOMs found, {total} expected; "
              "the later part(s) of the split will be short")
    start = 0
    for split, n in plan:
        part = selected[start:start + n]
        start += n
        ok, failed = convert_dicom_files(part, class_dir(data_root, split, cls), PREFIX)
        print(f"{split}/{cls}: {ok} saved, {len(failed)} failed (of {len(part)})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--normal-dir", required=True, help="class 0 DICOMs (normal and non-acute findings)")
    p.add_argument("--ischemic-dir", required=True, help="class 1 DICOMs (acute ischemic stroke)")
    p.add_argument("--hemorrhage-dir", required=True, help="class 2 DICOMs (hemorrhagic stroke)")
    p.add_argument("--data-root", required=True, help="root of the curated dataset (output)")
    a = p.parse_args()
    sources = {"0": a.normal_dir, "1": a.ischemic_dir, "2": a.hemorrhage_dir}
    for d in sources.values():
        if not os.path.isdir(d):
            p.error(f"not a directory: {d}")
    for cls, source in sources.items():
        prepare_class(source, cls, a.data_root)


if __name__ == "__main__":
    main()
