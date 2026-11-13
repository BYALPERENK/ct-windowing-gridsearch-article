"""Count the curated E-NCCT slices per split and class (paper Table 1).

Prints the split totals, a split x class table with the class share within
each split, the class totals, and how each class is distributed over the
splits. With --check-paper the counts are also compared with Table 1 of the
paper, and the exit code is 1 if any cell differs.
"""

from __future__ import annotations

import argparse
import os
import sys

from ctwin.layout import CLASSES, CLASS_NAMES, SPLITS

# Table 1 of the paper: slices per (split, class)
PAPER_TABLE_1 = {
    ("train", "0"): 3600, ("val", "0"): 534, ("test", "0"): 1067,
    ("train", "1"): 2176, ("val", "1"): 414, ("test", "1"): 763,
    ("train", "2"): 3500, ("val", "2"): 541, ("test", "2"): 1041,
}


def count_files(folder: str, ext: str = ".npy") -> int:
    """Number of regular files in folder with the given extension (0 if missing)."""
    if not os.path.isdir(folder):
        return 0
    return sum(1 for name in os.listdir(folder)
               if os.path.isfile(os.path.join(folder, name)) and name.lower().endswith(ext.lower()))


def pct(part: int, whole: int) -> float:
    return 100.0 * part / whole if whole else 0.0


def count_dataset(data_root: str) -> dict[tuple[str, str], int]:
    return {(s, c): count_files(os.path.join(data_root, s, c)) for s in SPLITS for c in CLASSES}


def report(counts: dict[tuple[str, str], int], data_root: str) -> None:
    split_totals = {s: sum(counts[(s, c)] for c in CLASSES) for s in SPLITS}
    class_totals = {c: sum(counts[(s, c)] for s in SPLITS) for c in CLASSES}
    total = sum(split_totals.values())

    print(f"data root: {data_root}\n")
    print("[1] slices per split")
    for s in SPLITS:
        print(f"  {s:6s}: {split_totals[s]:6d} ({pct(split_totals[s], total):5.2f}%)")
    print(f"  {'total':6s}: {total:6d}")

    print("\n[2] split x class (share of the class within the split)")
    for s in SPLITS:
        cells = " | ".join(f"c{c} {counts[(s, c)]:6d} ({pct(counts[(s, c)], split_totals[s]):5.2f}%)"
                           for c in CLASSES)
        print(f"  {s:6s} | {cells} | {split_totals[s]:6d}")

    print("\n[3] slices per class (share of the whole dataset)")
    for c in CLASSES:
        print(f"  class {c} ({CLASS_NAMES[c]:11s}): {class_totals[c]:6d} ({pct(class_totals[c], total):5.2f}%)")

    print("\n[4] distribution of each class over the splits (as in Table 1)")
    for c in CLASSES:
        cells = " | ".join(f"{s} {counts[(s, c)]:6d} ({pct(counts[(s, c)], class_totals[c]):5.2f}%)"
                           for s in SPLITS)
        print(f"  class {c} | {cells} | {class_totals[c]:6d}")

    missing = [os.path.join(data_root, s, c) for s in SPLITS for c in CLASSES
               if not os.path.isdir(os.path.join(data_root, s, c))]
    if missing:
        print("\n[!] missing folders:")
        for m in missing:
            print(f"  - {m}")


def check_paper(counts: dict[tuple[str, str], int]) -> bool:
    diffs = [(k, counts[k], v) for k, v in PAPER_TABLE_1.items() if counts[k] != v]
    if not diffs:
        print("\nall counts match Table 1 of the paper")
        return True
    print("\ncounts that differ from Table 1 of the paper:")
    for (s, c), got, want in diffs:
        print(f"  {s}/{c}: {got} (paper: {want})")
    return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True,
                   help="root of the curated dataset: <root>/{train,val,test}/{0,1,2}/*.npy")
    p.add_argument("--check-paper", action="store_true", help="compare with Table 1 of the paper")
    a = p.parse_args()
    if not os.path.isdir(a.data_root):
        p.error(f"not a directory: {a.data_root}")
    counts = count_dataset(a.data_root)
    report(counts, a.data_root)
    if a.check_paper and not check_paper(counts):
        sys.exit(1)


if __name__ == "__main__":
    main()
