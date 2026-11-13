"""Lesion-based slice selection for the 3D ischemic sources (paper Sec. 3.1.1).

For AISD and ISLES 2024 the lesion coverage of every lesion-bearing slice is
computed, the slices above the median coverage of the source are kept, and at
most five slices per patient (those with the largest coverage) are retained.
"""

from __future__ import annotations

import pandas as pd

MAX_SLICES_PER_PATIENT = 5


def above_median_top_k(
    df: pd.DataFrame,
    coverage_col: str,
    patient_col: str,
    sort_by: str | list[str],
    ascending: bool | list[bool],
    k: int = MAX_SLICES_PER_PATIENT,
) -> tuple[pd.DataFrame, float]:
    """Keep the rows strictly above the median coverage, then the first k per patient.

    The sort keys are passed in by the caller because the original scripts
    differ per source (ISLES sorts by patient then coverage, AISD by coverage
    only), and with tied coverage values the sort decides which slices are kept.
    The pandas defaults of the original scripts are used unchanged.
    Returns (selected rows, median coverage).
    """
    median = df[coverage_col].median()
    kept = df[df[coverage_col] > median].copy()
    if kept.empty:
        raise ValueError("no slice lies strictly above the median coverage")
    kept = kept.sort_values(by=sort_by, ascending=ascending)
    return kept.groupby(patient_col).head(k), median
