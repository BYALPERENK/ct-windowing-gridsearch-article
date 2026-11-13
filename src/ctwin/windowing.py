"""Windowing and intensity normalization applied at load time (paper Sec. 3.2).

For a window of width WW and center WC, the bounds are

    L = WC - WW / 2,    H = WC + WW / 2,

HU values are clipped to [L, H] and min-max scaled to [0, 1]:

    X_norm = (CLIP(X, L, H) - L) / (H - L).

Experiment A additionally uses a RAW condition without a window: the minimum
and maximum HU value over the whole split are taken, and every slice in that
split is scaled to [0, 1] against that single pair of constants.
"""

from __future__ import annotations

import numpy as np


def window_bounds(ww: float, wc: float) -> tuple[float, float]:
    """Return the (L, H) bounds of a window."""
    return wc - ww / 2.0, wc + ww / 2.0


def apply_window(hu_images: np.ndarray, ww: float, wc: float) -> np.ndarray:
    """Clip HU values to the window and scale them to [0, 1] (float32).

    Works on a single slice or on a stack of slices; the input is not modified.
    """
    lo, hi = window_bounds(ww, wc)
    out = hu_images.astype(np.float32)
    np.clip(out, lo, hi, out=out)
    out -= lo
    denom = hi - lo
    if denom == 0:
        denom = 1e-8
    out /= denom
    return out


def raw_minmax_normalize(hu_images: np.ndarray) -> np.ndarray:
    """RAW condition of Experiment A: scale a whole split to [0, 1] (float32)
    with the split's global HU minimum and maximum.

    The constants come from the array that is passed in, so call this once per
    split (training, validation, test), as in the original experiment.
    """
    out = hu_images.astype(np.float32)
    gmin = out.min()
    gmax = out.max()
    denom = gmax - gmin
    if denom == 0:
        denom = 1e-8
    out = (out - gmin) / denom
    return out
