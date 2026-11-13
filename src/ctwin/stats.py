"""Seed-matched (paired) comparisons and multiple-comparison adjustment
(paper Sec. 3.6).

Every comparison pairs the five per-seed values of two conditions. The
primary test is the two-sided paired t-test; the effect size is Cohen's d_z
(mean paired difference / its SD), the 95% CI is mean +- t_{0.975, n-1} SD/sqrt(n),
and the Wilcoxon signed-rank (SciPy defaults; exact for five non-zero, untied
differences) and Shapiro-Wilk tests of the differences are reported alongside.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests


def paired_comparison(x, y) -> dict:
    """Compare x against y, seed by seed (difference = x - y).

    When all differences are equal (zero variance) the test statistics are
    undefined: p, d_z, and the Wilcoxon and Shapiro-Wilk p-values are NaN, and
    adjust() refuses a family that contains such a comparison.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 3:
        raise ValueError("x and y must be 1-D arrays of equal length >= 3")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("paired_comparison needs finite values")
    d = x - y
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    half = float(stats.t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    degenerate = sd == 0
    out = {
        "delta": mean,
        "sd": sd,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "dz": math.nan if degenerate else mean / sd,
        "p": math.nan if degenerate else float(stats.ttest_rel(x, y).pvalue),
        "p_wilcoxon": math.nan if np.all(d == 0) else float(stats.wilcoxon(d).pvalue),
        "p_shapiro": math.nan if degenerate else float(stats.shapiro(d).pvalue),
    }
    return out


def adjust(pvalues) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg and Bonferroni adjusted p-values over one family.

    Raises if any p-value is missing, so that an undefined comparison never
    silently changes the family.
    """
    p = np.asarray(pvalues, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("every p-value of a family must be a number in [0, 1]")
    return (multipletests(p, method="fdr_bh")[1], multipletests(p, method="bonferroni")[1])


def spearman(a, b) -> float:
    """Spearman rank correlation (average ranks for ties) of two flattened surfaces."""
    a, b = np.ravel(a), np.ravel(b)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        raise ValueError("Spearman correlation of a constant surface is undefined")
    return float(stats.spearmanr(a, b).statistic)
