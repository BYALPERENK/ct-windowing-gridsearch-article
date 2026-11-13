import math

import numpy as np
import pytest

from ctwin.stats import adjust, paired_comparison, spearman


def test_paired_comparison_matches_hand_computation():
    x = np.array([0.70, 0.72, 0.69, 0.71, 0.73])
    y = np.array([0.60, 0.65, 0.61, 0.66, 0.62])
    r = paired_comparison(x, y)
    d = x - y
    sd = d.std(ddof=1)
    assert r["delta"] == pytest.approx(d.mean())
    assert r["dz"] == pytest.approx(d.mean() / sd)
    assert r["ci_high"] - r["delta"] == pytest.approx(2.7764451 * sd / math.sqrt(5), rel=1e-6)
    assert 0 < r["p"] < 0.01
    assert r["p_wilcoxon"] == pytest.approx(0.0625)  # smallest attainable two-sided p at n = 5


def test_adjust_bh_and_bonferroni():
    bh, bonf = adjust([0.01, 0.04, 0.03, 0.5])
    np.testing.assert_allclose(bh, [0.04, 0.0533333, 0.0533333, 0.5], rtol=1e-5)
    np.testing.assert_allclose(bonf, [0.04, 0.16, 0.12, 1.0])


def test_spearman_of_monotone_surfaces():
    a = np.arange(12.0).reshape(3, 4)
    assert spearman(a, a ** 2) == pytest.approx(1.0)
    assert spearman(a, -a) == pytest.approx(-1.0)
