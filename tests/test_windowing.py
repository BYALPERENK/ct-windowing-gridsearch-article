import numpy as np

from ctwin.windowing import apply_window, raw_minmax_normalize, window_bounds


def test_bounds_of_brain_window():
    assert window_bounds(80, 40) == (0.0, 80.0)


def test_apply_window_clips_and_scales():
    hu = np.array([[-100, 0, 40], [80, 200, 20]], dtype=np.int16)
    out = apply_window(hu, 80, 40)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [[0.0, 0.0, 0.5], [1.0, 1.0, 0.25]])
    assert hu[0, 0] == -100  # input untouched


def test_raw_minmax_uses_global_extremes():
    hu = np.stack([np.full((2, 2), -1000), np.full((2, 2), 1000)]).astype(np.int16)
    out = raw_minmax_normalize(hu)
    assert out.min() == 0.0 and out.max() == 1.0
    np.testing.assert_allclose(out[0], 0.0)


def test_constant_input_does_not_divide_by_zero():
    out = raw_minmax_normalize(np.zeros((2, 2), dtype=np.int16))
    assert np.isfinite(out).all()
