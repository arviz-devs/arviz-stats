"""Test array functions."""

# pylint: disable=redefined-outer-name, no-self-use, protected-access
import numpy as np
import pytest
from numpy.testing import assert_allclose

from ..helpers import importorskip

importorskip("numba")

from arviz_stats.numba.array import NumbaArray, _histogram_jit, _quantile_ufunc, process_ary_axes


class TestProcessAryAxes:
    def test_process_ary_axes_single_axis(self, rng):
        ary = rng.normal(size=(3, 4, 5))
        result, axes = process_ary_axes(ary, -1)
        assert result.shape == (3, 4, 5)
        assert axes == [2]

    def test_process_ary_axes_multiple_axes(self, rng):
        ary = rng.normal(size=(3, 4, 5))
        result, axes = process_ary_axes(ary, [1, 2])
        assert result.shape == (3, 20)
        assert axes == [1, 2]

    def test_process_ary_axes_negative_index(self, rng):
        ary = rng.normal(size=(3, 4, 5))
        result, axes = process_ary_axes(ary, -2)
        assert result.shape == (3, 5, 4)
        assert axes == [1]


class TestQuantileUfunc:
    def test_quantile_ufunc_basic(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        q = np.array([0.25, 0.5, 0.75])
        out = np.zeros(3)
        _quantile_ufunc(a, q, out)
        assert out.shape == (3,)
        assert_allclose(out[1], 3.0, rtol=0.01)

    def test_quantile_ufunc_single_quantile(self):
        a = np.arange(100, dtype=float)
        q = np.array([0.5])
        out = np.zeros(1)
        _quantile_ufunc(a, q, out)
        assert_allclose(out[0], 49.5, rtol=0.01)


class TestHistogramJit:
    def test_histogram_jit_basic(self, rng):
        ary = rng.normal(size=100)
        hist, edges = _histogram_jit(ary, 10)
        assert len(hist) == 10
        assert len(edges) == 11
        assert np.sum(hist) == 100

    def test_histogram_jit_with_range(self, rng):
        ary = rng.normal(size=100)
        hist, edges = _histogram_jit(ary, 10, range=(-3, 3))
        assert len(hist) == 10
        assert len(edges) == 11
        assert edges[0] == -3
        assert edges[-1] == 3


class TestNumbaArray:
    def test_quantile_linear(self, rng):
        array_stats = NumbaArray()
        ary = rng.normal(size=(4, 100))
        result = array_stats.quantile(ary, np.array([0.5]), axis=-1)
        assert result.shape == (4, 1)

    def test_quantile_multiple_quantiles(self, rng):
        array_stats = NumbaArray()
        ary = rng.normal(size=(4, 100))
        result = array_stats.quantile(ary, [0.25, 0.5, 0.75], axis=-1)
        assert result.shape == (4, 3)

    @pytest.mark.parametrize("axis", [0, 1, -1, -2])
    def test_quantile_axis(self, rng, axis):
        array_stats = NumbaArray()
        ary = rng.normal(size=(10, 20, 30))
        result = array_stats.quantile(ary, np.array([0.5]), axis=axis)
        expected_shape = {
            0: (20, 30, 1),
            1: (10, 30, 1),
            -1: (10, 20, 1),
            -2: (10, 30, 1),
        }
        assert result.shape == expected_shape[axis]

    @pytest.mark.parametrize("axis", [0, 1, -1])
    def test_quantile_axis_multiple_quantiles(self, rng, axis):
        array_stats = NumbaArray()
        ary = rng.normal(size=(10, 20, 30))
        result = array_stats.quantile(ary, [0.25, 0.5, 0.75], axis=axis)
        expected_shape = {
            0: (20, 30, 3),
            1: (10, 30, 3),
            -1: (10, 20, 3),
        }
        assert result.shape == expected_shape[axis]

    def test_quantile_fallback_non_linear(self, rng):
        array_stats = NumbaArray()
        ary = rng.normal(size=(4, 100))
        result = array_stats.quantile(ary, 0.5, axis=-1, method="midpoint")
        assert result.shape == (4,)

    def test_histogram_basic(self, rng):
        array_stats = NumbaArray()
        ary = rng.normal(size=100)
        hist, edges = array_stats._histogram(ary, bins=10)
        assert len(hist) == 10
        assert len(edges) == 11

    def test_histogram_density(self, rng):
        array_stats = NumbaArray()
        ary = rng.normal(size=100)
        hist, edges = array_stats._histogram(ary, bins=10, density=True)
        bin_width = edges[1] - edges[0]
        assert_allclose(np.sum(hist) * bin_width, 1.0, rtol=0.01)

    def test_histogram_weights_not_supported(self, rng):
        array_stats = NumbaArray()
        ary = rng.normal(size=100)
        weights = np.ones(100)
        with pytest.raises(NotImplementedError, match="weights"):
            array_stats._histogram(ary, bins=10, weights=weights)

    def test_kde_basic(self, rng):
        array_stats = NumbaArray()
        ary = rng.normal(size=(4, 100))
        grid, pdf, bw = array_stats.kde(ary, axis=-1, grid_len=256)  # pylint: disable=unpacking-non-sequence
        assert grid.shape == (4, 256)
        assert pdf.shape == (4, 256)
        assert bw.shape == (4,)

    @pytest.mark.parametrize("axis", [0, 1, -1, -2])
    def test_kde_axis(self, rng, axis):
        array_stats = NumbaArray()
        ary = rng.normal(size=(10, 20, 30))
        grid, pdf, bw = array_stats.kde(ary, axis=axis, grid_len=128)  # pylint: disable=unpacking-non-sequence
        expected_shape = {
            0: (20, 30, 128),
            1: (10, 30, 128),
            -1: (10, 20, 128),
            -2: (10, 30, 128),
        }
        expected_bw_shape = {
            0: (20, 30),
            1: (10, 30),
            -1: (10, 20),
            -2: (10, 30),
        }
        assert grid.shape == expected_shape[axis]
        assert pdf.shape == expected_shape[axis]
        assert bw.shape == expected_bw_shape[axis]

    def test_kde_ufunc_caching(self):
        array_stats = NumbaArray()
        assert array_stats._kde_ufunc is None
        _ = array_stats.kde_ufunc
        assert array_stats._kde_ufunc is not None
        ufunc1 = array_stats.kde_ufunc
        ufunc2 = array_stats.kde_ufunc
        assert ufunc1 is ufunc2
