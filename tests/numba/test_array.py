# pylint: disable=redefined-outer-name, unused-import, no-self-use, protected-access
# ruff: noqa: F811, F401
import numpy as np
import pytest
from numpy.testing import assert_allclose

from ..helpers import importorskip

importorskip("numba")

from arviz_stats.numba.array import NumbaArray, _histogram_jit, _quantile_ufunc, process_ary_axes


class TestProcessAryAxes:
    def test_process_ary_axes_single_axis(self):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(3, 4, 5))
        result, axes = process_ary_axes(ary, -1)
        assert result.shape == (3, 4, 5)
        assert axes == [2]

    def test_process_ary_axes_multiple_axes(self):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(3, 4, 5))
        result, axes = process_ary_axes(ary, [1, 2])
        assert result.shape == (3, 20)
        assert axes == [1, 2]

    def test_process_ary_axes_negative_index(self):
        rng = np.random.default_rng(42)
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
    def test_histogram_jit_basic(self):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=100)
        hist, edges = _histogram_jit(ary, 10)
        assert len(hist) == 10
        assert len(edges) == 11
        assert np.sum(hist) == 100

    def test_histogram_jit_with_range(self):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=100)
        hist, edges = _histogram_jit(ary, 10, range=(-3, 3))
        assert len(hist) == 10
        assert len(edges) == 11
        assert edges[0] == -3
        assert edges[-1] == 3


class TestNumbaArray:
    def test_quantile_linear(self):
        array_stats = NumbaArray()
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(4, 100))
        result = array_stats.quantile(ary, np.array([0.5]), axis=-1)
        assert result.shape == (4, 1)

    def test_quantile_multiple_quantiles(self):
        array_stats = NumbaArray()
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(4, 100))
        result = array_stats.quantile(ary, [0.25, 0.5, 0.75], axis=-1)
        assert result.shape == (4, 3)

    def test_quantile_fallback_non_linear(self):
        array_stats = NumbaArray()
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(4, 100))
        result = array_stats.quantile(ary, 0.5, axis=-1, method="midpoint")
        assert result.shape == (4,)

    def test_histogram_basic(self):
        array_stats = NumbaArray()
        rng = np.random.default_rng(42)
        ary = rng.normal(size=100)
        hist, edges = array_stats._histogram(ary, bins=10)
        assert len(hist) == 10
        assert len(edges) == 11

    def test_histogram_density(self):
        array_stats = NumbaArray()
        rng = np.random.default_rng(42)
        ary = rng.normal(size=100)
        hist, edges = array_stats._histogram(ary, bins=10, density=True)
        bin_width = edges[1] - edges[0]
        assert_allclose(np.sum(hist) * bin_width, 1.0, rtol=0.01)

    def test_histogram_weights_not_supported(self):
        array_stats = NumbaArray()
        rng = np.random.default_rng(42)
        ary = rng.normal(size=100)
        weights = np.ones(100)
        with pytest.raises(NotImplementedError, match="weights"):
            array_stats._histogram(ary, bins=10, weights=weights)

    def test_kde_basic(self):
        array_stats = NumbaArray()
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(4, 100))
        grid, pdf, bw = array_stats.kde(ary, axis=-1, grid_len=256)  # pylint: disable=unpacking-non-sequence
        assert grid.shape == (4, 256)
        assert pdf.shape == (4, 256)
        assert bw.shape == (4,)

    def test_kde_ufunc_caching(self):
        array_stats = NumbaArray()
        assert array_stats._kde_ufunc is None
        _ = array_stats.kde_ufunc
        assert array_stats._kde_ufunc is not None
        ufunc1 = array_stats.kde_ufunc
        ufunc2 = array_stats.kde_ufunc
        assert ufunc1 is ufunc2
