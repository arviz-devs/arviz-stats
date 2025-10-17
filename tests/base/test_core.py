"""Tests for core base functions."""

# ruff: noqa: E731
# pylint: disable=no-member, redefined-outer-name, no-self-use, protected-access
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal

from arviz_stats.base.core import _CoreBase


@pytest.fixture
def core():
    return _CoreBase()


class TestFFT:
    def test_fft(self, core):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        result = core.fft(x)
        expected = np.fft.fft(x)
        assert_array_almost_equal(result, expected)

    @pytest.mark.parametrize("n", [5, 10, 20])
    def test_rfft(self, core, rng, n):
        x = rng.normal(size=5)
        result = core.rfft(x, n=n)
        expected = np.fft.rfft(x, n=n)
        assert_array_almost_equal(result, expected)

    @pytest.mark.parametrize("n", [5, 10, 20])
    def test_irfft(self, core, rng, n):
        x = rng.normal(size=5)
        fft_result = np.fft.rfft(x, n=n)
        result = core.irfft(fft_result, n=n)
        expected = np.fft.irfft(fft_result, n=n)
        assert_array_almost_equal(result, expected)

    @pytest.mark.parametrize("axis", [-1, 0, 1])
    def test_rfft_axis(self, core, rng, axis):
        data = rng.normal(size=(4, 10, 5))
        result = core.rfft(data, n=20, axis=axis)
        expected = np.fft.rfft(data, n=20, axis=axis)
        assert_array_almost_equal(result, expected)


class TestAutoCorrelation:
    @pytest.mark.parametrize("axis", [-1, 1, 2])
    def test_autocov_shape(self, core, rng, axis):
        x = rng.normal(size=(10, 20, 5))
        result = core.autocov(x, axis=axis)
        assert result.shape == x.shape

    def test_autocov_lag_zero(self, core, rng):
        x = rng.normal(size=(4, 100))
        result = core.autocov(x, axis=-1)
        for i in range(x.shape[0]):
            assert_allclose(result[i, 0], np.var(x[i, :]), rtol=0.01)

    @pytest.mark.parametrize("axis", [-1, 1, 2])
    def test_autocorr_shape(self, core, rng, axis):
        x = rng.normal(size=(10, 20, 5))
        result = core.autocorr(x, axis=axis)
        assert result.shape == x.shape

    def test_autocorr_lag_zero(self, core, rng):
        x = rng.normal(size=(4, 100))
        result = core.autocorr(x, axis=-1)
        assert_allclose(result[:, 0], 1.0, atol=1e-10)

    def test_autocorr_invalid_axis(self, core, rng):
        x = rng.normal(size=(5, 10))
        with pytest.raises(ValueError, match="Only integer values"):
            core.autocorr(x, axis=1.5)

    def test_autocov_invalid_axis(self, core, rng):
        x = rng.normal(size=(5, 10))
        with pytest.raises(ValueError, match="Only integer values"):
            core.autocov(x, axis=1.5)


class TestCircularStatistics:
    def test_circular_mean_zero(self, core):
        angles = np.array([0.1, -0.1, 0.05, -0.05])
        result = core.circular_mean(angles)
        assert_allclose(result, 0.0, atol=0.05)

    def test_circular_mean_pi_boundary(self, core):
        angles = np.array([np.pi - 0.1, -np.pi + 0.1])
        result = core.circular_mean(angles)
        assert abs(abs(result) - np.pi) < 0.15

    @pytest.mark.parametrize(
        "input_val,expected",
        [
            (0.0, 0.0),
            (np.pi / 2, np.pi / 2),
            (np.pi, -np.pi),
            (-np.pi / 2, -np.pi / 2),
            (2 * np.pi, 0.0),
            (3 * np.pi, -np.pi),
        ],
    )
    def test_circular_standardize(self, core, input_val, expected):
        result = core._circular_standardize(np.array([input_val]))
        assert_allclose(result[0], expected, atol=1e-10)

    def test_circular_standardize_range(self, core):
        angles = np.linspace(-4 * np.pi, 4 * np.pi, 100)
        result = core._circular_standardize(angles)
        assert np.all(result >= -np.pi)
        assert np.all(result < np.pi)


class TestQuantile:
    @pytest.mark.parametrize("q", [0.25, 0.5, 0.75, 0.9])
    def test_quantile(self, core, q):
        x = np.arange(1, 101, dtype=float)
        result = core.quantile(x, q)
        expected = np.quantile(x, q)
        assert_allclose(result, expected)

    def test_quantile_multiple(self, core):
        x = np.arange(1, 101, dtype=float)
        result = core.quantile(x, [0.25, 0.5, 0.75])
        expected = np.quantile(x, [0.25, 0.5, 0.75])
        assert_allclose(result, expected)

    @pytest.mark.parametrize("axis", [None, 0, 1, -1])
    def test_quantile_axis(self, core, rng, axis):
        x = rng.normal(size=(10, 20))
        result = core.quantile(x, 0.5, axis=axis)
        expected = np.quantile(x, 0.5, axis=axis)
        assert_allclose(result, expected)

    @pytest.mark.parametrize("skipna", [True, False])
    def test_quantile_nan(self, core, skipna):
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = core.quantile(x, 0.5, skipna=skipna)
        if skipna:
            assert_allclose(result, 3.0)
        else:
            assert np.isnan(result)

    @pytest.mark.parametrize("prob", [0.5, 0.9, 0.95])
    def test_eti(self, core, prob):
        x = np.arange(1, 101, dtype=float)
        result = core.eti(x, prob=prob)
        assert result.shape == (2,)
        assert result[0] < result[1]
        edge_prob = (1 - prob) / 2
        expected = np.quantile(x, [edge_prob, 1 - edge_prob])
        assert_allclose(result, expected)


class TestRanking:
    def test_float_rankdata(self, core):
        x = np.array([3.0, 1.0, 2.0, 4.0])
        result = core._float_rankdata(x)
        expected = np.array([3, 1, 2, 4])
        assert_array_equal(result, expected)

    def test_float_rankdata_range(self, core, rng):
        x = rng.normal(size=100)
        result = core._float_rankdata(x)
        assert_array_equal(np.sort(result), np.arange(1, 101))

    @pytest.mark.parametrize("relative", [True, False])
    def test_compute_ranks(self, core, relative):
        x = np.array([3.0, 1.0, 2.0, 4.0])
        result = core._compute_ranks(x, relative=relative)
        if relative:
            assert np.all(result >= 0)
            assert np.all(result <= 1)
        else:
            assert_array_equal(np.sort(result), [1, 2, 3, 4])

    def test_compute_ranks_discrete(self, core):
        x = np.array([3, 1, 2, 4], dtype=int)
        result = core._compute_ranks(x)
        assert result.dtype == np.int64


class TestBinning:
    def test_get_bininfo(self, core, rng):
        x = rng.normal(size=100)
        x_min, x_max, width = core._get_bininfo(x)
        assert x_min <= x.min()
        assert x_max >= x.max()
        assert width > 0

    @pytest.mark.parametrize("bins", [10, 20, 50])
    def test_get_bininfo_integer_bins(self, core, rng, bins):
        x = rng.normal(size=100)
        _, _, width = core._get_bininfo(x, bins=bins)
        expected_width = (x.max() - x.min()) / bins
        assert_allclose(width, expected_width, rtol=0.01)

    def test_get_bininfo_discrete(self, core, rng):
        x = rng.integers(0, 50, size=100)
        _, _, width = core._get_bininfo(x)
        assert width >= 1

    def test_get_bins_sorted(self, core, rng):
        x = rng.normal(size=100)
        bins = core._get_bins(x)
        assert_array_equal(bins, np.sort(bins))

    def test_get_bins_coverage(self, core, rng):
        x = rng.normal(size=100)
        bins = core._get_bins(x)
        assert bins[0] <= x.min()
        assert bins[-1] >= x.max()

    def test_histogram_counts(self, core, rng):
        x = rng.normal(size=1000)
        counts, edges = core._histogram(x)
        assert counts.sum() == 1000
        assert len(counts) == len(edges) - 1

    @pytest.mark.parametrize("bins", [10, 20, 50])
    def test_histogram_bins(self, core, rng, bins):
        x = rng.normal(size=1000)
        counts, _ = core._histogram(x, bins=bins)
        assert len(counts) == bins

    def test_histogram_density(self, core, rng):
        x = rng.normal(size=1000)
        counts, edges = core._histogram(x, density=True)
        widths = np.diff(edges)
        integral = np.sum(counts * widths)
        assert_allclose(integral, 1.0, rtol=0.01)

    def test_histogram_range(self, core, rng):
        x = rng.normal(size=1000)
        _, edges = core._histogram(x, bins=20, range=(-2, 2))
        assert_allclose(edges[0], -2)
        assert_allclose(edges[-1], 2)


class TestHDI:
    @pytest.mark.parametrize("prob", [0.5, 0.9, 0.95])
    def test_hdi_linear_nearest_common(self, core, prob):
        x = np.arange(100, dtype=float)
        result = core._hdi_linear_nearest_common(x, prob=prob)
        assert result.shape == (2,)
        assert result[0] < result[1]

    def test_hdi_nearest(self, core, rng):
        x = rng.normal(size=1000)
        result = core._hdi_nearest(x, prob=0.9, circular=False, skipna=False)
        assert result.shape == (2,)
        assert result[0] < result[1]

    @pytest.mark.parametrize("prob", [0.5, 0.9, 0.95])
    def test_hdi_nearest_circular(self, core, rng, prob):
        x = rng.vonmises(0, 2, size=1000)
        result = core._hdi_nearest(x, prob=prob, circular=True, skipna=False)
        assert result.shape == (2,)
        assert np.all(result >= -np.pi)
        assert np.all(result <= np.pi)

    def test_hdi_nearest_skipna(self, core, rng):
        x = np.concatenate([rng.normal(size=100), [np.nan] * 10])
        result = core._hdi_nearest(x, prob=0.9, circular=False, skipna=True)
        assert not np.any(np.isnan(result))


class TestMode:
    def test_mode_discrete(self, core):
        x = np.array([1, 2, 2, 2, 3, 3, 4])
        result = core._mode(x)
        assert result == 2

    def test_mode_continuous(self, core, rng):
        x = rng.normal(5, 1, 1000)
        result = core._mode(x)
        assert 3 < result < 7

    def test_mode_single_value(self, core):
        x = np.array([5.0])
        result = core._mode(x)
        assert result == 5.0

    def test_mode_empty(self, core):
        x = np.array([])
        result = core._mode(x)
        assert np.isnan(result)

    def test_mode_with_nan(self, core):
        x = np.array([1.0, 2.0, 2.0, np.nan, 2.0, 3.0])
        result = core._mode(x)
        assert not np.isnan(result)
