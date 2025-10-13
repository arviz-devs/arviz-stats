"""Tests for array interface base functions."""

# pylint: disable=no-member, redefined-outer-name, no-self-use, protected-access
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from arviz_stats.base.array import BaseArray, process_ary_axes, process_chain_none


@pytest.fixture
def array_stats():
    return BaseArray()


class TestHelperFunctions:
    def test_process_chain_none_with_none(self):
        ary = np.random.randn(100, 50)
        ary_out, chain_axis, draw_axis = process_chain_none(ary, None, -1)
        assert ary_out.shape == (1, 100, 50)
        assert chain_axis == 0
        assert draw_axis == -1

    def test_process_chain_none_without_none(self):
        ary = np.random.randn(4, 100, 50)
        ary_out, chain_axis, draw_axis = process_chain_none(ary, 0, 1)
        assert ary_out.shape == ary.shape
        assert chain_axis == 0
        assert draw_axis == 1

    def test_process_chain_none_negative_draw_axis(self):
        ary = np.random.randn(100, 50)
        ary_out, chain_axis, draw_axis = process_chain_none(ary, None, -2)
        assert ary_out.shape == (1, 100, 50)
        assert chain_axis == 0
        assert draw_axis == -2

    @pytest.mark.parametrize("axis", [0, 1, -1, -2])
    def test_process_ary_axes_single_axis(self, axis):
        ary = np.random.randn(10, 20, 30)
        ary_out, axes = process_ary_axes(ary, axis)
        assert ary_out.ndim == ary.ndim
        assert axes[-1] == -1

    def test_process_ary_axes_multiple_axes(self):
        ary = np.random.randn(10, 20, 30, 40)
        ary_out, axes = process_ary_axes(ary, [0, 2])
        assert ary_out.ndim == ary.ndim
        assert len(axes) == 2
        assert_array_equal(axes, [-2, -1])

    def test_process_ary_axes_none(self):
        ary = np.random.randn(10, 20, 30)
        ary_out, axes = process_ary_axes(ary, None)
        assert ary_out.shape == ary.shape
        assert len(axes) == ary.ndim


class TestHDI:
    def test_hdi_basic(self, array_stats):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(100,))
        result = array_stats.hdi(ary, prob=0.9)
        assert result.shape == (2,)
        assert result[0] < result[1]

    @pytest.mark.parametrize("method", ["nearest", "multimodal", "multimodal_sample"])
    def test_hdi_methods(self, array_stats, method):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(100,))
        result = array_stats.hdi(ary, prob=0.9, method=method)
        if method.startswith("multimodal"):
            assert result.ndim == 2
            assert result.shape[-1] == 2
        else:
            assert result.shape == (2,)

    def test_hdi_invalid_prob(self, array_stats):
        ary = np.random.randn(100)
        with pytest.raises(ValueError, match="must be in the"):
            array_stats.hdi(ary, prob=1.5)

    def test_hdi_circular(self, array_stats):
        rng = np.random.default_rng(42)
        ary = rng.vonmises(0, 2, size=1000)
        result = array_stats.hdi(ary, prob=0.9, circular=True)
        assert result.shape == (2,)

    @pytest.mark.parametrize("axis", [0, 1, -1])
    def test_hdi_axis(self, array_stats, axis):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(10, 20, 30))
        result = array_stats.hdi(ary, prob=0.9, axis=axis)
        assert result.shape[-1] == 2


class TestESS:
    def test_ess_basic(self, array_stats):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(4, 100))
        result = array_stats.ess(ary)
        assert result.shape == ()
        assert result > 0

    @pytest.mark.parametrize("method", ["bulk", "tail", "mean", "sd", "median", "mad"])
    def test_ess_methods(self, array_stats, method):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(4, 100))
        prob = 0.95 if method == "tail" else None
        result = array_stats.ess(ary, method=method, prob=prob)
        assert result.shape == ()
        assert result > 0

    def test_ess_invalid_method(self, array_stats):
        ary = np.random.randn(4, 100)
        with pytest.raises(ValueError, match="must be one of"):
            array_stats.ess(ary, method="invalid")

    def test_ess_relative(self, array_stats):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(4, 100))
        result = array_stats.ess(ary, relative=True)
        assert 0 < result <= 1

    def test_ess_multiple_vars(self, array_stats):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(3, 4, 100))
        result = array_stats.ess(ary)
        assert result.shape == (3,)


class TestRhat:
    def test_rhat_basic(self, array_stats):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(4, 100))
        result = array_stats.rhat(ary)
        assert result.shape == ()
        assert result > 0

    @pytest.mark.parametrize("method", ["rank", "folded", "z_scale", "split", "identity"])
    def test_rhat_methods(self, array_stats, method):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(4, 100))
        result = array_stats.rhat(ary, method=method)
        assert result.shape == ()
        assert result > 0

    def test_rhat_invalid_method(self, array_stats):
        ary = np.random.randn(4, 100)
        with pytest.raises(ValueError, match="must be one of"):
            array_stats.rhat(ary, method="invalid")

    def test_rhat_multiple_vars(self, array_stats):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(3, 4, 100))
        result = array_stats.rhat(ary)
        assert result.shape == (3,)


class TestRhatNested:
    def test_rhat_nested_basic(self, array_stats):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(4, 100))
        superchain_ids = np.array([0, 0, 1, 1])
        result = array_stats.rhat_nested(ary, superchain_ids)
        assert result.shape == ()
        assert result > 0

    @pytest.mark.parametrize("method", ["rank", "folded", "z_scale", "split", "identity"])
    def test_rhat_nested_methods(self, array_stats, method):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(4, 100))
        superchain_ids = np.array([0, 0, 1, 1])
        result = array_stats.rhat_nested(ary, superchain_ids, method=method)
        assert result.shape == ()

    def test_rhat_nested_invalid_method(self, array_stats):
        ary = np.random.randn(4, 100)
        superchain_ids = np.array([0, 0, 1, 1])
        with pytest.raises(ValueError, match="must be one of"):
            array_stats.rhat_nested(ary, superchain_ids, method="invalid")


class TestMCSE:
    def test_mcse_basic(self, array_stats):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(4, 100))
        result = array_stats.mcse(ary)
        assert result.shape == ()
        assert result > 0

    @pytest.mark.parametrize("method", ["mean", "sd", "median", "quantile"])
    def test_mcse_methods(self, array_stats, method):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(4, 100))
        prob = 0.95 if method == "quantile" else None
        result = array_stats.mcse(ary, method=method, prob=prob)
        assert result.shape == ()
        assert result > 0

    def test_mcse_invalid_method(self, array_stats):
        ary = np.random.randn(4, 100)
        with pytest.raises(ValueError, match="must be one of"):
            array_stats.mcse(ary, method="invalid")


class TestParetoFunctions:
    def test_pareto_min_ss(self, array_stats):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(4, 100))
        result = array_stats.pareto_min_ss(ary)
        assert result.shape == ()
        assert result > 0

    def test_psislw_basic(self, array_stats):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(100,))
        log_weights, khat = array_stats.psislw(ary)
        assert log_weights.shape == (100,)
        assert khat.shape == ()

    def test_psislw_r_eff(self, array_stats):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(100,))
        log_weights, khat = array_stats.psislw(ary, r_eff=0.5)
        assert log_weights.shape == (100,)
        assert khat.shape == ()

    def test_pareto_khat(self, array_stats):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(4, 100))
        result = array_stats.pareto_khat(ary)
        assert result.shape == ()


class TestPowerScale:
    def test_power_scale_lw(self, array_stats):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(100,))
        result = array_stats.power_scale_lw(ary, alpha=0.5)
        assert result.shape == (100,)

    def test_power_scale_sense(self, array_stats):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(4, 100))
        lower_w = rng.normal(size=(4, 100))
        upper_w = rng.normal(size=(4, 100))
        result = array_stats.power_scale_sense(ary, lower_w, upper_w, -0.5, 0.5)
        assert result.shape == ()


class TestComputeRanks:
    def test_compute_ranks_basic(self, array_stats):
        x = np.array([3.0, 1.0, 2.0, 4.0])
        result = array_stats.compute_ranks(x)
        assert result.shape == (4,)

    def test_compute_ranks_relative(self, array_stats):
        x = np.array([3.0, 1.0, 2.0, 4.0])
        result = array_stats.compute_ranks(x, relative=True)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    @pytest.mark.parametrize("axis", [0, 1, -1])
    def test_compute_ranks_axis(self, array_stats, axis):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(10, 20, 30))
        result = array_stats.compute_ranks(ary, axis=axis)
        assert result.ndim == ary.ndim


class TestBinning:
    def test_get_bins_basic(self, array_stats):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(100,))
        bins = array_stats.get_bins(ary)
        assert bins.ndim == 1
        assert len(bins) > 0

    @pytest.mark.parametrize("bins", [10, 20, 50])
    def test_get_bins_integer(self, array_stats, bins):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(100,))
        result = array_stats.get_bins(ary, bins=bins)
        assert result.ndim == 1

    def test_histogram_basic(self, array_stats):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(1000,))
        counts, edges = array_stats.histogram(ary)
        assert len(counts) == len(edges) - 1
        assert counts.sum() == 1000

    @pytest.mark.parametrize("bins", [10, 20, 50])
    def test_histogram_bins(self, array_stats, bins):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(1000,))
        counts, _ = array_stats.histogram(ary, bins=bins)
        assert len(counts) == bins

    def test_histogram_density(self, array_stats):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(1000,))
        counts, edges = array_stats.histogram(ary, density=True)
        widths = np.diff(edges)
        integral = np.sum(counts * widths)
        assert_allclose(integral, 1.0, rtol=0.01)

    def test_histogram_range(self, array_stats):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(1000,))
        _, edges = array_stats.histogram(ary, bins=20, range=(-2, 2))
        assert_allclose(edges[0], -2)
        assert_allclose(edges[-1], 2)


class TestKDE:
    def test_kde_basic(self, array_stats):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(100,))
        grid, pdf, bw = array_stats.kde(ary)
        assert len(grid) == 512
        assert len(pdf) == 512
        assert bw > 0

    @pytest.mark.parametrize("grid_len", [128, 256, 512, 1024])
    def test_kde_grid_len(self, array_stats, grid_len):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(100,))
        grid, pdf, _ = array_stats.kde(ary, grid_len=grid_len)
        assert len(grid) == grid_len
        assert len(pdf) == grid_len

    def test_kde_circular(self, array_stats):
        rng = np.random.default_rng(42)
        ary = rng.vonmises(0, 2, size=100)
        grid, pdf, bw = array_stats.kde(ary, circular=True)
        assert len(grid) == 512
        assert len(pdf) == 512
        assert bw > 0


class TestQuantileDots:
    def test_qds_basic(self, array_stats):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(100,))
        x, y, radius = array_stats.qds(ary, nquantiles=20)
        assert len(x) == 20
        assert len(y) == 20
        assert radius > 0

    @pytest.mark.parametrize("nquantiles", [10, 20, 50, 100])
    def test_qds_nquantiles(self, array_stats, nquantiles):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(200,))
        x, y, _ = array_stats.qds(ary, nquantiles=nquantiles)
        assert len(x) == nquantiles
        assert len(y) == nquantiles


class TestECDF:
    def test_ecdf_basic(self, array_stats):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(100,))
        x, y = array_stats.ecdf(ary, npoints=50)
        assert len(x) == 50
        assert len(y) == 50
        assert np.all(y >= 0)
        assert np.all(y <= 1)

    def test_ecdf_monotonic(self, array_stats):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(100,))
        _, y = array_stats.ecdf(ary, npoints=50)
        assert np.all(np.diff(y) >= -1e-10)

    def test_ecdf_pit(self, array_stats):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(100,))
        x, y = array_stats.ecdf(ary, npoints=50, pit=True)
        assert len(x) == 50
        assert len(y) == 50


class TestThinning:
    def test_thin_factor_basic(self, array_stats):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(4, 100))
        factor = array_stats.thin_factor(ary)
        assert isinstance(factor, int)
        assert factor >= 1

    def test_thin_factor_target_ess(self, array_stats):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(4, 1000))
        factor = array_stats.thin_factor(ary, target_ess=100)
        assert isinstance(factor, int)
        assert factor >= 1

    @pytest.mark.parametrize("reduce_func", ["mean", "min"])
    def test_thin_factor_reduce_func(self, array_stats, reduce_func):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(4, 100))
        factor = array_stats.thin_factor(ary, reduce_func=reduce_func)
        assert isinstance(factor, int)
        assert factor >= 1

    def test_thin_factor_invalid_reduce_func(self, array_stats):
        ary = np.random.randn(4, 100)
        with pytest.raises(ValueError, match="not recognized"):
            array_stats.thin_factor(ary, reduce_func="invalid")

    def test_thin_auto(self, array_stats):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(4, 1000))
        result = array_stats.thin(ary, factor="auto")
        assert result.ndim == ary.ndim
        assert result.shape[0] == ary.shape[0]
        assert result.shape[1] <= ary.shape[1]

    @pytest.mark.parametrize("factor", [2, 5, 10])
    def test_thin_integer_factor(self, array_stats, factor):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(4, 1000))
        result = array_stats.thin(ary, factor=factor)
        assert result.shape[1] == ary.shape[1] // factor

    def test_thin_factor_one(self, array_stats):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(4, 100))
        result = array_stats.thin(ary, factor=1)
        assert_array_equal(result, ary)

    def test_thin_invalid_factor(self, array_stats):
        ary = np.random.randn(4, 100)
        with pytest.raises(ValueError, match="must be greater than 1"):
            array_stats.thin(ary, factor=0)


class TestMode:
    def test_mode_basic(self, array_stats):
        x = np.array([1, 2, 2, 2, 3, 3, 4])
        result = array_stats.mode(x)
        assert result == 2

    def test_mode_continuous(self, array_stats):
        rng = np.random.default_rng(42)
        x = rng.normal(5, 1, 1000)
        result = array_stats.mode(x)
        assert 3 < result < 7

    @pytest.mark.parametrize("axis", [0, 1, -1])
    def test_mode_axis(self, array_stats, axis):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(10, 20, 30))
        result = array_stats.mode(ary, axis=axis)
        assert result.ndim < ary.ndim


class TestMetrics:
    def test_r2_score_basic(self, array_stats):
        rng = np.random.default_rng(42)
        y_true = rng.normal(size=(100,))
        y_pred = rng.normal(size=(4, 100))
        result = array_stats.r2_score(y_true, y_pred)
        assert result.shape == (4,)
