"""Tests for array interface base functions."""

# pylint: disable=redefined-outer-name, no-self-use, protected-access
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from arviz_stats import loo
from arviz_stats.base.array import BaseArray, process_ary_axes, process_chain_none
from arviz_stats.loo.helper_loo import _get_r_eff, _prepare_loo_inputs
from arviz_stats.utils import get_log_likelihood_dataset

from ..helpers import importorskip

einstats = importorskip("xarray_einstats")


@pytest.fixture
def array_stats():
    return BaseArray()


class TestHelperFunctions:
    def test_process_chain_none_with_none(self):
        ary = np.empty((100, 50))
        ary_out, chain_axis, draw_axis = process_chain_none(ary, None, -1)
        assert ary_out.shape == (1, 100, 50)
        assert chain_axis == 0
        assert draw_axis == -1

    def test_process_chain_none_without_none(self):
        ary = np.empty((4, 100, 50))
        ary_out, chain_axis, draw_axis = process_chain_none(ary, 0, 1)
        assert ary_out.shape == ary.shape
        assert chain_axis == 0
        assert draw_axis == 1

    def test_process_chain_none_negative_draw_axis(self):
        ary = np.empty((100, 50))
        ary_out, chain_axis, draw_axis = process_chain_none(ary, None, -2)
        assert ary_out.shape == (1, 100, 50)
        assert chain_axis == 0
        assert draw_axis == -2

    @pytest.mark.parametrize("axis", [0, 1, -1, -2])
    def test_process_ary_axes_single_axis(self, axis):
        ary = np.empty((10, 20, 30))
        ary_out, axes = process_ary_axes(ary, axis)
        assert ary_out.ndim == ary.ndim
        assert len(axes) == 1
        assert axes[-1] == -1
        assert ary.shape[axis] == ary_out.shape[-1]

    def test_process_ary_axes_multiple_axes(self):
        ary = np.empty((10, 20, 30, 40))
        ary_out, axes = process_ary_axes(ary, [0, 2])
        assert ary_out.ndim == ary.ndim
        assert ary_out.shape == (20, 40, 10, 30)
        assert len(axes) == 2
        assert_array_equal(axes, [-2, -1])

    def test_process_ary_axes_none(self):
        ary = np.empty((10, 20, 30))
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
        ary = np.empty((100,))
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
        expected_shape = {
            0: (20, 30, 2),
            1: (10, 30, 2),
            -1: (10, 20, 2),
        }
        assert result.shape == expected_shape[axis]


class TestESS:
    def test_ess_basic(self, array_stats):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(4, 100))
        result = array_stats.ess(ary)
        assert result.shape == ()
        assert result > 0

    @pytest.mark.parametrize(
        "method",
        [
            "bulk",
            "tail",
            "mean",
            "sd",
            "median",
            "mad",
            "z_scale",
            "folded",
            "identity",
            "quantile",
            "local",
        ],
    )
    def test_ess_methods(self, array_stats, method):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(4, 100))
        if method == "local":
            prob = [0.025, 0.975]
        elif method in ("tail", "quantile"):
            prob = 0.95
        else:
            prob = None
        result = array_stats.ess(ary, method=method, prob=prob)
        assert result.shape == ()
        assert result > 0

    def test_ess_invalid_method(self, array_stats):
        ary = np.empty((4, 100))
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

    def test_ess_chain_draw_axis(self, array_stats):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(5, 4, 10, 100))
        result = array_stats.ess(ary, chain_axis=1, draw_axis=3)
        assert result.shape == (5, 10)
        assert result.min() > 0


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
        ary = np.empty((4, 100))
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
        ary = np.empty((4, 100))
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
        ary = np.empty((4, 100))
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

    def test_pareto_min_ss_chain_draw_axis(self, array_stats):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(5, 4, 10, 100))
        result = array_stats.pareto_min_ss(ary, chain_axis=1, draw_axis=3)
        assert result.shape == (5, 10)
        assert result.min() > 0

    @pytest.mark.parametrize("axis", [0, 1, -1])
    def test_psislw_axis_integer(self, array_stats, axis):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(50, 60, 70))
        log_weights, khat = array_stats.psislw(ary, axis=axis)
        assert log_weights.ndim == ary.ndim
        assert khat.shape == tuple(s for i, s in enumerate(ary.shape) if i != axis % ary.ndim)

    def test_psislw_axis_list(self, array_stats):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(50, 60, 70, 80))
        log_weights, khat = array_stats.psislw(ary, axis=[1, 3])
        assert log_weights.ndim == ary.ndim
        assert khat.shape == (50, 70)

    def test_pareto_khat_chain_draw_axis(self, array_stats):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(5, 4, 10, 100))
        result = array_stats.pareto_khat(ary, chain_axis=1, draw_axis=3)
        assert result.shape == (5, 10)


class TestPowerScale:
    def test_power_scale_lw(self, array_stats, rng):
        ary = rng.normal(size=(100,))
        result = array_stats.power_scale_lw(ary, alpha=0.5)
        assert result.shape == (100,)

    @pytest.mark.parametrize("axis", [0, 1, -1, -2])
    @pytest.mark.filterwarnings("ignore:Number of tail draws cannot be less than 5:UserWarning")
    def test_power_scale_lw_axis(self, array_stats, rng, axis):
        ary = rng.normal(size=(10, 20, 30))
        result = array_stats.power_scale_lw(ary, alpha=0.5, axis=axis)
        assert result.ndim == ary.ndim
        expected_shape = {
            0: (20, 30, 10),
            1: (10, 30, 20),
            -1: (10, 20, 30),
            -2: (10, 30, 20),
        }
        assert result.shape == expected_shape[axis]

    @pytest.mark.filterwarnings("ignore:Number of tail draws cannot be less than 5:UserWarning")
    def test_power_scale_lw_axis_list(self, array_stats, rng):
        ary = rng.normal(size=(10, 20, 30, 40))
        result = array_stats.power_scale_lw(ary, alpha=0.5, axis=[1, 3])
        assert result.ndim == ary.ndim
        assert result.shape == (10, 30, 20, 40)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.filterwarnings("ignore:Number of tail draws cannot be less than 5:UserWarning")
    def test_power_scale_sense(self, array_stats, rng):
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
        assert_array_equal(result, np.array([3.0, 1.0, 2.0, 4.0]))

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
        assert result.max() == ary.shape[axis]


class TestBinning:
    def test_get_bins_basic(self, array_stats, rng):
        ary = rng.normal(size=(100,))
        bins = array_stats.get_bins(ary)
        assert bins.ndim == 1
        assert len(bins) == 9
        assert bins[0] == ary.min()
        assert ary.max() <= bins[-1]

    @pytest.mark.parametrize("bins", [10, 20, 50])
    def test_get_bins_integer(self, array_stats, rng, bins):
        ary = rng.normal(size=(100,))
        result = array_stats.get_bins(ary, bins=bins)
        assert result.ndim == 1
        assert len(result) == bins + 1

    @pytest.mark.parametrize("axis", [0, 1, -1])
    def test_get_bins_axis(self, array_stats, rng, axis):
        ary = rng.normal(size=(10, 20, 30))
        result = array_stats.get_bins(ary, axis=axis)
        expected_shape = {
            0: (20, 30),
            1: (10, 30),
            -1: (10, 20),
        }
        assert result.shape[:-1] == expected_shape[axis]
        assert result.shape[-1] > 0

    def test_histogram_basic(self, array_stats, rng):
        ary = rng.normal(size=(1000,))
        counts, edges = array_stats.histogram(ary)
        assert len(counts) == len(edges) - 1
        assert counts.sum() == 1000

    @pytest.mark.parametrize("bins", [10, 20, 50])
    def test_histogram_bins(self, array_stats, rng, bins):
        ary = rng.normal(size=(1000,))
        counts, _ = array_stats.histogram(ary, bins=bins)
        assert len(counts) == bins

    def test_histogram_density(self, array_stats, rng):
        ary = rng.normal(size=(1000,))
        counts, edges = array_stats.histogram(ary, density=True)
        widths = np.diff(edges)
        integral = np.sum(counts * widths)
        assert_allclose(integral, 1.0, rtol=0.01)

    def test_histogram_range(self, array_stats, rng):
        ary = rng.normal(size=(1000,))
        _, edges = array_stats.histogram(ary, bins=20, range=(-2, 2))
        assert_allclose(edges[0], -2)
        assert_allclose(edges[-1], 2)

    @pytest.mark.parametrize("axis", [0, 1, -1])
    def test_histogram_axis(self, array_stats, rng, axis):
        ary = rng.normal(size=(10, 20, 30))
        counts, edges = array_stats.histogram(ary, bins=15, axis=axis)
        assert counts.ndim == ary.ndim
        assert edges.ndim == ary.ndim
        expected_counts_shape = {
            0: (20, 30, 15),
            1: (10, 30, 15),
            -1: (10, 20, 15),
        }
        expected_edges_shape = {
            0: (20, 30, 16),
            1: (10, 30, 16),
            -1: (10, 20, 16),
        }
        assert counts.shape == expected_counts_shape[axis]
        assert edges.shape == expected_edges_shape[axis]

    def test_histogram_axis_list(self, array_stats, rng):
        ary = rng.normal(size=(10, 20, 30, 40))
        counts, edges = array_stats.histogram(ary, bins=12, axis=[1, 3])
        assert counts.ndim == 3
        assert edges.ndim == 3
        assert counts.shape == (10, 30, 12)
        assert edges.shape == (10, 30, 13)


class TestKDE:
    def test_kde_basic(self, array_stats):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(100,))
        grid, pdf, bw = array_stats.kde(ary)
        assert len(grid) == 512
        assert len(pdf) == 512
        assert bw > 0

    @pytest.mark.parametrize("grid_len", [128, 256])
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

    @pytest.mark.parametrize("axis", [0, 1, -1])
    def test_kde_axis(self, array_stats, axis):
        rng = np.random.default_rng(42)
        ary = rng.normal(size=(10, 20, 30))
        grid, pdf, bw = array_stats.kde(ary, axis=axis)
        expected_shape = tuple(s for i, s in enumerate(ary.shape) if i != axis % ary.ndim) + (512,)
        assert grid.shape == expected_shape
        assert pdf.shape == expected_shape
        assert bw.shape == tuple(s for i, s in enumerate(ary.shape) if i != axis % ary.ndim)


class TestQuantileDots:
    def test_qds_basic(self, array_stats, rng):
        ary = rng.normal(size=(100,))
        x, y, radius = array_stats.qds(ary, nquantiles=20)
        assert len(x) == 20
        assert len(y) == 20
        assert radius > 0

    @pytest.mark.parametrize("nquantiles", [10, 20, 50, 100])
    def test_qds_nquantiles(self, array_stats, rng, nquantiles):
        ary = rng.normal(size=(200,))
        x, y, _ = array_stats.qds(ary, nquantiles=nquantiles)
        assert len(x) == nquantiles
        assert len(y) == nquantiles

    @pytest.mark.parametrize("axis", [0, 1, -1])
    def test_qds_axis(self, array_stats, rng, axis):
        ary = rng.normal(size=(15, 25, 35))
        x, y, radius = array_stats.qds(ary, nquantiles=10, axis=axis)
        expected_shape = {
            0: (25, 35, 10),
            1: (15, 35, 10),
            -1: (15, 25, 10),
        }
        assert x.shape == expected_shape[axis]
        assert y.shape == expected_shape[axis]
        expected_radius_shape = {
            0: (25, 35),
            1: (15, 35),
            -1: (15, 25),
        }
        assert radius.shape == expected_radius_shape[axis]


class TestECDF:
    def test_ecdf_basic(self, array_stats, rng):
        ary = rng.normal(size=(100,))
        x, y = array_stats.ecdf(ary, npoints=50)
        assert len(x) == 50
        assert len(y) == 50
        assert np.all(y >= 0)
        assert np.all(y <= 1)

    def test_ecdf_monotonic(self, array_stats, rng):
        ary = rng.normal(size=(100,))
        _, y = array_stats.ecdf(ary, npoints=50)
        assert np.all(np.diff(y) >= -1e-10)

    def test_ecdf_pit(self, array_stats, rng):
        ary = rng.normal(size=(100,))
        x, y = array_stats.ecdf(ary, npoints=50, pit=True)
        assert len(x) == 50
        assert len(y) == 50

    @pytest.mark.parametrize("axis", [0, 1, -1, -2])
    def test_ecdf_axis(self, array_stats, rng, axis):
        ary = rng.normal(size=(10, 20, 30))
        x, y = array_stats.ecdf(ary, npoints=40, axis=axis)
        expected_shape = {
            0: (20, 30, 40),
            1: (10, 30, 40),
            -1: (10, 20, 40),
            -2: (10, 30, 40),
        }
        assert x.shape == expected_shape[axis]
        assert y.shape == expected_shape[axis]


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
        ary = np.empty((4, 100))
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
        ary = np.empty((4, 100))
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


class TestLOO:
    def test_loo_basic(self, array_stats, rng):
        ary = rng.normal(-2, 1, size=(4, 100))
        elpd_i, pareto_k, p_loo_i = array_stats.loo(ary)

        assert elpd_i.shape == ()
        assert pareto_k.shape == ()
        assert p_loo_i.shape == ()
        assert np.isfinite(elpd_i)
        assert np.isfinite(pareto_k)
        assert np.isfinite(p_loo_i)

    def test_loo_multiple_obs(self, array_stats, rng):
        ary = rng.normal(-2, 1, size=(4, 100, 8))
        elpd_i, pareto_k, p_loo_i = array_stats.loo(ary, chain_axis=0, draw_axis=1)

        assert elpd_i.shape == (8,)
        assert pareto_k.shape == (8,)
        assert p_loo_i.shape == (8,)
        assert np.all(np.isfinite(elpd_i))
        assert np.all(np.isfinite(pareto_k))
        assert np.all(np.isfinite(p_loo_i))

    def test_loo_chain_axis_none(self, array_stats, rng):
        ary = rng.normal(-2, 1, size=(100,))
        elpd_i, pareto_k, p_loo_i = array_stats.loo(ary, chain_axis=None, draw_axis=-1)

        assert elpd_i.shape == ()
        assert pareto_k.shape == ()
        assert p_loo_i.shape == ()

    def test_loo_diff_chain_draw_axes(self, array_stats, rng):
        ary = rng.normal(-2, 1, size=(5, 4, 100))
        elpd_i, pareto_k, p_loo_i = array_stats.loo(ary, chain_axis=1, draw_axis=2)

        assert elpd_i.shape == (5,)
        assert pareto_k.shape == (5,)
        assert p_loo_i.shape == (5,)
        assert np.all(np.isfinite(elpd_i))
        assert np.all(np.isfinite(pareto_k))
        assert np.all(np.isfinite(p_loo_i))

    def test_loo_with_reff(self, array_stats, rng):
        ary = rng.normal(-2, 1, size=(4, 100))

        elpd_i_1, pareto_k_1, _ = array_stats.loo(ary, reff=0.5)
        elpd_i_2, pareto_k_2, _ = array_stats.loo(ary, reff=1.0)

        assert not np.isclose(pareto_k_1, pareto_k_2) or not np.isclose(elpd_i_1, elpd_i_2)

    def test_loo_with_log_weights_error(self, array_stats, rng):
        ary = rng.normal(-2, 1, size=(4, 100))
        log_weights = rng.normal(0, 1, size=(4, 100))

        with pytest.raises(ValueError, match="pareto_k must also be provided"):
            array_stats.loo(ary, log_weights=log_weights)

    def test_loo_with_log_jacobian(self, array_stats, rng):
        ary = rng.normal(-2, 1, size=(4, 100, 3))
        log_jacobian = rng.normal(0, 0.1, size=(3,))

        elpd_i_with, pareto_k_with, _ = array_stats.loo(
            ary, chain_axis=0, draw_axis=1, log_jacobian=log_jacobian
        )
        elpd_i_without, pareto_k_without, _ = array_stats.loo(ary, chain_axis=0, draw_axis=1)

        expected_diff = elpd_i_with - elpd_i_without
        assert_allclose(expected_diff, log_jacobian, rtol=1e-10)
        assert_allclose(pareto_k_with, pareto_k_without)

    def test_loo_matches_xarray(self, array_stats, centered_eight):
        log_lik = get_log_likelihood_dataset(centered_eight, var_names="obs")["obs"]
        loo_inputs = _prepare_loo_inputs(centered_eight, var_name="obs")
        reff = _get_r_eff(centered_eight, loo_inputs.n_samples)

        loo_xr = loo(centered_eight, pointwise=True, var_name="obs")
        elpd_i_array, pareto_k_array, p_loo_i_array = array_stats.loo(
            log_lik.values, chain_axis=0, draw_axis=1, reff=reff
        )

        lppd_xr = einstats.logsumexp(log_lik, b=1 / loo_inputs.n_samples, dims=["chain", "draw"])
        p_loo_i_xr = lppd_xr - loo_xr.elpd_i

        assert_allclose(elpd_i_array, loo_xr.elpd_i.values, rtol=1e-10)
        assert_allclose(pareto_k_array, loo_xr.pareto_k.values, rtol=1e-10)
        assert_allclose(p_loo_i_array, p_loo_i_xr.values, rtol=1e-10)
