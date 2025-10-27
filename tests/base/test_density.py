"""Tests for density estimation base functions."""

# pylint: disable=redefined-outer-name, no-self-use, protected-access
import numpy as np
import pytest
from numpy.testing import assert_allclose

from arviz_stats.base.density import _DensityBase


@pytest.fixture
def density():
    return _DensityBase()


class TestDCT:
    def test_dct1d_shape(self, density, rng):
        x = rng.normal(size=100)
        result = density.dct1d(x)
        assert result.shape == x.shape

    def test_dct1d_values(self, density):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        result = density.dct1d(x)
        assert result.dtype == np.float64

    @pytest.mark.parametrize("size", [10, 50, 100, 256])
    def test_dct1d_different_sizes(self, density, rng, size):
        x = rng.normal(size=size)
        result = density.dct1d(x)
        assert len(result) == size


class TestBandwidthEstimators:
    def test_bw_scott(self, density, rng):
        x = rng.normal(size=100)
        bw = density.bw_scott(x)
        assert bw > 0
        assert isinstance(bw, float)

    def test_bw_scott_with_std(self, density, rng):
        x = rng.normal(size=100)
        x_std = np.std(x)
        bw1 = density.bw_scott(x)
        bw2 = density.bw_scott(x, x_std=x_std)
        assert_allclose(bw1, bw2)

    def test_bw_silverman(self, density, rng):
        x = rng.normal(size=100)
        bw = density.bw_silverman(x)
        assert bw > 0
        assert isinstance(bw, float)

    def test_bw_silverman_with_std(self, density, rng):
        x = rng.normal(size=100)
        x_std = np.std(x)
        bw1 = density.bw_silverman(x)
        bw2 = density.bw_silverman(x, x_std=x_std)
        assert_allclose(bw1, bw2)

    def test_bw_isj(self, density, rng):
        x = rng.normal(size=100)
        bw = density.bw_isj(x)
        assert bw > 0

    def test_bw_isj_with_precomputed(self, density, rng):
        x = rng.normal(size=100)
        x_std = np.std(x)
        x_range = x.max() - x.min()
        grid_counts, _ = density._histogram(x, bins=256)
        bw = density.bw_isj(x, grid_counts=grid_counts, x_std=x_std, x_range=x_range)
        assert bw > 0

    def test_bw_experimental(self, density, rng):
        x = rng.normal(size=100)
        bw = density.bw_experimental(x)
        assert bw > 0

    def test_bw_experimental_average(self, density, rng):
        x = rng.normal(size=100)
        bw_silverman = density.bw_silverman(x)
        bw_isj = density.bw_isj(x)
        bw_experimental = density.bw_experimental(x)
        assert_allclose(bw_experimental, 0.5 * (bw_silverman + bw_isj))

    @pytest.mark.parametrize("method", ["scott", "silverman", "isj", "experimental"])
    def test_get_bw_string(self, density, rng, method):
        x = rng.normal(size=100)
        bw = density.get_bw(x, method)
        assert bw > 0

    @pytest.mark.parametrize("bw_value", [0.1, 0.5, 1.0, 2.0])
    def test_get_bw_numeric(self, density, rng, bw_value):
        x = rng.normal(size=100)
        bw = density.get_bw(x, bw_value)
        assert bw == bw_value

    def test_get_bw_invalid_bool(self, density, rng):
        x = rng.normal(size=100)
        with pytest.raises(ValueError, match="must not be of type `bool`"):
            density.get_bw(x, True)

    def test_get_bw_negative(self, density, rng):
        x = rng.normal(size=100)
        with pytest.raises(ValueError, match="must be positive"):
            density.get_bw(x, -0.5)

    def test_get_bw_invalid_method(self, density, rng):
        x = rng.normal(size=100)
        with pytest.raises(ValueError, match="Unrecognized bandwidth method"):
            density.get_bw(x, "invalid_method")

    def test_bw_taylor(self, density, rng):
        x = rng.vonmises(0, 2, 100)
        bw = density.bw_taylor(x)
        assert bw > 0


class TestCircularHelpers:
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
    def test_normalize_angle_zero_centered(self, density, input_val, expected):
        result = density._normalize_angle(input_val, zero_centered=True)
        assert_allclose(result, expected, atol=1e-10)

    @pytest.mark.parametrize(
        "input_val,expected",
        [
            (0.0, 0.0),
            (np.pi / 2, np.pi / 2),
            (2 * np.pi, 0.0),
            (3 * np.pi, np.pi),
            (-np.pi / 2, 3 * np.pi / 2),
        ],
    )
    def test_normalize_angle_not_zero_centered(self, density, input_val, expected):
        result = density._normalize_angle(input_val, zero_centered=False)
        assert_allclose(result, expected, atol=1e-10)

    def test_normalize_angle_array(self, density):
        angles = np.linspace(-4 * np.pi, 4 * np.pi, 100)
        result = density._normalize_angle(angles, zero_centered=True)
        assert np.all(result >= -np.pi)
        assert np.all(result < np.pi)

    @pytest.mark.parametrize("kappa", [0.5, 1.0, 2.0, 5.0])
    def test_vonmises_pdf(self, density, kappa):
        x = np.linspace(-np.pi, np.pi, 100)
        mu = 0.0
        pdf = density._vonmises_pdf(x, mu, kappa)
        assert np.all(pdf >= 0)

    def test_vonmises_pdf_invalid_kappa(self, density):
        x = np.array([0.0])
        with pytest.raises(ValueError, match="must be positive"):
            density._vonmises_pdf(x, 0, -1)

    def test_vonmises_pdf_normalization(self, density):
        x = np.linspace(-np.pi, np.pi, 1000)
        mu = 0.0
        kappa = 2.0
        pdf = density._vonmises_pdf(x, mu, kappa)
        dx = x[1] - x[0]
        integral = np.sum(pdf * dx)
        assert_allclose(integral, 1.0, rtol=0.01)

    @pytest.mark.parametrize("x", [0.0, 0.2, 0.5, 0.7, 0.84])
    def test_a1inv_ranges(self, density, x):
        result = density._a1inv(x)
        assert result >= 0

    def test_kappa_mle(self, density, rng):
        x = rng.vonmises(0, 2, 100)
        kappa = density._kappa_mle(x)
        assert kappa > 0


class TestGridUtilities:
    def test_check_custom_lims_valid(self, density):
        result = density.check_custom_lims([-2, 12], -1, 11)
        assert result == [-2, 12]

    def test_check_custom_lims_none_lower(self, density):
        result = density.check_custom_lims([None, 10], -5, 8)
        assert result == [-5, 10]

    def test_check_custom_lims_none_upper(self, density):
        result = density.check_custom_lims([-6, None], -5, 8)
        assert result == [-6, 8]

    def test_check_custom_lims_invalid_type(self, density):
        with pytest.raises(TypeError, match="must be a numeric list or tuple"):
            density.check_custom_lims("invalid", 0, 10)

    def test_check_custom_lims_wrong_length(self, density):
        with pytest.raises(AttributeError, match="must be 2"):
            density.check_custom_lims([1, 2, 3], 0, 10)

    def test_check_custom_lims_bool(self, density):
        with pytest.raises(TypeError, match="must be numeric or None, not bool"):
            density.check_custom_lims([True, 10], 0, 10)

    def test_check_custom_lims_reversed(self, density):
        with pytest.raises(ValueError, match="must be smaller"):
            density.check_custom_lims([10, 0], 0, 10)

    def test_check_custom_lims_outside_data(self, density):
        with pytest.raises(ValueError, match="outside `custom_lims` boundaries"):
            density.check_custom_lims([2, 8], 0, 10)

    @pytest.mark.parametrize("grid_len", [100, 256, 512, 1024])
    def test_get_grid_length(self, density, grid_len):
        x_min, x_max = 0.0, 10.0
        x_std = 2.0
        _, _, result_len = density.get_grid(
            x_min, x_max, x_std, extend_fct=0.5, grid_len=grid_len, custom_lims=None
        )
        assert result_len == grid_len

    def test_get_grid_extend(self, density):
        x_min, x_max = 0.0, 10.0
        x_std = 2.0
        grid_min, grid_max, _ = density.get_grid(
            x_min, x_max, x_std, extend_fct=0.5, grid_len=512, custom_lims=None, extend=True
        )
        assert grid_min < x_min
        assert grid_max > x_max

    def test_get_grid_no_extend(self, density):
        x_min, x_max = 0.0, 10.0
        x_std = 2.0
        grid_min, grid_max, _ = density.get_grid(
            x_min, x_max, x_std, extend_fct=0.5, grid_len=512, custom_lims=None, extend=False
        )
        assert grid_min == x_min
        assert grid_max == x_max

    def test_get_grid_custom_lims(self, density):
        x_min, x_max = 0.0, 10.0
        x_std = 2.0
        custom_lims = [-5, 15]
        grid_min, grid_max, _ = density.get_grid(
            x_min, x_max, x_std, extend_fct=0.5, grid_len=512, custom_lims=custom_lims
        )
        assert grid_min == -5
        assert grid_max == 15

    def test_get_grid_bound_correction(self, density):
        x_min, x_max = 0.0, 10.0
        x_std = 2.0
        grid_min, grid_max, _ = density.get_grid(
            x_min,
            x_max,
            x_std,
            extend_fct=0.5,
            grid_len=512,
            custom_lims=None,
            extend=True,
            bound_correction=True,
        )
        assert grid_min == x_min
        assert grid_max == x_max


class TestKDELinear:
    def test_kde_linear_basic(self, density, rng):
        x = rng.normal(size=100)
        grid, pdf, bw = density.kde_linear(x)
        assert len(grid) == 512
        assert len(pdf) == 512
        assert bw > 0

    @pytest.mark.parametrize("bw_method", ["scott", "silverman", "isj", "experimental"])
    def test_kde_linear_bw_methods(self, density, rng, bw_method):
        x = rng.normal(size=100)
        grid, pdf, bw = density.kde_linear(x, bw=bw_method)
        assert len(grid) == len(pdf)
        assert bw > 0

    def test_kde_linear_numeric_bw(self, density, rng):
        x = rng.normal(size=100)
        bw_value = 0.5
        grid, pdf, bw = density.kde_linear(x, bw=bw_value)
        assert len(grid) == 512
        assert len(pdf) == 512
        assert_allclose(bw, bw_value)

    def test_kde_linear_normalization(self, density, rng):
        x = rng.normal(size=1000)
        grid, pdf, _ = density.kde_linear(x, cumulative=False)
        dx = grid[1] - grid[0]
        integral = np.sum(pdf * dx)
        assert_allclose(integral, 1.0, rtol=0.01)

    def test_kde_linear_cumulative(self, density, rng):
        x = rng.normal(size=100)
        _, pdf, _ = density.kde_linear(x, cumulative=True)
        assert np.all(np.diff(pdf) >= -1e-10)
        assert_allclose(pdf[-1], 1.0, rtol=0.01)

    @pytest.mark.parametrize("grid_len", [128, 256, 512, 1024])
    def test_kde_linear_grid_len(self, density, rng, grid_len):
        x = rng.normal(size=100)
        grid, pdf, _ = density.kde_linear(x, grid_len=grid_len)
        assert len(grid) == grid_len
        assert len(pdf) == grid_len

    def test_kde_linear_adaptive(self, density, rng):
        x = rng.normal(size=100)
        grid, pdf, bw = density.kde_linear(x, adaptive=True)
        assert len(grid) == len(pdf)
        assert bw > 0

    @pytest.mark.parametrize("bound_correction", [True, False])
    def test_kde_linear_bound_correction(self, density, rng, bound_correction):
        x = rng.normal(size=100)
        grid, pdf, _ = density.kde_linear(x, bound_correction=bound_correction)
        assert len(grid) == len(pdf)

    def test_kde_linear_custom_lims(self, density, rng):
        x = rng.normal(size=100)
        custom_lims = [x.min() - 1, x.max() + 1]
        grid, _, _ = density.kde_linear(x, custom_lims=custom_lims)
        assert grid.min() >= custom_lims[0]
        assert grid.max() <= custom_lims[1]

    @pytest.mark.parametrize("bw_fct", [0.5, 1.0, 2.0])
    def test_kde_linear_bw_fct(self, density, rng, bw_fct):
        x = rng.normal(size=100)
        grid, pdf, _ = density.kde_linear(x, bw_fct=bw_fct)
        assert len(grid) == len(pdf)

    def test_kde_linear_invalid_bw_fct_type(self, density, rng):
        x = rng.normal(size=100)
        with pytest.raises(TypeError, match="must be a positive number"):
            density.kde_linear(x, bw_fct="invalid")

    def test_kde_linear_negative_bw_fct(self, density, rng):
        x = rng.normal(size=100)
        with pytest.raises(ValueError, match="must be a positive number"):
            density.kde_linear(x, bw_fct=-0.5)


class TestKDEConvolution:
    def test_kde_convolution_basic(self, density, rng):
        x = rng.normal(size=100)
        grid_len = 256
        grid_counts, grid_edges = density._histogram(x, bins=grid_len, density=False)
        bw = density.bw_silverman(x)
        grid, pdf = density.kde_convolution(x, bw, grid_edges, grid_counts, grid_len, False)
        assert len(grid) == grid_len
        assert len(pdf) == grid_len

    def test_kde_convolution_bound_correction(self, density, rng):
        x = rng.normal(size=100)
        grid_len = 256
        grid_counts, grid_edges = density._histogram(x, bins=grid_len, density=False)
        bw = density.bw_silverman(x)
        grid, pdf = density.kde_convolution(x, bw, grid_edges, grid_counts, grid_len, True)
        assert len(grid) == grid_len
        assert len(pdf) == grid_len


class TestKDEAdaptive:
    def test_kde_adaptive_basic(self, density, rng):
        x = rng.normal(size=100)
        grid_len = 256
        grid_counts, grid_edges = density._histogram(x, bins=grid_len, density=False)
        bw = density.bw_silverman(x)
        grid, pdf = density.kde_adaptive(x, bw, grid_edges, grid_counts, grid_len, False)
        assert len(grid) == grid_len
        assert len(pdf) == grid_len

    def test_kde_adaptive_bound_correction(self, density, rng):
        x = rng.normal(size=100)
        grid_len = 256
        grid_counts, grid_edges = density._histogram(x, bins=grid_len, density=False)
        bw = density.bw_silverman(x)
        grid, pdf = density.kde_adaptive(x, bw, grid_edges, grid_counts, grid_len, True)
        assert len(grid) == grid_len
        assert len(pdf) == grid_len


class TestKDECircular:
    def test_kde_circular_basic(self, density, rng):
        x = rng.vonmises(0, 2, 100)
        grid, pdf, bw = density.kde_circular(x)
        assert len(grid) == 512
        assert len(pdf) == 512
        assert bw > 0

    def test_kde_circular_normalization(self, density, rng):
        x = rng.vonmises(0, 2, 1000)
        grid, pdf, _ = density.kde_circular(x, cumulative=False)
        dx = grid[1] - grid[0]
        integral = np.sum(pdf * dx)
        assert_allclose(integral, 1.0, rtol=0.01)

    def test_kde_circular_cumulative(self, density, rng):
        x = rng.vonmises(0, 2, 100)
        _, pdf, _ = density.kde_circular(x, cumulative=True)
        assert_allclose(pdf[-1], 1.0, rtol=0.01)

    def test_kde_circular_numeric_bw(self, density, rng):
        x = rng.vonmises(0, 2, 100)
        bw_value = 5.0
        grid, pdf, bw = density.kde_circular(x, bw=bw_value)
        assert len(grid) == 512
        assert len(pdf) == 512
        assert_allclose(bw, bw_value)

    @pytest.mark.parametrize("grid_len", [128, 256, 512, 1024])
    def test_kde_circular_grid_len(self, density, rng, grid_len):
        x = rng.vonmises(0, 2, 100)
        grid, pdf, _ = density.kde_circular(x, grid_len=grid_len)
        assert len(grid) == grid_len
        assert len(pdf) == grid_len

    @pytest.mark.parametrize("bw_fct", [0.5, 1.0, 2.0])
    def test_kde_circular_bw_fct(self, density, rng, bw_fct):
        x = rng.vonmises(0, 2, 100)
        grid, pdf, _ = density.kde_circular(x, bw_fct=bw_fct)
        assert len(grid) == len(pdf)

    def test_kde_circular_invalid_bw_fct_type(self, density, rng):
        x = rng.vonmises(0, 2, 100)
        with pytest.raises(TypeError, match="must be a positive number"):
            density.kde_circular(x, bw_fct="invalid")

    def test_kde_circular_negative_bw_fct(self, density, rng):
        x = rng.vonmises(0, 2, 100)
        with pytest.raises(ValueError, match="must be a positive number"):
            density.kde_circular(x, bw_fct=-0.5)

    def test_kde_circular_invalid_bw_bool(self, density, rng):
        x = rng.vonmises(0, 2, 100)
        with pytest.raises(ValueError, match="can't be of type `bool`"):
            density.kde_circular(x, bw=True)

    def test_kde_circular_negative_bw(self, density, rng):
        x = rng.vonmises(0, 2, 100)
        with pytest.raises(ValueError, match="must be positive"):
            density.kde_circular(x, bw=-1.0)

    def test_kde_circular_invalid_bw_string(self, density, rng):
        x = rng.vonmises(0, 2, 100)
        with pytest.raises(ValueError, match="must be a positive numeric or `taylor`"):
            density.kde_circular(x, bw="invalid")


class TestKDEWrapper:
    def test_kde_wrapper_linear(self, density, rng):
        x = rng.normal(size=100)
        grid, pdf, bw = density._kde(x, circular=False)
        assert len(grid) == 512
        assert len(pdf) == 512
        assert bw > 0

    def test_kde_wrapper_circular(self, density, rng):
        x = rng.vonmises(0, 2, 100)
        grid, pdf, bw = density._kde(x, circular=True)
        assert len(grid) == 512
        assert len(pdf) == 512
        assert bw > 0

    def test_kde_wrapper_circular_degrees(self, density, rng):
        x = rng.uniform(0, 360, 100)
        grid, pdf, bw = density._kde(x, circular="degrees")
        assert len(grid) == 512
        assert len(pdf) == 512
        assert bw > 0

    def test_kde_wrapper_single_value(self, density):
        x = np.ones(100)
        with pytest.warns(UserWarning, match="single value"):
            _, pdf, _ = density._kde(x)
        assert np.all(np.isnan(pdf))

    def test_kde_wrapper_empty(self, density):
        x = np.array([np.nan, np.nan])
        with pytest.warns(UserWarning, match="single value or no finite values"):
            _, pdf, _ = density._kde(x)
        assert np.all(np.isnan(pdf))


class TestKDE2D:
    def test_fast_kde_2d_basic(self, density, rng):
        x = rng.normal(size=100)
        y = rng.normal(size=100)
        grid, xmin, xmax, ymin, ymax = density._fast_kde_2d(x, y)
        assert grid.shape == (128, 128)
        assert xmin < xmax
        assert ymin < ymax

    @pytest.mark.parametrize("gridsize", [(64, 64), (128, 128), (256, 256)])
    def test_fast_kde_2d_gridsize(self, density, rng, gridsize):
        x = rng.normal(size=100)
        y = rng.normal(size=100)
        grid, _, _, _, _ = density._fast_kde_2d(x, y, gridsize=gridsize)
        assert grid.shape == gridsize

    def test_fast_kde_2d_circular(self, density, rng):
        x = rng.normal(size=100)
        y = rng.normal(size=100)
        grid, _, _, _, _ = density._fast_kde_2d(x, y, circular=True)
        assert grid.shape == (128, 128)

    def test_fast_kde_2d_normalization(self, density, rng):
        x = rng.normal(size=1000)
        y = rng.normal(size=1000)
        grid, xmin, xmax, ymin, ymax = density._fast_kde_2d(x, y)
        dx = (xmax - xmin) / (grid.shape[0] - 1)
        dy = (ymax - ymin) / (grid.shape[1] - 1)
        integral = np.sum(grid * dx * dy)
        assert_allclose(integral, 1.0, rtol=0.1)


class TestHDIContours:
    def test_find_hdi_contours_basic(self, density):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 1000)
        y = rng.normal(0, 1, 1000)
        grid, _, _, _, _ = density._fast_kde_2d(x, y)
        hdi_probs = np.array([0.5, 0.9])
        contours = density._find_hdi_contours(grid, hdi_probs)
        assert len(contours) == 2
        assert contours[0] > contours[1]

    @pytest.mark.parametrize("prob", [0.5, 0.8, 0.9, 0.95])
    def test_find_hdi_contours_single_prob(self, density, prob):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 1000)
        y = rng.normal(0, 1, 1000)
        grid, _, _, _, _ = density._fast_kde_2d(x, y)
        contours = density._find_hdi_contours(grid, np.array([prob]))
        assert len(contours) == 1
        assert contours[0] >= 0


class TestQuantileDots:
    def test_qds_basic(self, density, rng):
        x = rng.normal(size=100)
        x_out, y_out, radius = density._qds(
            x, nquantiles=20, binwidth=None, dotsize=1, stackratio=1
        )
        assert len(x_out) == 20
        assert len(y_out) == 20
        assert radius > 0

    @pytest.mark.parametrize("nquantiles", [10, 20, 50, 100])
    def test_qds_nquantiles(self, density, rng, nquantiles):
        x = rng.normal(size=200)
        x_out, y_out, _ = density._qds(
            x, nquantiles=nquantiles, binwidth=None, dotsize=1, stackratio=1
        )
        assert len(x_out) == nquantiles
        assert len(y_out) == nquantiles

    def test_qds_too_many_quantiles(self, density, rng):
        x = rng.normal(size=50)
        with pytest.warns(UserWarning, match="nquantiles"):
            x_out, _, _ = density._qds(x, nquantiles=100, binwidth=None, dotsize=1, stackratio=1)
        assert len(x_out) == 50

    def test_compute_quantiles_and_binwidth(self, density, rng):
        x = rng.normal(size=100)
        qvalues, binwidth = density._compute_quantiles_and_binwidth(x, nquantiles=10)
        assert len(qvalues) == 10
        assert binwidth > 0
        assert np.all(np.diff(qvalues) >= 0)

    def test_compute_quantiles_custom_binwidth(self, density, rng):
        x = rng.normal(size=100)
        custom_binwidth = 0.5
        _, binwidth = density._compute_quantiles_and_binwidth(
            x, nquantiles=10, binwidth=custom_binwidth
        )
        assert binwidth == custom_binwidth

    def test_wilkinson_algorithm(self, density):
        values = np.array([1, 1.1, 1.2, 5, 5.1, 10])
        stack_locs, stack_counts = density._wilkinson_algorithm(values, binwidth=0.5)
        assert len(stack_locs) == len(stack_counts)
        assert sum(stack_counts) == len(values)

    def test_layout_stacks(self, density):
        stack_locs = [1.0, 5.0, 10.0]
        stack_counts = [3, 2, 1]
        x, y = density._layout_stacks(stack_locs, stack_counts, binwidth=0.5, stackratio=1.0)
        assert len(x) == 6
        assert len(y) == 6
        assert np.all(y >= 0)


class TestECDF:
    def test_ecdf_basic(self, density, rng):
        x = rng.normal(size=100)
        eval_points, ecdf = density._ecdf(x, npoints=50, pit=False)
        assert len(eval_points) == 50
        assert len(ecdf) == 50
        assert np.all(ecdf >= 0)
        assert np.all(ecdf <= 1)

    def test_ecdf_monotonic(self, density, rng):
        x = rng.normal(size=100)
        _, ecdf = density._ecdf(x, npoints=50, pit=False)
        assert np.all(np.diff(ecdf) >= -1e-10)

    def test_ecdf_pit(self, density, rng):
        x = rng.normal(size=100)
        eval_points, ecdf = density._ecdf(x, npoints=50, pit=True)
        assert len(eval_points) == 50
        assert len(ecdf) == 50

    @pytest.mark.parametrize("npoints", [10, 50, 100, 200])
    def test_ecdf_npoints(self, density, rng, npoints):
        x = rng.normal(size=100)
        eval_points, ecdf = density._ecdf(x, npoints=npoints, pit=False)
        assert len(eval_points) == npoints
        assert len(ecdf) == npoints

    def test_ecdf_none_npoints(self, density, rng):
        x = rng.normal(size=50)
        eval_points, ecdf = density._ecdf(x, npoints=None, pit=False)
        assert len(eval_points) == 50
        assert len(ecdf) == 50

    def test_ecdf_large_dataset(self, density, rng):
        x = rng.normal(size=1000)
        eval_points, ecdf = density._ecdf(x, npoints=None, pit=False)
        assert len(eval_points) == 200
        assert len(ecdf) == 200

    def test_ecdf_single_value(self, density):
        x = np.ones(100)
        eval_points, ecdf = density._ecdf(x, npoints=50, pit=False)
        assert len(eval_points) == 50
        assert len(ecdf) == 50

    def test_ecdf_with_nans(self, density, rng):
        x = rng.normal(size=100)
        x[:10] = np.nan
        eval_points, ecdf = density._ecdf(x, npoints=50, pit=False)
        assert len(eval_points) == 50
        assert len(ecdf) == 50


class TestKDEEdgeCases:
    def test_kde_two_values(self, density):
        x = np.array([1.0] * 50 + [2.0] * 50)
        grid, pdf, bw = density.kde_linear(x)
        assert len(grid) == 512
        assert len(pdf) == 512
        assert bw > 0

    def test_kde_with_outliers(self, density, rng):
        x = np.concatenate([rng.normal(size=95), [100, -100, 200, -200, 300]])
        grid, _pdf, bw = density.kde_linear(x)
        assert len(grid) == 512
        assert bw > 0

    def test_kde_circular_boundary_wrapping(self, density, rng):
        x = np.concatenate(
            [rng.uniform(-np.pi, -np.pi + 0.3, 50), rng.uniform(np.pi - 0.3, np.pi, 50)]
        )
        grid, _pdf, bw = density.kde_circular(x)
        assert len(grid) == 512
        assert bw > 0


class TestHistogramEdgeCases:
    def test_histogram_empty_bins(self, density, rng):
        x = rng.normal(size=100)
        counts, _edges = density._histogram(x, bins=100, range=(-10, -5))
        assert len(counts) == 100
        assert np.sum(counts) == 0

    def test_histogram_single_bin(self, density, rng):
        x = rng.normal(size=100)
        counts, edges = density._histogram(x, bins=1)
        assert len(counts) == 1
        assert len(edges) == 2
        assert np.sum(counts) == 100

    def test_histogram_all_same_value(self, density):
        x = np.ones(100) * 5.0
        counts, _edges = density._histogram(x, bins=10)
        assert np.sum(counts > 0) <= 2
