# pylint: disable=redefined-outer-name, no-self-use

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from arviz_stats.base.array import BaseArray

from ..helpers import importorskip


@pytest.fixture
def array_stats():
    return BaseArray()


@pytest.fixture(scope="module")
def _xr():
    return importorskip("xarray")


class TestHistogram2D:
    def test_positional_bins(self, array_stats):
        histogram, x_edges, y_edges = array_stats.histogram2d(
            np.arange(6), np.arange(6), 3, density=False
        )

        assert histogram.shape == (3, 3)
        assert x_edges.shape == (4,)
        assert y_edges.shape == (4,)

    @pytest.mark.parametrize("density", [False, True])
    def test_matches_numpy(self, array_stats, density):
        rng = np.random.default_rng(42)
        x = rng.normal(size=500)
        y = 0.5 * x + rng.normal(size=500)
        weights = rng.uniform(0.5, 1.5, size=500)
        kwargs = {
            "bins": (np.linspace(-2, 2, 9), np.linspace(-3, 3, 7)),
            "range": ((-2, 2), (-3, 3)),
            "weights": weights,
            "density": density,
        }

        actual = array_stats.histogram2d(x, y, **kwargs)
        expected = np.histogram2d(x, y, **kwargs)

        for actual_item, expected_item in zip(actual, expected):
            assert_allclose(actual_item, expected_item)

    def test_batched_axes(self, array_stats):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(3, 4, 20))
        y = rng.normal(size=(3, 4, 20))

        histogram, x_edges, y_edges = array_stats.histogram2d(
            x,
            y,
            bins=(5, 7),
            range=((-3, 3), (-3, 3)),
            axis=(1, 2),
            density=False,
        )

        assert histogram.shape == (3, 5, 7)
        assert x_edges.shape == (3, 6)
        assert y_edges.shape == (3, 8)
        for index in range(3):
            expected = np.histogram2d(
                x[index].ravel(),
                y[index].ravel(),
                bins=(5, 7),
                range=((-3, 3), (-3, 3)),
                density=False,
            )
            assert_allclose(histogram[index], expected[0])
            assert_allclose(x_edges[index], expected[1])
            assert_allclose(y_edges[index], expected[2])

    def test_removes_nonfinite_pairs_and_weights(self, array_stats):
        x = np.array([0.1, 0.2, np.nan, 0.8, 0.9])
        y = np.array([0.1, np.inf, 0.3, 0.8, 0.9])
        weights = np.array([1.0, 2.0, 3.0, np.nan, 5.0])

        actual = array_stats.histogram2d(
            x,
            y,
            bins=2,
            range=((0, 1), (0, 1)),
            weights=weights,
            density=False,
        )
        expected = np.histogram2d(
            x[[0, 4]],
            y[[0, 4]],
            bins=2,
            range=((0, 1), (0, 1)),
            weights=weights[[0, 4]],
            density=False,
        )

        for actual_item, expected_item in zip(actual, expected):
            assert_allclose(actual_item, expected_item)

    @pytest.mark.parametrize(
        ("x", "y", "weights", "message"),
        [
            (np.ones(3), np.ones(4), None, "same shape"),
            (np.ones(3), np.ones(3), np.ones(4), "same shape"),
            (np.array([np.nan]), np.array([1.0]), None, "No finite paired samples"),
        ],
    )
    def test_invalid_inputs(self, array_stats, x, y, weights, message):
        with pytest.raises(ValueError, match=message):
            array_stats.histogram2d(x, y, weights=weights)


class TestHexbin:
    @pytest.mark.parametrize("density", [False, True])
    def test_weights(self, array_stats, density):
        x = np.array([0.25, 0.75, 0.75])
        y = np.array([0.25, 0.75, 0.75])
        weights = np.array([1.0, 2.0, 3.0])

        values, _ = array_stats.hexbin(
            x,
            y,
            gridsize=(2, 1),
            extent=(0, 1, 0, 1),
            weights=weights,
            density=density,
        )

        expected = np.array([0, 0, 0, 0, 0, 0, 1, 5], dtype=float)
        if density:
            expected /= weights.sum() / 4
        assert_allclose(values, expected)

    def test_matches_fixed_matplotlib_reference(self, array_stats):
        x = np.array([0.0, 0.2, 0.8, 1.0])
        y = np.array([0.0, 0.8, 0.2, 1.0])
        expected_values = np.array([1, 0, 0, 0, 0, 1, 1, 1], dtype=float)
        expected_offsets = np.array(
            [
                [-1e-9, 0.0],
                [-1e-9, 1.0],
                [0.5000000000000001, 0.0],
                [0.5000000000000001, 1.0],
                [1.000000001, 0.0],
                [1.000000001, 1.0],
                [0.24999999950000004, 0.5],
                [0.7500000005000002, 0.5],
            ]
        )

        values, offsets = array_stats.hexbin(
            x,
            y,
            gridsize=(2, 1),
            extent=(0, 1, 0, 1),
            density=False,
        )

        assert_array_equal(values, expected_values)
        assert_allclose(offsets, expected_offsets, rtol=0, atol=1e-15)

    @pytest.mark.parametrize(("gridsize", "expected_cells"), [(4, 23), ((4, 3), 32)])
    def test_grid_shapes_and_zero_cells(self, array_stats, gridsize, expected_cells):
        values, offsets = array_stats.hexbin(
            np.array([0.25, 0.75]),
            np.array([0.25, 0.75]),
            gridsize=gridsize,
            extent=(0, 1, 0, 1),
            density=False,
        )

        assert values.shape == (expected_cells,)
        assert offsets.shape == (expected_cells, 2)
        assert values.sum() == 2
        assert np.count_nonzero(values == 0) == expected_cells - 2

    def test_extent_clips_samples(self, array_stats):
        x = np.array([0.25, 0.75, 2.0])
        y = np.array([0.25, 0.75, 2.0])
        values, _ = array_stats.hexbin(
            x,
            y,
            gridsize=(2, 1),
            extent=(0, 1, 0, 1),
            density=False,
        )

        assert_array_equal(values, [0, 0, 0, 0, 0, 0, 1, 1])

    def test_density_normalizes_samples_inside_extent(self, array_stats):
        x = np.array([0.25, 0.75, 2.0])
        y = np.array([0.25, 0.75, 2.0])
        values, _ = array_stats.hexbin(
            x,
            y,
            gridsize=(2, 1),
            extent=(0, 1, 0, 1),
            density=True,
        )
        assert_allclose(values, [0, 0, 0, 0, 0, 0, 2, 2])

    def test_empty_extent(self, array_stats):
        x = np.array([2.0, 3.0])
        y = np.array([2.0, 3.0])

        with pytest.raises(ValueError, match="No samples fall inside"):
            array_stats.hexbin(
                x,
                y,
                gridsize=(2, 1),
                extent=(0, 1, 0, 1),
                density=True,
            )

        values, _ = array_stats.hexbin(
            x,
            y,
            gridsize=(2, 1),
            extent=(0, 1, 0, 1),
            density=False,
        )

        assert_array_equal(values, np.zeros_like(values))

    def test_batched_axis(self, array_stats):
        x = np.array([[0.1, 0.9], [0.2, 0.8]])
        y = np.array([[0.1, 0.9], [0.8, 0.2]])
        values, offsets = array_stats.hexbin(
            x,
            y,
            gridsize=(2, 1),
            extent=(0, 1, 0, 1),
            axis=1,
            density=False,
        )

        for index in range(2):
            expected_values, expected_offsets = array_stats.hexbin(
                x[index],
                y[index],
                gridsize=(2, 1),
                extent=(0, 1, 0, 1),
                density=False,
            )
            assert_array_equal(values[index], expected_values)
            assert_allclose(offsets[index], expected_offsets)

    def test_singular_and_nonfinite_data(self, array_stats):
        values, offsets = array_stats.hexbin(
            np.array([1.0, 1.0, np.nan]),
            np.array([0.0, 0.0, 1.0]),
            gridsize=(2, 1),
            density=False,
        )

        assert values.sum() == 2
        assert np.all(np.isfinite(offsets))

    @pytest.mark.parametrize("gridsize", [0, -1, (2, 0), (2, 1.5), (1, 2, 3)])
    def test_invalid_gridsize(self, array_stats, gridsize):
        with pytest.raises(ValueError, match="gridsize"):
            array_stats.hexbin([0], [0], gridsize=gridsize)

    @pytest.mark.parametrize("extent", [(0, 1, 0), (1, 0, 0, 1), (0, np.inf, 0, 1)])
    def test_invalid_extent(self, array_stats, extent):
        with pytest.raises(ValueError, match="Extent|extent"):
            array_stats.hexbin([0], [0], extent=extent)


class TestLabeledBivariateHistograms:
    @pytest.mark.parametrize("method", ["histogram2d", "hexbin"])
    def test_aligns_transposed_dimensions(self, method, _xr):
        import arviz_stats as azs

        x = _xr.DataArray(
            [[0, 1, 2], [10, 11, 12]],
            dims=("batch", "sample"),
            coords={"batch": ["a", "b"], "sample": [0, 1, 2]},
        )
        y = _xr.DataArray(
            [[3, 1, 2], [13, 10, 12]],
            dims=("batch", "sample"),
            coords={"batch": ["a", "b"], "sample": [0, 1, 2]},
        ).transpose()

        kwargs = {"bins": 2} if method == "histogram2d" else {"gridsize": (2, 1)}
        result = getattr(azs, method)(x, y, dim=("batch", "sample"), density=False, **kwargs)
        aligned_x, aligned_y = _xr.align(x, y)
        expected = getattr(BaseArray(), method)(
            aligned_x.values,
            aligned_y.transpose(*aligned_x.dims).values,
            axis=(0, 1),
            density=False,
            **kwargs,
        )

        if method == "histogram2d":
            assert_allclose(result.histogram.values, expected[0])
            assert_allclose(result.x_edges.values, expected[1])
            assert_allclose(result.y_edges.values, expected[2])
        else:
            assert_allclose(result["values"].values, expected[0])
            assert_allclose(result.x_centers.values, expected[1][:, 0])
            assert_allclose(result.y_centers.values, expected[1][:, 1])

    def test_histogram2d_dataarray_and_top_level(self, _xr):
        import arviz_stats as azs

        x = _xr.DataArray(np.arange(12).reshape(2, 6), dims=("batch", "sample"))
        y = _xr.DataArray(np.arange(12).reshape(2, 6), dims=("batch", "sample"))

        result = azs.histogram2d(x, y, bins=(3, 2), dim="sample", density=False)
        expected = BaseArray().histogram2d(
            x.values,
            y.values,
            bins=(3, 2),
            axis=1,
            density=False,
        )

        assert set(result.data_vars) == {"histogram", "x_edges", "y_edges"}
        assert result.histogram.dims == ("batch", "histogram2d_x", "histogram2d_y")
        assert_allclose(result.histogram.values, expected[0])
        assert_allclose(result.x_edges.values, expected[1])
        assert_allclose(result.y_edges.values, expected[2])

    def test_hexbin_dataarray_and_top_level(self, _xr):
        import arviz_stats as azs

        x = _xr.DataArray(np.arange(12).reshape(2, 6), dims=("batch", "sample"))
        y = _xr.DataArray(np.arange(12).reshape(2, 6), dims=("batch", "sample"))

        result = azs.hexbin(
            x,
            y,
            gridsize=(2, 1),
            extent=(0, 12, 0, 12),
            dim="sample",
            density=False,
        )
        expected_values, expected_offsets = BaseArray().hexbin(
            x.values,
            y.values,
            gridsize=(2, 1),
            extent=(0, 12, 0, 12),
            axis=1,
            density=False,
        )

        assert set(result.data_vars) == {"values", "x_centers", "y_centers"}
        assert result["values"].dims == ("batch", "hexbin")
        assert_allclose(result["values"].values, expected_values)
        assert_allclose(result.x_centers.values, expected_offsets[..., 0])
        assert_allclose(result.y_centers.values, expected_offsets[..., 1])

    def test_hexbin_dataarray_weights(self, _xr):
        import arviz_stats as azs

        x = _xr.DataArray([0.25, 0.75, 0.75], dims="sample")
        y = _xr.DataArray([0.25, 0.75, 0.75], dims="sample")
        weights = _xr.DataArray([1.0, 2.0, 3.0], dims="sample")

        result = azs.hexbin(
            x,
            y,
            gridsize=(2, 1),
            extent=(0, 1, 0, 1),
            weights=weights,
            dim="sample",
            density=False,
        )

        assert_allclose(result["values"], [0, 0, 0, 0, 0, 0, 1, 5])

    def test_array_top_level_dispatch(self, _xr):
        import arviz_stats as azs

        x = np.arange(12).reshape(2, 6)
        y = np.arange(12).reshape(2, 6)

        actual_histogram = azs.histogram2d(x, y, bins=3, axis=1, density=False)
        expected_histogram = BaseArray().histogram2d(x, y, bins=3, axis=1, density=False)
        values, offsets = azs.hexbin(
            x,
            y,
            gridsize=(2, 1),
            extent=(0, 12, 0, 12),
            axis=1,
            density=False,
        )
        expected_values, expected_offsets = BaseArray().hexbin(
            x,
            y,
            gridsize=(2, 1),
            extent=(0, 12, 0, 12),
            axis=1,
            density=False,
        )

        for actual, expected in zip(actual_histogram, expected_histogram):
            assert_allclose(actual, expected)
        assert_allclose(values, expected_values)
        assert_allclose(offsets, expected_offsets)
