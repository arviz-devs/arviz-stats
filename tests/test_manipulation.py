"""Test for thinning function."""

import numpy as np
import pytest

from .helpers import importorskip

azb = importorskip("arviz_base")
xr = importorskip("xarray")

from arviz_stats.manipulation import thin


def test_thin_numpy_array():
    """Test thinning with numpy arrays."""
    np.random.seed(42)
    data = np.random.normal(size=(4, 100, 3))

    thinned = thin(data, factor=5, chain_axis=0, draw_axis=1)
    assert thinned.shape == (4, 20, 3)

    # Test with factor larger than draws
    thinned = thin(data, factor=200, chain_axis=0, draw_axis=1)
    assert thinned.shape == (4, 1, 3)


def test_thin_dataarray(datatree):
    """Test thinning with xarray DataArray."""
    data = datatree.posterior.theta

    thinned = thin(data, factor=5)
    assert thinned.sizes["chain"] == data.sizes["chain"]
    assert thinned.sizes["draw"] == data.sizes["draw"] // 5


def test_thin_dataset(datatree):
    """Test thinning with xarray Dataset."""
    data = datatree.posterior

    thinned = thin(data, factor=5)
    assert thinned.sizes["chain"] == data.sizes["chain"]
    assert thinned.sizes["draw"] == data.sizes["draw"] // 5
    for var in data.data_vars:
        assert var in thinned


def test_thin_datatree(datatree):
    """Test thinning with DataTree (InferenceData-like object)."""
    original_draws = datatree.posterior.sizes["draw"]

    thinned = thin(datatree, factor=5)
    expected_draws = original_draws // 5
    assert thinned.sizes["draw"] == expected_draws
    assert thinned.sizes["chain"] == datatree.posterior.sizes["chain"]


def test_thin_with_coords(datatree):
    """Test thinning with coordinate selection."""
    hierarchies = [0, 4]
    thinned = thin(datatree, factor=5, coords={"hierarchy": hierarchies})
    assert thinned.sizes["hierarchy"] == 2
    assert all(hierarchy in thinned.hierarchy.values for hierarchy in hierarchies)


def test_thin_with_var_names(datatree):
    """Test thinning with variable name selection."""
    var_names = ["mu", "tau"]

    thinned = thin(datatree, factor=5, var_names=var_names)
    for var in var_names:
        assert var in thinned

    for var in datatree.posterior.data_vars:
        if var not in var_names:
            assert var not in thinned


def test_thin_auto_factor(datatree):
    """Test thinning with automatic factor determination."""
    thinned = thin(datatree, factor="auto")
    assert thinned.sizes["draw"] <= datatree.posterior.sizes["draw"]


def test_thin_error_cases(datatree):
    """Test error cases for thin function."""
    data = datatree.posterior["mu"]

    with pytest.raises(ValueError, match="factor must be greater than 1"):
        thin(data, factor=0)
