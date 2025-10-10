# pylint: disable=redefined-outer-name, unused-import
# ruff: noqa: F811
"""Test for thinning function."""

import numpy as np
import pytest

from .helpers import datatree, importorskip  # noqa: F401

azb = importorskip("arviz_base")
xr = importorskip("xarray")

from arviz_stats.manipulation import thin


@pytest.mark.parametrize("factor", [2, 5, 10])
def test_thin_numpy(factor):
    rng = np.random.default_rng(42)
    data = rng.normal(size=(4, 100, 3))
    thinned = thin(data, factor=factor, chain_axis=0, draw_axis=1)
    assert thinned.shape == (4, 100 // factor, 3)


def test_thin_numpy_large_factor():
    rng = np.random.default_rng(42)
    data = rng.normal(size=(4, 100, 3))
    thinned = thin(data, factor=200, chain_axis=0, draw_axis=1)
    assert thinned.shape == (4, 1, 3)


def test_thin_numpy_list():
    data = [[1, 2, 3, 4], [5, 6, 7, 8]]
    thinned = thin(data, factor=2, chain_axis=0, draw_axis=1)
    assert thinned.shape == (2, 2)


def test_thin_numpy_tuple():
    data = ((1, 2, 3, 4), (5, 6, 7, 8))
    thinned = thin(data, factor=2, chain_axis=0, draw_axis=1)
    assert thinned.shape == (2, 2)


@pytest.mark.parametrize("factor", [2, 5, 10])
def test_thin_dataarray(datatree, factor):
    data = datatree.posterior.theta
    thinned = thin(data, factor=factor)
    assert thinned.sizes["chain"] == data.sizes["chain"]
    assert thinned.sizes["draw"] == data.sizes["draw"] // factor


def test_thin_dataarray_with_coords(datatree):
    data = datatree.posterior.theta
    hierarchies = [0, 4]
    thinned = thin(data, factor=5, coords={"hierarchy": hierarchies})
    assert thinned.sizes["hierarchy"] == 2
    assert all(h in thinned.hierarchy.values for h in hierarchies)


@pytest.mark.parametrize("factor", [2, 5, 10])
def test_thin_dataset(datatree, factor):
    data = datatree.posterior
    thinned = thin(data, factor=factor)
    assert thinned.sizes["chain"] == data.sizes["chain"]
    assert thinned.sizes["draw"] == data.sizes["draw"] // factor
    for var in data.data_vars:
        assert var in thinned


def test_thin_dataset_with_coords(datatree):
    data = datatree.posterior
    hierarchies = [0, 4]
    thinned = thin(data, factor=5, coords={"hierarchy": hierarchies})
    assert thinned.sizes["hierarchy"] == 2


@pytest.mark.parametrize("factor", [2, 5, 10])
def test_thin_datatree(datatree, factor):
    original_draws = datatree.posterior.sizes["draw"]
    thinned = thin(datatree, factor=factor)
    expected_draws = original_draws // factor
    assert thinned.sizes["draw"] == expected_draws
    assert thinned.sizes["chain"] == datatree.posterior.sizes["chain"]


def test_thin_datatree_with_coords(datatree):
    hierarchies = [0, 4]
    thinned = thin(datatree, factor=5, coords={"hierarchy": hierarchies})
    assert thinned.sizes["hierarchy"] == 2
    assert all(h in thinned.hierarchy.values for h in hierarchies)


def test_thin_datatree_var_names(datatree):
    var_names = ["mu", "tau"]
    thinned = thin(datatree, factor=5, var_names=var_names)
    for var in var_names:
        assert var in thinned
    for var in datatree.posterior.data_vars:
        if var not in var_names:
            assert var not in thinned


def test_thin_datatree_single_var(datatree):
    thinned = thin(datatree, factor=5, var_names="mu")
    assert "mu" in thinned
    assert "tau" not in thinned


def test_thin_auto_factor(datatree):
    thinned = thin(datatree, factor="auto")
    assert thinned.sizes["draw"] <= datatree.posterior.sizes["draw"]


def test_thin_different_sample_dims(datatree):
    thinned = thin(datatree.posterior, factor=2, sample_dims=["draw"])
    assert thinned.sizes["draw"] == datatree.posterior.sizes["draw"] // 2


def test_thin_invalid_factor(datatree):
    data = datatree.posterior["mu"]
    with pytest.raises(ValueError, match="factor must be greater than 1"):
        thin(data, factor=0)


def test_thin_negative_factor(datatree):
    data = datatree.posterior["mu"]
    with pytest.raises(ValueError, match="factor must be greater than 1"):
        thin(data, factor=-5)
