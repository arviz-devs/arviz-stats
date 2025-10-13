# pylint: disable=redefined-outer-name, unused-import
# ruff: noqa: F811
import numpy as np

from .helpers import importorskip

azb = importorskip("arviz_base")
xr = importorskip("xarray")

from arviz_stats.sampling_diagnostics import ess, mcse, rhat, rhat_nested


def test_ess_datatree_returns_datatree(centered_eight):
    result = ess(centered_eight, group="posterior")
    assert isinstance(result, xr.DataTree)
    assert result.name == "posterior"


def test_ess_datatree_with_var_names(centered_eight):
    result = ess(centered_eight, group="posterior", var_names="mu")
    ds = result.dataset
    assert "mu" in ds.data_vars
    assert "theta" not in ds.data_vars


def test_ess_dataarray_returns_dataarray(centered_eight):
    da = centered_eight.posterior["mu"]
    result = ess(da)
    assert isinstance(result, xr.DataArray)


def test_ess_dataset_returns_dataset(centered_eight):
    ds = centered_eight.posterior.dataset
    result = ess(ds, var_names="mu")
    assert isinstance(result, xr.Dataset)
    assert "mu" in result.data_vars


def test_ess_list_input():
    data = [[1, 2, 3, 4], [5, 6, 7, 8]]
    result = ess(data)
    assert isinstance(result, float | np.floating | np.ndarray)


def test_ess_tuple_input():
    data = ([1, 2, 3, 4], [5, 6, 7, 8])
    result = ess(data)
    assert isinstance(result, float | np.floating | np.ndarray)


def test_rhat_datatree_returns_datatree(centered_eight):
    result = rhat(centered_eight, group="posterior")
    assert isinstance(result, xr.DataTree)
    assert result.name == "posterior"


def test_rhat_datatree_with_var_names(centered_eight):
    result = rhat(centered_eight, group="posterior", var_names="mu")
    ds = result.dataset
    assert "mu" in ds.data_vars
    assert "theta" not in ds.data_vars


def test_rhat_dataarray_returns_dataarray(centered_eight):
    da = centered_eight.posterior["mu"]
    result = rhat(da)
    assert isinstance(result, xr.DataArray)


def test_rhat_dataset_returns_dataset(centered_eight):
    ds = centered_eight.posterior.dataset
    result = rhat(ds, var_names="mu")
    assert isinstance(result, xr.Dataset)
    assert "mu" in result.data_vars


def test_rhat_list_input():
    data = [[1, 2, 3, 4], [5, 6, 7, 8]]
    result = rhat(data)
    assert isinstance(result, float | np.floating | np.ndarray)


def test_rhat_nested_datatree_returns_datatree(centered_eight):
    superchain_ids = [0, 0, 1, 1]
    result = rhat_nested(
        centered_eight, group="posterior", var_names="mu", superchain_ids=superchain_ids
    )
    assert isinstance(result, xr.DataTree)
    ds = result.dataset
    assert "mu" in ds.data_vars


def test_rhat_nested_datatree_with_var_names(centered_eight):
    superchain_ids = [0, 0, 1, 1]
    result = rhat_nested(
        centered_eight, group="posterior", var_names="mu", superchain_ids=superchain_ids
    )
    ds = result.dataset
    assert "mu" in ds.data_vars
    assert "theta" not in ds.data_vars


def test_rhat_nested_dataarray_returns_dataarray(centered_eight):
    da = centered_eight.posterior["mu"]
    superchain_ids = [0, 0, 1, 1]
    result = rhat_nested(da, superchain_ids=superchain_ids)
    assert isinstance(result, xr.DataArray)


def test_mcse_datatree_returns_datatree(centered_eight):
    result = mcse(centered_eight, group="posterior")
    assert isinstance(result, xr.DataTree)
    assert result.name == "posterior"


def test_mcse_datatree_with_var_names(centered_eight):
    result = mcse(centered_eight, group="posterior", var_names="mu")
    ds = result.dataset
    assert "mu" in ds.data_vars
    assert "theta" not in ds.data_vars


def test_mcse_dataarray_returns_dataarray(centered_eight):
    da = centered_eight.posterior["mu"]
    result = mcse(da)
    assert isinstance(result, xr.DataArray)


def test_mcse_dataset_returns_dataset(centered_eight):
    ds = centered_eight.posterior.dataset
    result = mcse(ds, var_names="mu")
    assert isinstance(result, xr.Dataset)
    assert "mu" in result.data_vars


def test_mcse_list_input():
    data = [[1, 2, 3, 4], [5, 6, 7, 8]]
    result = mcse(data)
    assert isinstance(result, float | np.floating | np.ndarray)
