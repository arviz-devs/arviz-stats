# pylint: disable=redefined-outer-name
"""Test accessors.

Accessor methods are very short, with the bulk of the computation/processing
handled by private methods. Testing this shared infrastructural methods
is the main goal of this module even if it does so via specific "regular" methods.
"""

import numpy as np
import pytest

from .helpers import importorskip

azb = importorskip("arviz_base")
xr = importorskip("xarray")

from arviz_stats.accessors import check_var_name_subset, update_dims, update_kwargs_with_dims


@pytest.fixture(scope="module")
def idata():
    return azb.from_dict(
        {
            "posterior": {
                "a": np.random.normal(size=(4, 100)),
                "b": np.random.normal(size=(4, 100, 3)),
            },
            "posterior_predictive": {
                "y": np.random.normal(size=(4, 100, 7)),
            },
        }
    )


def test_accessors_available(idata):
    assert hasattr(idata, "azstats")
    assert hasattr(idata.posterior.ds, "azstats")
    assert hasattr(idata.posterior["a"], "azstats")


def test_datatree_single_group(idata):
    out = idata.azstats.ess(group="posterior")
    assert isinstance(out, xr.DataTree)
    assert not out.children
    assert out.name == "posterior"


def test_datatree_multiple_groups(idata):
    out = idata.azstats.ess(group=["posterior", "posterior_predictive"])
    assert isinstance(out, xr.DataTree)
    assert len(out.children) == 2
    assert "posterior" in out.children
    assert "posterior_predictive" in out.children


def test_dataset_accessor(idata):
    out = idata.posterior.ds.azstats.ess()
    assert isinstance(out, xr.Dataset)
    assert "a" in out.data_vars
    assert "b" in out.data_vars


def test_dataarray_accessor(idata):
    out = idata.posterior["a"].azstats.ess()
    assert isinstance(out, xr.DataArray)
    assert out.shape == ()


@pytest.mark.parametrize("method", ["eti", "hdi", "ess", "rhat", "mcse", "mode"])
def test_accessor_methods(idata, method):
    out = getattr(idata.azstats, method)(group="posterior")
    assert isinstance(out, xr.DataTree)
    assert out.name == "posterior"


@pytest.mark.parametrize("filter_vars", [None, "like", "regex"])
def test_dataset_filter_vars(idata, filter_vars):
    ds = idata.posterior.ds
    if filter_vars == "like":
        filtered = ds.azstats.filter_vars(var_names=["a"], filter_vars=filter_vars)
        assert "a" in filtered.ds.data_vars
    elif filter_vars == "regex":
        filtered = ds.azstats.filter_vars(var_names=["^a"], filter_vars=filter_vars)
        assert "a" in filtered.ds.data_vars
    else:
        filtered = ds.azstats.filter_vars(var_names=["a"], filter_vars=filter_vars)
        assert "a" in filtered.ds.data_vars
        assert "b" not in filtered.ds.data_vars


@pytest.mark.parametrize("filter_vars", [None, "like", "regex"])
def test_datatree_filter_vars(idata, filter_vars):
    if filter_vars == "like":
        filtered = idata.azstats.filter_vars(
            group="posterior", var_names=["a"], filter_vars=filter_vars
        )
        assert "a" in filtered.datatree.ds.data_vars
    elif filter_vars == "regex":
        filtered = idata.azstats.filter_vars(
            group="posterior", var_names=["^a"], filter_vars=filter_vars
        )
        assert "a" in filtered.datatree.ds.data_vars
    else:
        filtered = idata.azstats.filter_vars(
            group="posterior", var_names=["a"], filter_vars=filter_vars
        )
        assert "a" in filtered.datatree.ds.data_vars
        assert "b" not in filtered.datatree.ds.data_vars


def test_datatree_filter_vars_none(idata):
    filtered = idata.azstats.filter_vars(group="posterior", var_names=None)
    assert filtered.datatree is idata
    assert "a" in idata.posterior.ds.data_vars


def test_datatree_filter_vars_multiple_groups(idata):
    filtered = idata.azstats.filter_vars(group=["posterior"], var_names=["a"])
    assert isinstance(filtered.datatree, xr.DataTree)
    assert "a" in filtered.datatree["posterior"].ds.data_vars


def test_dataset_thin_factor(idata):
    factor = idata.posterior.ds.azstats.thin_factor()
    assert isinstance(factor, int)
    assert factor >= 1


def test_dataset_thin(idata):
    thinned = idata.posterior.ds.azstats.thin(sample_dims="draw", factor=2)
    assert isinstance(thinned, xr.Dataset)
    assert thinned["a"].sizes["draw"] == 50


def test_dataset_thin_auto(idata):
    thinned = idata.posterior.ds.azstats.thin(factor="auto")
    assert isinstance(thinned, xr.Dataset)


def test_dataarray_thin(idata):
    thinned = idata.posterior["a"].azstats.thin(sample_dims="draw", factor=2)
    assert isinstance(thinned, xr.DataArray)
    assert thinned.sizes["draw"] == 50


def test_datatree_thin_factor(idata):
    factor = idata.azstats.thin_factor(group="posterior")
    assert isinstance(factor, int)
    assert factor >= 1


def test_datatree_thin_factor_invalid_group(idata):
    with pytest.raises(ValueError, match="single group"):
        idata.azstats.thin_factor(group=["posterior", "posterior_predictive"])


def test_datatree_thin(idata):
    thinned = idata.azstats.thin(group="posterior", sample_dims="draw", factor=2)
    assert isinstance(thinned, xr.DataTree)
    assert thinned.ds["a"].sizes["draw"] == 50


def test_datatree_ecdf_not_implemented(idata):
    with pytest.raises(NotImplementedError, match="DataTree ecdf not available"):
        idata.azstats.ecdf(group="posterior")


@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("ess", {"method": "bulk"}),
        ("ess", {"method": "tail", "prob": 0.95}),
        ("rhat", {"method": "rank"}),
        ("mcse", {"method": "mean"}),
    ],
)
def test_accessor_method_parameters(idata, method, kwargs):
    out = getattr(idata.posterior.ds.azstats, method)(**kwargs)
    assert isinstance(out, xr.Dataset)


def test_dataset_ds_property(idata):
    accessor = idata.posterior.ds.azstats
    assert accessor.ds.equals(idata.posterior.ds)
    assert accessor.dataset.equals(idata.posterior.ds)


def test_datatree_datatree_property(idata):
    accessor = idata.azstats
    assert accessor.datatree is idata


@pytest.mark.parametrize(
    "dims,da_dims,expected",
    [
        (None, ["chain", "draw"], None),
        ("chain", ["chain", "draw"], ["chain"]),
        (["chain", "draw"], ["chain", "draw"], ["chain", "draw"]),
        (["chain", "missing"], ["chain", "draw"], ["chain"]),
    ],
)
def test_update_dims(dims, da_dims, expected):
    da = xr.DataArray(np.random.randn(4, 100), dims=da_dims)
    result = update_dims(dims, da)
    assert result == expected


def test_update_kwargs_with_dims():
    da = xr.DataArray(np.random.randn(4, 100), dims=["chain", "draw"])
    kwargs = {"dim": ["chain", "missing"], "other_param": 42}
    result = update_kwargs_with_dims(da, kwargs)
    assert result["dim"] == ["chain"]
    assert result["other_param"] == 42


def test_datatree_group_warning(idata):
    with pytest.warns(UserWarning, match="doesn't match the group argument"):
        idata["posterior"].azstats.ess(group="prior")


@pytest.mark.parametrize(
    "method",
    [
        "hdi",
        "eti",
        "kde",
        "qds",
        "get_bins",
        "histogram",
        "compute_ranks",
        "pareto_min_ss",
        "psislw",
        "pareto_khat",
        "power_scale_lw",
        "autocorr",
    ],
)
def test_dataset_statistical_methods(idata, method):
    out = getattr(idata.posterior.ds.azstats, method)()
    assert isinstance(out, xr.Dataset | tuple)


def test_rhat_nested_dataset(idata):
    superchain_ids = np.array([0, 0, 1, 1])
    out = idata.posterior.ds.azstats.rhat_nested(superchain_ids=superchain_ids)
    assert isinstance(out, xr.Dataset)
    assert "a" in out.data_vars
    assert "b" in out.data_vars


def test_rhat_nested_datatree(idata):
    superchain_ids = np.array([0, 0, 1, 1])
    out = idata.azstats.rhat_nested(group="posterior", superchain_ids=superchain_ids)
    assert isinstance(out, xr.DataTree)
    assert out.name == "posterior"


def test_rhat_nested_dataarray(idata):
    superchain_ids = np.array([0, 0, 1, 1])
    out = idata.posterior["a"].azstats.rhat_nested(superchain_ids=superchain_ids)
    assert isinstance(out, xr.DataArray)
    assert out.shape == ()


@pytest.mark.parametrize("method", ["rank", "split", "folded"])
def test_rhat_nested_method(idata, method):
    superchain_ids = np.array([0, 0, 1, 1])
    out = idata.posterior.ds.azstats.rhat_nested(superchain_ids=superchain_ids, method=method)
    assert isinstance(out, xr.Dataset)
    assert "a" in out.data_vars


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_power_scale_sense_dataset(idata):
    lower_w = idata.posterior.ds["a"]
    upper_w = idata.posterior.ds["a"]
    out = idata.posterior.ds.azstats.power_scale_sense(
        lower_w=lower_w, upper_w=upper_w, lower_alpha=0.8, upper_alpha=0.9
    )
    assert isinstance(out, xr.Dataset)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_power_scale_sense_dataarray(idata):
    da = idata.posterior["a"]
    out = da.azstats.power_scale_sense(lower_w=da, upper_w=da, lower_alpha=0.8, upper_alpha=0.9)
    assert isinstance(out, xr.DataArray)


@pytest.mark.parametrize("reduce_func", ["mean", "min"])
def test_dataset_thin_factor_reduce_func(idata, reduce_func):
    factor = idata.posterior.ds.azstats.thin_factor(reduce_func=reduce_func)
    assert isinstance(factor, int)
    assert factor >= 1


@pytest.mark.parametrize("reduce_func", ["mean", "min"])
def test_datatree_thin_factor_reduce_func(idata, reduce_func):
    factor = idata.azstats.thin_factor(group="posterior", reduce_func=reduce_func)
    assert isinstance(factor, int)
    assert factor >= 1


def test_dataset_thin_factor_invalid_reduce_func(idata):
    with pytest.raises(ValueError, match="not recognized"):
        idata.posterior.ds.azstats.thin_factor(reduce_func="invalid")


def test_datatree_thin_factor_invalid_reduce_func(idata):
    with pytest.raises(ValueError, match="not recognized"):
        idata.azstats.thin_factor(group="posterior", reduce_func="invalid")


def test_check_var_name_subset_dataset(idata):
    result = check_var_name_subset(idata.posterior.ds, "a")
    assert isinstance(result, xr.DataArray)
    assert result.name == "a"


def test_check_var_name_subset_datatree(idata):
    result = check_var_name_subset(idata.posterior, "a")
    assert isinstance(result, xr.DataArray)
    assert result.name == "a"


def test_check_var_name_subset_dataarray(idata):
    da = idata.posterior["a"]
    result = check_var_name_subset(da, "a")
    assert result is da


def test_datatree_process_input_invalid_group(idata):
    accessor = idata.azstats
    with pytest.raises(ValueError, match="Group.*not available"):
        accessor._process_input("invalid_group", "test_method", allow_non_matching=False)  # pylint: disable=protected-access


def test_update_kwargs_with_dims_both_dim_and_dims():
    da = xr.DataArray(np.random.randn(4, 100), dims=["chain", "draw"])
    kwargs = {"dim": ["chain"], "dims": ["draw", "missing"], "other": 123}
    result = update_kwargs_with_dims(da, kwargs)
    assert result["dim"] == ["chain"]
    assert result["dims"] == ["draw"]
    assert result["other"] == 123
