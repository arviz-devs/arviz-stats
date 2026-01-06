"""Test for general computational backend agnostic utilities."""

import inspect

import numpy as np
import pytest

from .helpers import importorskip

azb = importorskip("arviz_base")
xr = importorskip("xarray")

from arviz_stats.base.array import array_stats
from arviz_stats.base.dataarray import dataarray_stats
from arviz_stats.base.stats_utils import round_num
from arviz_stats.utils import (
    ELPDData,
    get_array_function,
    get_function,
    get_log_likelihood,
    get_log_likelihood_dataarray,
    get_log_likelihood_dataset,
    get_log_prior,
)


def test_get_function_str_module():
    azb.rcParams["stats.module"] = "base"
    func = get_function("eti")
    assert inspect.ismethod(func)
    assert func.__self__ is dataarray_stats


def test_get_function_obj_module():
    azb.rcParams["stats.module"] = dataarray_stats
    func = get_function("eti")
    assert inspect.ismethod(func)
    assert func.__self__ is dataarray_stats


def test_get_log_likelihood():
    idata = azb.from_dict(
        {
            "log_likelihood": {
                "y1": np.random.normal(size=(4, 100, 6)),
                "y2": np.random.normal(size=(4, 100, 8)),
            }
        }
    )
    lik1 = get_log_likelihood(idata, "y1")
    lik2 = get_log_likelihood(idata, "y2")
    assert lik1.shape == (4, 100, 6)
    assert lik2.shape == (4, 100, 8)


def test_get_log_likelihood_warning():
    idata = azb.from_dict(
        {
            "sample_stats": {
                "log_likelihood": np.random.normal(size=(4, 100, 6)),
            }
        }
    )
    with pytest.warns(DeprecationWarning):
        get_log_likelihood(idata)


def test_get_log_likelihood_no_var_name():
    idata = azb.from_dict(
        {
            "log_likelihood": {
                "y1": np.random.normal(size=(4, 100, 6)),
                "y2": np.random.normal(size=(4, 100, 8)),
            }
        }
    )
    with pytest.raises(TypeError, match="Found several"):
        get_log_likelihood(idata)


def test_get_log_likelihood_no_group():
    idata = azb.from_dict(
        {
            "posterior": {
                "a": np.random.normal(size=(4, 100)),
                "b": np.random.normal(size=(4, 100)),
            }
        }
    )
    with pytest.raises(TypeError, match="log likelihood not found"):
        get_log_likelihood(idata)


def test_elpddata_base():
    elpddata = ELPDData(
        kind="loo",
        elpd=-20,
        se=2,
        p=5.6,
        n_samples=1000,
        n_data_points=370,
        scale="log",
        warning=False,
        good_k=0.7,
    )

    printed = str(elpddata)
    assert "warning" not in printed
    assert "1000 posterior samples" in printed
    assert "370 observations" in printed


def test_get_function_invalid():
    azb.rcParams["stats.module"] = "base"
    with pytest.raises(KeyError, match="not available"):
        get_function("invalid_function_name")


def test_get_array_function_str_module():
    azb.rcParams["stats.module"] = "base"
    func = get_array_function("mode")
    assert inspect.ismethod(func)
    assert func.__self__ is array_stats


def test_get_array_function_invalid():
    azb.rcParams["stats.module"] = "base"
    with pytest.raises(KeyError, match="not available"):
        get_array_function("invalid_function_name")


def test_get_log_likelihood_invalid_var_name():
    idata = azb.from_dict({"log_likelihood": {"y1": np.random.normal(size=(4, 100, 6))}})
    with pytest.raises(TypeError, match="No log likelihood data named"):
        get_log_likelihood(idata, "invalid_var")


@pytest.mark.parametrize(
    "var_names,expected_vars",
    [(None, ["y1", "y2"]), ("y1", ["y1"]), (["y1", "y2"], ["y1", "y2"])],
)
def test_get_log_likelihood_dataset_var_names(var_names, expected_vars):
    idata = azb.from_dict(
        {
            "log_likelihood": {
                "y1": np.random.normal(size=(4, 100, 6)),
                "y2": np.random.normal(size=(4, 100, 8)),
            }
        }
    )
    ds = get_log_likelihood_dataset(idata, var_names=var_names)
    for var in expected_vars:
        assert var in ds.data_vars


def test_get_log_likelihood_dataarray():
    idata = azb.from_dict({"log_likelihood": {"y1": np.random.normal(size=(4, 100, 6))}})
    da = get_log_likelihood_dataarray(idata, var_name="y1")
    assert da.shape == (4, 100, 6)


def test_get_log_likelihood_dataarray_no_var_name_multiple():
    idata = azb.from_dict(
        {
            "log_likelihood": {
                "y1": np.random.normal(size=(4, 100, 6)),
                "y2": np.random.normal(size=(4, 100, 8)),
            }
        }
    )
    with pytest.raises(TypeError, match="Found several"):
        get_log_likelihood_dataarray(idata)


def test_get_log_likelihood_dataarray_invalid_var():
    idata = azb.from_dict({"log_likelihood": {"y1": np.random.normal(size=(4, 100, 6))}})
    with pytest.raises(TypeError, match="No log likelihood data named"):
        get_log_likelihood_dataarray(idata, var_name="invalid")


@pytest.mark.parametrize(
    "var_names,expected_vars,not_expected",
    [
        (None, ["mu", "sigma"], []),
        ("mu", ["mu"], ["sigma"]),
        (["mu", "sigma"], ["mu", "sigma"], []),
    ],
)
def test_get_log_prior_var_names(var_names, expected_vars, not_expected):
    idata = azb.from_dict(
        {
            "log_prior": {
                "mu": np.random.normal(size=(4, 100)),
                "sigma": np.random.normal(size=(4, 100)),
            }
        }
    )
    ds = get_log_prior(idata, var_names=var_names)
    for var in expected_vars:
        assert var in ds.data_vars
    for var in not_expected:
        assert var not in ds.data_vars


def test_get_log_prior_not_found():
    idata = azb.from_dict({"posterior": {"mu": np.random.normal(size=(4, 100))}})
    with pytest.raises(TypeError, match="log prior not found"):
        get_log_prior(idata)


def test_elpddata_with_warning():
    elpddata = ELPDData(
        kind="loo",
        elpd=-20,
        se=2,
        p=5.6,
        n_samples=1000,
        n_data_points=370,
        scale="log",
        warning=True,
        good_k=0.7,
    )
    printed = str(elpddata)
    assert "warning" in printed.lower()


def test_elpddata_with_pareto_k():
    pareto_k = xr.DataArray(
        np.array([0.3, 0.5, 0.8, 1.2]), dims=["observation"], coords={"observation": range(4)}
    )
    elpddata = ELPDData(
        kind="loo",
        elpd=-20,
        se=2,
        p=5.6,
        n_samples=1000,
        n_data_points=4,
        scale="log",
        warning=False,
        good_k=0.7,
        pareto_k=pareto_k,
    )
    printed = str(elpddata)
    assert "Pareto k diagnostic" in printed
    assert "good" in printed
    assert "bad" in printed


def test_elpddata_kfold():
    elpddata = ELPDData(
        kind="loo_kfold",
        elpd=-20,
        se=2,
        p=5.6,
        n_samples=1000,
        n_data_points=370,
        scale="log",
        warning=False,
        good_k=0.7,
        n_folds=10,
    )
    printed = str(elpddata)
    assert "10-fold" in printed
    assert "kfold" in printed


def test_elpddata_subsample():
    elpddata = ELPDData(
        kind="loo",
        elpd=-20,
        se=2,
        p=5.6,
        n_samples=1000,
        n_data_points=370,
        scale="log",
        warning=False,
        good_k=0.7,
        subsample_size=100,
        subsampling_se=0.5,
    )
    printed = str(elpddata)
    assert "subsampled" in printed
    assert "100" in printed


def test_elpddata_approx_posterior():
    elpddata = ELPDData(
        kind="loo",
        elpd=-20,
        se=2,
        p=5.6,
        n_samples=1000,
        n_data_points=370,
        scale="log",
        warning=False,
        good_k=0.7,
        approx_posterior=True,
    )
    printed = str(elpddata)
    assert "Posterior approximation" in printed


@pytest.mark.parametrize(
    "scale,expected", [("log", "elpd"), ("deviance", "deviance"), ("negative_log", "-elpd")]
)
def test_elpddata_scale(scale, expected):
    elpddata = ELPDData(
        kind="loo",
        elpd=-20,
        se=2,
        p=5.6,
        n_samples=1000,
        n_data_points=370,
        scale=scale,
        warning=False,
        good_k=0.7,
    )
    printed = str(elpddata)
    assert expected in printed


def test_elpddata_getitem_setitem():
    elpddata = ELPDData(
        kind="loo",
        elpd=-20,
        se=2,
        p=5.6,
        n_samples=1000,
        n_data_points=370,
        scale="log",
        warning=False,
        good_k=0.7,
    )
    assert elpddata["elpd"] == -20
    assert elpddata["se"] == 2
    elpddata["elpd"] = -25
    assert elpddata.elpd == -25


def test_elpddata_repr():
    elpddata = ELPDData(
        kind="loo",
        elpd=-20,
        se=2,
        p=5.6,
        n_samples=1000,
        n_data_points=370,
        scale="log",
        warning=False,
        good_k=0.7,
    )
    assert repr(elpddata) == str(elpddata)


@pytest.mark.parametrize(
    "value,precision,expected",
    [
        (3.14159, 2, 3.14),
        (123.456, "3g", 123),
        (0.0, "3g", 0),
        (np.nan, 2, np.nan),
        (3.14159, None, 3.14159),
    ],
)
def test_round_num(value, precision, expected):
    result = round_num(value, precision)
    if np.isnan(expected):
        assert np.isnan(result)
    else:
        assert result == expected


def test_round_num_array():
    result = round_num(np.array([3.14159]), 2)
    assert result == 3.14


def test_round_num_dataarray():
    da = xr.DataArray([3.14159])
    result = round_num(da, 2)
    assert result == 3.14
