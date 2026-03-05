"""Test kwargs validation for accessor functions that accept **kwargs."""

# pylint: disable=redefined-outer-name
import numpy as np
import pytest

from .helpers import importorskip

azb = importorskip("arviz_base")
xr = importorskip("xarray")

import arviz_stats.accessors  # noqa: F401, E402


@pytest.mark.parametrize(
    "func",
    (
        "ess",
        "rhat",
        "rhat_nested",
        "mcse",
        "qds",
        "get_bins",
        "compute_ranks",
        "ecdf",
        "pareto_min_ss",
        "psislw",
        "bfmi",
        "pareto_khat",
        "loo_expectation",
        "loo_quantile",
        "power_scale_lw",
        "power_scale_sense",
        "autocorr",
        "mean",
        "median",
        "mode",
    ),
)
def test_accessor_kwargs_raise(datatree, func):
    accessor = datatree.posterior.ds.azstats
    with pytest.raises(TypeError, match=".*unexpected keyword argument.*"):
        getattr(accessor, func)(invalid_kwarg="value")


def test_loo_score_kwargs_raise(datatree):
    accessor = datatree.posterior.ds.azstats
    y_obs = xr.DataArray(np.random.randn(8))
    with pytest.raises(TypeError, match=".*unexpected keyword argument.*"):
        accessor.loo_score(y_obs, invalid_kwarg="value")


def test_loo_pit_kwargs_raise(datatree):
    accessor = datatree.posterior.ds.azstats
    y_obs = xr.DataArray(np.random.randn(8))
    with pytest.raises(TypeError, match=".*unexpected keyword argument.*"):
        accessor.loo_pit(y_obs, invalid_kwarg="value")


def test_loo_r2_kwargs_raise(datatree):
    accessor = datatree.posterior.ds.azstats
    ypred_loo = xr.DataArray(np.random.randn(8))
    with pytest.raises(TypeError, match=".*unexpected keyword argument.*"):
        accessor.loo_r2(ypred_loo, invalid_kwarg="value")
