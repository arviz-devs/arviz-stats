# pylint: disable=redefined-outer-name, unused-import
# ruff: noqa: F811
from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from .helpers import datatree, fake_dt, importorskip  # noqa: F401

azb = importorskip("arviz_base")

from arviz_stats import ci_in_rope, eti, hdi, qds, summary


def test_summary_ndarray():
    array = np.random.randn(4, 100, 2)
    summary_df = summary(array)
    assert summary_df.shape


@pytest.mark.parametrize("var_names_expected", ((None, 9), ("mu", 1), (["mu", "tau"], 2)))
def test_summary_var_names(datatree, var_names_expected):
    var_names, expected = var_names_expected
    summary_df = summary(datatree, var_names=var_names)
    assert len(summary_df.index) == expected


def test_summary_group_argument(datatree):
    summary_df_posterior = summary(datatree, group="posterior", kind="stats")
    summary_df_prior = summary(datatree, group="prior", kind="stats")
    assert list(summary_df_posterior["mean"]) != list(summary_df_prior["mean"])


METRICS_NAMES = [
    "mean",
    "sd",
    "eti94_lb",
    "eti94_ub",
    "ess_bulk",
    "ess_tail",
    "r_hat",
    "mcse_mean",
    "mcse_sd",
    "median",
    "mad",
    "eti94_lb",
    "eti94_ub",
    "ess_median",
    "ess_tail",
    "r_hat",
    "mcse_median",
    "mcse_mean",
    "ess_mean",
    "min_ss",
]


@pytest.mark.parametrize(
    "params",
    (
        ("all", METRICS_NAMES[:9]),
        ("stats", METRICS_NAMES[:4]),
        ("diagnostics", METRICS_NAMES[4:9]),
        ("all_median", METRICS_NAMES[9:17]),
        ("stats_median", METRICS_NAMES[9:13]),
        ("diagnostics_median", METRICS_NAMES[13:17]),
        ("mc_diagnostics", METRICS_NAMES[17:]),
    ),
)
def test_summary_focus_kind(datatree, params):
    kind, metrics_names_ = params
    summary_df = summary(datatree, kind=kind)
    assert_array_equal(summary_df.columns, metrics_names_)


def test_summary_nan(datatree):
    datatree = deepcopy(datatree)
    datatree.posterior["theta"].loc[{"hierarchy": 0}] = np.nan
    summary_xarray = summary(datatree, skipna=True)
    assert summary_xarray is not None
    assert summary_xarray.loc["theta[0]"].isnull().all()
    assert (
        summary_xarray.loc[[ix for ix in summary_xarray.index if ix != "theta[0]"]]
        .notnull()
        .all()
        .all()
    )


def test_summary_skip_nan(datatree):
    datatree = deepcopy(datatree)
    datatree.posterior["theta"].loc[{"draw": slice(10), "hierarchy": 0}] = np.nan
    summary_xarray = summary(datatree, skipna=True)
    theta_1 = summary_xarray.loc["theta[0]"].isnull()
    assert summary_xarray is not None
    assert ~theta_1[:4].all()
    assert theta_1[4:].all()


def test_rope_single(fake_dt):
    result = ci_in_rope(fake_dt, var_names=["a", "b"], rope=(-1, 1))
    assert all(result > 60)
    result = ci_in_rope(fake_dt, var_names=["a", "b"], rope=(-1, 1), ci_prob=0.5, ci_kind="hdi")
    assert all(result == 100)
    assert "a" in result.data_vars
    assert "b" in result.data_vars


def test_rope_multiple(fake_dt):
    result = ci_in_rope(fake_dt, rope={"a": (-100, -99), "b": (-2, 2)})
    assert result["a"] == 0
    assert result["b"] > 90
    assert "a" in result.data_vars
    assert "b" in result.data_vars


def test_hdi(datatree):
    result = hdi(datatree, prob=0.5)
    assert result["mu"].shape == (2,)
    assert result["theta"].shape == (7, 2)
    assert "ci_bound" in result.dims
    result = hdi(datatree.posterior, method="multimodal")
    assert result["mu"].shape == (1, 2)
    assert result["theta"].shape == (7, 1, 2)
    result = hdi(datatree.posterior["mu"])
    assert result.shape == (2,)
    result = hdi(datatree.posterior["mu"].values)
    assert result.shape == (4, 2)


def test_eti(datatree):
    result = eti(datatree, prob=0.5)
    assert result["mu"].shape == (2,)
    assert result["theta"].shape == (7, 2)
    assert "ci_bound" in result.dims
    result = eti(datatree.posterior, skipna=True)
    assert result["mu"].shape == (2,)
    assert result["theta"].shape == (7, 2)
    result = eti(datatree.posterior["mu"])
    assert result.shape == (2,)
    result = eti(datatree.posterior["mu"].values)
    assert result.shape == (4, 2)


def test_qds(datatree):
    result = qds(datatree, nquantiles=50)
    assert result["mu"].shape == (2, 50)
    assert result["theta"].shape == (2, 7, 50)
    assert "plot_axis" in result.dims
    result = qds(datatree.posterior)
    assert result["mu"].shape == (2, 100)
    assert result["theta"].shape == (2, 7, 100)
    result = qds(datatree.posterior["mu"])
    assert result.shape == (2, 100)
    result = qds(datatree.posterior["mu"].values)
    assert result[0].shape == (4, 100)
    assert result[2].shape == (4,)
