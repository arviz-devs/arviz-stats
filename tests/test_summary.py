# pylint: disable=redefined-outer-name
from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from .helpers import importorskip

azb = importorskip("arviz_base")

from arviz_stats.summary import ci_in_rope, summary


@pytest.fixture(scope="session")
def centered_eight():
    centered_eight = azb.load_arviz_data("centered_eight")
    return centered_eight


@pytest.fixture
def fake_post():
    return azb.from_dict(
        {
            "posterior": {
                "a": np.random.normal(size=(4, 100)),
                "b": np.random.normal(size=(4, 100)),
            },
        }
    )


def test_summary_ndarray():
    array = np.random.randn(4, 100, 2)
    summary_df = summary(array)
    assert summary_df.shape


@pytest.mark.parametrize("var_names_expected", ((None, 10), ("mu", 1), (["mu", "tau"], 2)))
def test_summary_var_names(centered_eight, var_names_expected):
    var_names, expected = var_names_expected
    summary_df = summary(centered_eight, var_names=var_names)
    assert len(summary_df.index) == expected


def test_summary_group_argument(centered_eight):
    summary_df_posterior = summary(centered_eight, group="posterior")
    summary_df_prior = summary(centered_eight, group="prior")
    assert list(summary_df_posterior.index) != list(summary_df_prior.index)


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
def test_summary_focus_kind(centered_eight, params):
    kind, metrics_names_ = params
    summary_df = summary(centered_eight, kind=kind)
    assert_array_equal(summary_df.columns, metrics_names_)


def test_summary_nan(centered_eight):
    centered_eight = deepcopy(centered_eight)
    centered_eight.posterior["theta"].loc[{"school": "Deerfield"}] = np.nan
    summary_xarray = summary(centered_eight, skipna=True)
    assert summary_xarray is not None
    assert summary_xarray.loc["theta[Deerfield]"].isnull().all()
    assert (
        summary_xarray.loc[[ix for ix in summary_xarray.index if ix != "theta[Deerfield]"]]
        .notnull()
        .all()
        .all()
    )


def test_summary_skip_nan(centered_eight):
    centered_eight = deepcopy(centered_eight)
    centered_eight.posterior["theta"].loc[{"draw": slice(10), "school": "Deerfield"}] = np.nan
    summary_xarray = summary(centered_eight, skipna=True)
    theta_1 = summary_xarray.loc["theta[Deerfield]"].isnull()
    assert summary_xarray is not None
    assert ~theta_1[:4].all()
    assert theta_1[4:].all()


def test_rope_single(fake_post):
    result = ci_in_rope(fake_post, var_names=["a", "b"], rope=(-1, 1))
    assert all(result > 60)
    result = ci_in_rope(fake_post, var_names=["a", "b"], rope=(-1, 1), ci_prob=0.5, ci_kind="hdi")
    assert all(result == 100)
    assert "a" in result.data_vars
    assert "b" in result.data_vars


def test_rope_multiple(fake_post):
    result = ci_in_rope(fake_post, rope={"a": (-100, -99), "b": (-2, 2)})
    assert result["a"] == 0
    assert result["b"] > 90
    assert "a" in result.data_vars
    assert "b" in result.data_vars
