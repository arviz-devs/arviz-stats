"""Test summary."""

# pylint: disable=redefined-outer-name
from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from .helpers import importorskip

azb = importorskip("arviz_base")
xr = importorskip("xarray")

from arviz_stats import ci_in_rope, eti, hdi, mode, qds, summary


def test_summary_ndarray():
    rng = np.random.default_rng(42)
    array = rng.normal(size=(4, 100, 2))
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
    "eti89_lb",
    "eti89_ub",
    "ess_bulk",
    "ess_tail",
    "r_hat",
    "mcse_mean",
    "mcse_sd",
    "median",
    "mad",
    "eti89_lb",
    "eti89_ub",
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


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
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


def test_mode(datatree):
    result = mode(datatree)
    assert result["mu"].shape == ()
    assert result["theta"].shape == (7,)
    result = mode(datatree.posterior)
    assert result["mu"].shape == ()
    assert result["theta"].shape == (7,)
    result = mode(datatree.posterior["mu"])
    assert result.shape == ()
    result = mode(datatree.posterior["mu"].values)
    assert result.shape == ()


@pytest.mark.parametrize("filter_vars", [None, "like", "regex"])
def test_summary_filter_vars(datatree, filter_vars):
    if filter_vars == "like":
        summary_df = summary(datatree, var_names=["mu"], filter_vars=filter_vars)
        assert "mu" in summary_df.index
    elif filter_vars == "regex":
        summary_df = summary(datatree, var_names=["^mu"], filter_vars=filter_vars)
        assert "mu" in summary_df.index
    else:
        summary_df = summary(datatree, var_names=["mu"], filter_vars=filter_vars)
        assert "mu" in summary_df.index


def test_summary_coords(datatree):
    summary_df = summary(datatree, coords={"hierarchy": [0, 1, 2]}, var_names=["theta"])
    assert len(summary_df.index) == 3


def test_summary_sample_dims(datatree):
    summary_df = summary(datatree, sample_dims=["chain", "draw"], var_names=["mu"])
    assert "mu" in summary_df.index
    assert "mean" in summary_df.columns


@pytest.mark.parametrize("ci_prob", [0.5, 0.89, 0.95])
def test_summary_ci_prob(datatree, ci_prob):
    summary_df = summary(datatree, ci_prob=ci_prob, var_names=["mu"])
    ci_perc = int(ci_prob * 100)
    assert f"eti{ci_perc}_lb" in summary_df.columns
    assert f"eti{ci_perc}_ub" in summary_df.columns


@pytest.mark.parametrize("ci_kind", ["hdi", "eti"])
def test_summary_ci_kind(datatree, ci_kind):
    summary_df = summary(datatree, ci_kind=ci_kind, ci_prob=0.89, var_names=["mu"])
    assert f"{ci_kind}89_lb" in summary_df.columns
    assert f"{ci_kind}89_ub" in summary_df.columns


@pytest.mark.parametrize("round_to", [0, 2, 4, "none"])
def test_summary_round_to(datatree, round_to):
    summary_df = summary(datatree, round_to=round_to, var_names=["mu"])
    assert "mean" in summary_df.columns
    if round_to == "none":
        assert summary_df["mean"].iloc[0] != round(summary_df["mean"].iloc[0], 2)


def test_summary_invalid_ci_kind():
    rng = np.random.default_rng(42)
    array = rng.normal(size=(4, 100, 2))
    with pytest.raises(ValueError, match="ci_kind must be either 'hdi' or 'eti'"):
        summary(array, ci_kind="invalid")


def test_summary_invalid_kind():
    rng = np.random.default_rng(42)
    array = rng.normal(size=(4, 100, 2))
    with pytest.raises(ValueError, match="valid options for kind are"):
        summary(array, kind="invalid")


def test_summary_median_stats_ci_kind(datatree):
    summary_df = summary(datatree, kind="stats_median", ci_kind="hdi", var_names=["mu"])
    assert "eti89_lb" in summary_df.columns
    assert "median" in summary_df.columns


def test_rope_dataset_input(fake_dt):
    rope_ds = xr.Dataset({"a": (["rope_dim"], [-1, 1]), "b": (["rope_dim"], [-2, 2])})
    result = ci_in_rope(fake_dt, rope=rope_ds, var_names=["a", "b"])
    assert result["a"] > 60
    assert result["b"] > 90


def test_rope_invalid_keys(fake_dt):
    with pytest.raises(ValueError, match="rope` must be a subset of the variables"):
        ci_in_rope(fake_dt, rope={"invalid_var": (-1, 1)})


def test_rope_invalid_values(fake_dt):
    with pytest.raises(ValueError, match="rope` must be a dict of tuples of length 2"):
        ci_in_rope(fake_dt, rope={"a": (-1,)})


def test_rope_invalid_rope_dim(fake_dt):
    rope_ds = xr.Dataset({"a": (["wrong_dim"], [-1, 1])})
    with pytest.raises(ValueError, match="rope_dim is not a dimension"):
        ci_in_rope(fake_dt, rope=rope_ds, rope_dim="rope_dim")


def test_rope_invalid_rope_length(fake_dt):
    rope_ds = xr.Dataset({"a": (["rope_dim"], [-1, 0, 1])})
    with pytest.raises(ValueError, match="Length of rope_dim dim must be 2"):
        ci_in_rope(fake_dt, rope=rope_ds)


def test_rope_with_dim(fake_dt):
    result = ci_in_rope(fake_dt, var_names=["a"], rope=(-1, 1), dim=["chain", "draw"])
    assert "a" in result.data_vars
    assert result["a"] > 60


@pytest.mark.parametrize("ci_kind", ["hdi", "eti"])
def test_rope_ci_kind(fake_dt, ci_kind):
    result = ci_in_rope(fake_dt, var_names=["a"], rope=(-1, 1), ci_kind=ci_kind, ci_prob=0.89)
    assert "a" in result.data_vars


def test_rope_filter_vars(fake_dt):
    result = ci_in_rope(fake_dt, var_names=["a"], rope=(-1, 1), filter_vars="like")
    assert "a" in result.data_vars


def test_mode_filter_vars(datatree):
    result = mode(datatree, var_names=["mu"], filter_vars="like")
    assert "mu" in result.data_vars


def test_mode_coords(datatree):
    result = mode(datatree, var_names=["theta"], coords={"hierarchy": [0, 1, 2]})
    assert result["theta"].shape == (3,)


def test_mode_array_input():
    rng = np.random.default_rng(42)
    array = rng.normal(size=(4, 100, 2))
    result = mode(array)
    assert result.shape == ()
    assert isinstance(result.item(), float)


def test_mode_integer_data():
    rng = np.random.default_rng(42)
    int_array = rng.integers(0, 10, size=(4, 100, 5))
    result = mode(int_array)
    assert result.shape == ()
    assert isinstance(result.item(), int | float)


def test_hdi_prob(datatree):
    result = hdi(datatree, prob=0.89)
    assert result["mu"].shape == (2,)
    assert result["theta"].shape == (7, 2)


def test_eti_prob(datatree):
    result = eti(datatree, prob=0.89)
    assert result["mu"].shape == (2,)
    assert result["theta"].shape == (7, 2)


def test_qds_nquantiles(datatree):
    result = qds(datatree, nquantiles=25)
    assert result["mu"].shape == (2, 25)
    assert result["theta"].shape == (2, 7, 25)


def test_summary_dataarray_input(datatree):
    summary_df = summary(datatree.posterior["mu"])
    assert "mean" in summary_df.columns
    assert len(summary_df.index) == 1


def test_summary_dataset_input(datatree):
    summary_df = summary(datatree.posterior)
    assert "mean" in summary_df.columns
    assert len(summary_df.index) > 0


def test_ci_in_rope_array_rope(fake_dt):
    result = ci_in_rope(fake_dt, var_names=["a"], rope=np.array([-1, 1]))
    assert "a" in result.data_vars
    assert result["a"] > 60


def test_hdi_empty_coords(datatree):
    result = hdi(datatree, var_names=["mu"], coords={})
    assert result["mu"].shape == (2,)


def test_eti_empty_coords(datatree):
    result = eti(datatree, var_names=["tau"], coords={})
    assert result["tau"].shape == (2,)


def test_summary_single_var(datatree):
    summary_df = summary(datatree, var_names=["mu"])
    assert len(summary_df.index) == 1
    assert "mu" in summary_df.index


def test_summary_small_data():
    rng = np.random.default_rng(42)
    array = rng.normal(size=(2, 10))
    summary_df = summary(array)
    assert "mean" in summary_df.columns
    assert "ess_bulk" in summary_df.columns


def test_hdi_filter_vars_multiple(datatree):
    result = hdi(datatree, var_names=["mu", "tau"], filter_vars=None)
    assert "mu" in result.data_vars
    assert "tau" in result.data_vars


def test_qds_small_nquantiles(datatree):
    result = qds(datatree, nquantiles=5, var_names=["mu"])
    assert result["mu"].shape == (2, 5)


def test_mode_single_value_array():
    array = np.ones((4, 100))
    result = mode(array)
    assert result.shape == ()
    assert result.item() == 1.0


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_summary_zero_variance():
    array = np.ones((4, 100, 2))
    summary_df = summary(array)
    assert summary_df["sd"].iloc[0] == 0.0


@pytest.mark.parametrize("prob", [0.5, 0.89, 0.95])
def test_hdi_prob_parameter(datatree, prob):
    result = hdi(datatree, prob=prob, var_names=["mu"])
    assert result["mu"].shape == (2,)


@pytest.mark.parametrize("prob", [0.5, 0.89, 0.95])
def test_eti_prob_parameter(datatree, prob):
    result = eti(datatree, prob=prob, var_names=["tau"])
    assert result["tau"].shape == (2,)


@pytest.mark.parametrize("fmt", ["wide", "long", "xarray"])
def test_summary_fmt(datatree, fmt):
    result = summary(datatree, fmt=fmt, var_names=["mu"])
    if fmt == "wide":
        assert "mu" in result.index
        assert "mean" in result.columns
    elif fmt == "long":
        assert "mean" in result.index
        assert "mu" in result.columns
    else:
        assert isinstance(result, xr.Dataset)
        assert "summary" in result.dims
