"""Test power-scaling sensitivity diagnostics."""

import os

import pytest
from numpy import isclose
from numpy.testing import assert_almost_equal

from .helpers import importorskip

azb = importorskip("arviz_base")

from arviz_stats import psense, psense_summary
from arviz_stats.psense import power_scale_dataset

file_path = os.path.join(os.path.dirname(__file__), "univariate_normal.nc")
uni_dt = azb.convert_to_datatree(file_path)


def test_psense():
    assert_almost_equal(psense(uni_dt, group="prior").to_array(), [0.404, 0.293], decimal=3)
    assert_almost_equal(psense(uni_dt, group="likelihood").to_array(), [0.575, 0.535], decimal=3)


def test_psense_var_names():
    result_0 = psense(uni_dt, group="prior", group_var_names=["mu"], var_names=["mu"])
    result_1 = psense(uni_dt, group="prior", var_names=["mu"])
    for result in (result_0, result_1):
        assert "sigma" not in result.data_vars
        assert "mu" in result.data_vars
    assert not isclose(result_0["mu"], result_1["mu"])


def test_psense_summary():
    psense_df = psense_summary(uni_dt)
    assert all(psense_df.index == ["mu", "sigma"])
    assert all(psense_df["prior"] == [0.404, 0.293])
    assert all(psense_df["likelihood"] == [0.576, 0.536])
    assert all(psense_df["diagnosis"] == "potential prior-data conflict")

    psense_df = psense_summary(uni_dt, threshold=1)
    assert all(psense_df["diagnosis"] == "✓")


@pytest.mark.parametrize("alphas", [(0.95, 1.05), (0.98, 1.02), (0.99, 1.01)])
def test_psense_alphas(alphas):
    result = psense(uni_dt, group="prior", alphas=alphas)
    assert "mu" in result.data_vars
    assert "sigma" in result.data_vars


@pytest.mark.parametrize("filter_vars", ["like", "regex"])
def test_psense_filter_vars(filter_vars):
    if filter_vars == "like":
        result = psense(uni_dt, group="prior", var_names=["mu"], filter_vars=filter_vars)
    else:
        result = psense(uni_dt, group="prior", var_names=["mu.*"], filter_vars=filter_vars)
    assert "mu" in result.data_vars


def test_psense_coords():
    result = psense(uni_dt, group="prior", coords={"chain": [0, 1]})
    assert "mu" in result.data_vars
    assert "sigma" in result.data_vars


@pytest.mark.parametrize("group", ["prior", "likelihood"])
def test_psense_groups(group):
    result = psense(uni_dt, group=group)
    assert "mu" in result.data_vars
    assert "sigma" in result.data_vars


@pytest.mark.parametrize("round_to", [1, 2, 3])
def test_psense_summary_round_to(round_to):
    psense_df = psense_summary(uni_dt, round_to=round_to)
    assert "prior" in psense_df.columns
    assert "likelihood" in psense_df.columns
    max_decimals = max(
        len(str(val).rsplit(".", maxsplit=1)[-1]) if "." in str(val) else 0
        for val in psense_df["prior"]
    )
    assert max_decimals <= round_to


def test_psense_summary_var_names():
    psense_df = psense_summary(uni_dt, var_names=["mu"])
    assert "mu" in psense_df.index
    assert "sigma" not in psense_df.index


def test_psense_summary_prior_var_names():
    psense_df = psense_summary(uni_dt, prior_var_names=["mu"])
    assert all(psense_df.index == ["mu", "sigma"])


def test_psense_summary_likelihood_var_names():
    psense_df = psense_summary(uni_dt, likelihood_var_names=["obs"])
    assert all(psense_df.index == ["mu", "sigma"])


def test_psense_summary_diagnosis_check_mark():
    psense_df = psense_summary(uni_dt, threshold=0.6)
    assert all(diag == "✓" for diag in psense_df["diagnosis"])


def test_power_scale_dataset_prior():
    result = power_scale_dataset(
        uni_dt,
        group="prior",
        alphas=(0.99, 1.01),
        sample_dims=["chain", "draw"],
        group_var_names=None,
        group_coords=None,
    )
    assert "alpha" in result.dims
    assert result.sizes["alpha"] == 3
    assert "mu" in result.data_vars


def test_power_scale_dataset_likelihood():
    result = power_scale_dataset(
        uni_dt,
        group="likelihood",
        alphas=(0.99, 1.01),
        sample_dims=["chain", "draw"],
        group_var_names=None,
        group_coords=None,
    )
    assert "alpha" in result.dims
    assert result.sizes["alpha"] == 3


def test_psense_invalid_group():
    with pytest.raises(ValueError, match="Value for `group` argument not recognized"):
        psense(uni_dt, group="invalid")


def test_psense_group_coords():
    result = psense(uni_dt, group="likelihood", group_coords={"obs_dim_0": [0, 1, 2, 3, 4]})
    assert "mu" in result.data_vars
    assert "sigma" in result.data_vars


def test_psense_summary_coords():
    psense_df = psense_summary(
        uni_dt,
        coords={"chain": [0, 1]},
        prior_coords={"chain": [0, 1]},
        likelihood_coords={"obs_dim_0": slice(0, 10)},
    )
    assert "prior" in psense_df.columns
    assert "likelihood" in psense_df.columns


def test_psense_summary_filter_vars():
    psense_df = psense_summary(uni_dt, var_names=["mu"], filter_vars="like")
    assert "mu" in psense_df.index
    assert "sigma" not in psense_df.index


def test_psense_summary_threshold_boundary():
    psense_df = psense_summary(uni_dt, threshold=0.404)
    assert "mu" in psense_df.index
    assert psense_df.loc["mu", "diagnosis"] == "potential prior-data conflict"
    assert psense_df.loc["sigma", "diagnosis"] == "✓"


def test_power_scale_dataset_with_var_names():
    result = power_scale_dataset(
        uni_dt,
        group="prior",
        alphas=(0.99, 1.01),
        sample_dims=["chain", "draw"],
        group_var_names=["mu"],
        group_coords=None,
    )
    assert "alpha" in result.dims
    assert result.sizes["alpha"] == 3


def test_power_scale_dataset_with_coords():
    result = power_scale_dataset(
        uni_dt,
        group="likelihood",
        alphas=(0.99, 1.01),
        sample_dims=["chain", "draw"],
        group_var_names=None,
        group_coords={"obs_dim_0": slice(0, 10)},
    )
    assert "alpha" in result.dims
    assert result.sizes["alpha"] == 3


def test_psense_sample_dims():
    result = psense(uni_dt, group="prior", sample_dims=["chain", "draw"])
    assert "mu" in result.data_vars
    assert "sigma" in result.data_vars


def test_psense_summary_alphas():
    psense_df = psense_summary(uni_dt, alphas=(0.95, 1.05))
    assert "prior" in psense_df.columns
    assert "likelihood" in psense_df.columns
