import os

import arviz as az

# from arviz_base import convert_to_datatree
from arviz_base import convert_to_datatree
from numpy import isclose
from numpy.testing import assert_almost_equal

# from arviz_stats import psense, psense_summary
from arviz_stats import psense, psense_summary

from numpy import isclose
from numpy.testing import assert_almost_equal

import pytest
import numpy as np

# import InferenceData 
from src.arviz_stats.psense import bayes_factor


file_path = os.path.join(os.path.dirname(__file__), "univariate_normal.nc")
uni_dt = convert_to_datatree(file_path)


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
    assert all(psense_df["likelihood"] == [0.575, 0.535])
    assert all(psense_df["diagnosis"] == "prior-data conflict")

    psense_df = psense_summary(uni_dt, threshold=1)
    assert all(psense_df["diagnosis"] == "✓")


# Test Case a) Valid Input
@pytest.fixture
def example_idata():
    posterior_data = np.random.normal(0, 1, 1000)
    prior_data = np.random.normal(0, 1, 1000)

    idata = az.from_dict(
        posterior={"test_var": posterior_data},
        prior={"test_var": prior_data}
    )
    return idata


def test_bayes_factor_valid_input(example_idata):
    result = bayes_factor(example_idata, var_name="test_var")

    assert "BF10" in result
    assert "BF01" in result
    assert result["BF10"] > 0
    assert result["BF01"] > 0


# Test Case b) Reference Value Outside Posterior Range
def test_bayes_factor_ref_val_outside_range(example_idata):
    ref_val = 100
    result = bayes_factor(example_idata, var_name="test_var", ref_val=ref_val)

    assert "BF10" in result
    assert "BF01" in result


# Test Case c1) Continuous Data
def test_bayes_factor_continuous_data():
    posterior_data = np.random.normal(0, 1, 1000)
    prior_data = np.random.normal(0, 1, 1000)

    idata = az.from_dict(
        posterior={"test_var": posterior_data},
        prior={"test_var": prior_data}
    )
    result = bayes_factor(idata, var_name="test_var")

    assert "BF10" in result
    assert "BF01" in result


# Test Case c2) Discrete Data
def test_bayes_factor_discrete_data():
    posterior_data = np.random.choice([0, 1], size=1000)
    prior_data = np.random.choice([0, 1], size=1000)

    idata = az.from_dict(
        posterior={"test_var": posterior_data},
        prior={"test_var": prior_data}
    )
    result = bayes_factor(idata, var_name="test_var")

    assert "BF10" in result
    assert "BF01" in result


# Test Case d) Return Reference Values
def test_bayes_factor_return_ref_vals(example_idata):
    ref_val = 0
    result, ref_vals = bayes_factor(example_idata, var_name="test_var", return_ref_vals=True)

    assert "BF10" in result
    assert "BF01" in result
    assert "prior" in ref_vals
    assert "posterior" in ref_vals