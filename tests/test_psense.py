import os

from arviz_base import convert_to_datatree
from arviz_stats import psense, psense_summary
from numpy import isclose
from numpy.testing import assert_almost_equal

file_path = os.path.join(os.path.dirname(__file__), "univariate_normal.nc")
uni_dt = convert_to_datatree(file_path)


def test_psense():
    assert_almost_equal(psense(uni_dt, group="log_prior").to_array(), [0.404, 0.293], decimal=3)
    assert_almost_equal(
        psense(uni_dt, group="log_likelihood").to_array(), [0.575, 0.535], decimal=3
    )


def test_psense_var_names():
    result_0 = psense(uni_dt, group="log_prior", component_var_names=["mu"], var_names=["mu"])
    result_1 = psense(uni_dt, group="log_prior", var_names=["mu"])
    for result in (result_0, result_1):
        assert "sigma" != result.name
        assert "mu" == result.name
    assert not isclose(result_0, result_1)


def test_psense_summary():
    psense_df = psense_summary(uni_dt)
    assert all(psense_df.index == ["mu", "sigma"])
    assert all(psense_df["prior"] == [0.404, 0.293])
    assert all(psense_df["likelihood"] == [0.575, 0.535])
    assert all(psense_df["diagnosis"] == "prior-data conflict")
