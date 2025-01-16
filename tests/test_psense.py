import os

from arviz_base import convert_to_datatree
from numpy import isclose
from numpy.testing import assert_almost_equal

from arviz_stats import psense, psense_summary

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