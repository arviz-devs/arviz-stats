# pylint: disable=redefined-outer-name, unused-import
# ruff: noqa: F811
import numpy as np
import pytest

from .helpers import fake_dt, importorskip  # noqa: F401

azb = importorskip("arviz_base")

from arviz_stats.bayes_factor import bayes_factor


def test_bayes_factor_default_prior(fake_dt):
    result = bayes_factor(data=fake_dt, var_names=["a", "b"], ref_vals=[0, 0])

    for var in ["a", "b"]:
        assert "BF10" in result[var]
        assert "BF01" in result[var]
        assert isinstance(result[var]["BF10"], float)
        assert isinstance(result[var]["BF01"], float)
        assert result[var]["BF10"] > 0
        assert result[var]["BF01"] > 0


def test_bayes_factor_with_custom_prior(fake_dt):
    custom_prior = {"a": np.random.normal(1, 2, 5000), "b": np.random.normal(2, 1, 5000)}
    result = bayes_factor(data=fake_dt, var_names=["a", "b"], ref_vals=[1, 2], prior=custom_prior)

    for var in ["a", "b"]:
        assert "BF10" in result[var]
        assert "BF01" in result[var]
        assert isinstance(result[var]["BF10"], float)
        assert isinstance(result[var]["BF01"], float)
        assert result[var]["BF10"] > 0
        assert result[var]["BF01"] > 0


def test_bayes_factor_invalid_ref_vals(fake_dt):
    with pytest.raises(ValueError, match="Reference value for .* must be numerical"):
        bayes_factor(data=fake_dt, var_names=["a", "b"], ref_vals=["invalid", "invalid"])


def test_bayes_factor_multiple_ref_vals(fake_dt):
    ref_vals_list = [[0.5, 1.5], [1, 2], [1.5, 2.5]]

    for ref_vals in ref_vals_list:
        result = bayes_factor(data=fake_dt, var_names=["a", "b"], ref_vals=ref_vals)
        for var in ["a", "b"]:
            assert "BF10" in result[var]
            assert "BF01" in result[var]
            assert result[var]["BF10"] > 0
            assert result[var]["BF01"] > 0
