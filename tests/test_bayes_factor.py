# pylint: disable=redefined-outer-name, unused-import
# ruff: noqa: F811
import numpy as np
import pytest

from .helpers import importorskip

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


def test_bayes_factor_single_var_string(fake_dt):
    result = bayes_factor(data=fake_dt, var_names="a", ref_vals=0)
    assert "a" in result
    assert "BF10" in result["a"]
    assert "BF01" in result["a"]
    assert result["a"]["BF10"] > 0
    assert result["a"]["BF01"] > 0


def test_bayes_factor_broadcast_ref_val(fake_dt):
    result = bayes_factor(data=fake_dt, var_names=["a", "b"], ref_vals=0.5)
    for var in ["a", "b"]:
        assert "BF10" in result[var]
        assert "BF01" in result[var]
        assert result[var]["BF10"] > 0
        assert result[var]["BF01"] > 0


def test_bayes_factor_reciprocal(fake_dt):
    result = bayes_factor(data=fake_dt, var_names=["a"], ref_vals=[0])
    assert np.isclose(result["a"]["BF10"] * result["a"]["BF01"], 1.0)


def test_bayes_factor_with_ref_vals(fake_dt):
    result, ref_vals = bayes_factor(
        data=fake_dt, var_names=["a", "b"], ref_vals=[0, 0], return_ref_vals=True
    )

    for var in ["a", "b"]:
        assert var in result
        assert var in ref_vals
        assert "prior" in ref_vals[var]
        assert "posterior" in ref_vals[var]
        assert ref_vals[var]["prior"] > 0
        assert ref_vals[var]["posterior"] > 0
        assert isinstance(ref_vals[var]["prior"], float)
        assert isinstance(ref_vals[var]["posterior"], float)


def test_bayes_factor_mismatched_lengths(fake_dt):
    with pytest.raises(ValueError, match="Length of var_names and ref_vals must match"):
        bayes_factor(data=fake_dt, var_names=["a", "b"], ref_vals=[0])


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_bayes_factor_outside_posterior(fake_dt):
    with pytest.warns(UserWarning, match="is outside the posterior range"):
        bayes_factor(data=fake_dt, var_names=["a"], ref_vals=[1000])


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_bayes_factor_outside_prior(fake_dt):
    with pytest.warns(UserWarning, match="is outside the prior range"):
        bayes_factor(data=fake_dt, var_names=["a"], ref_vals=[1000])


def test_bayes_factor_int_ref_val(fake_dt):
    result = bayes_factor(data=fake_dt, var_names=["a"], ref_vals=0)
    assert "a" in result
    assert result["a"]["BF10"] > 0
    assert result["a"]["BF01"] > 0


def test_bayes_factor_float_ref_val(fake_dt):
    result = bayes_factor(data=fake_dt, var_names=["a"], ref_vals=0.0)
    assert "a" in result
    assert result["a"]["BF10"] > 0
    assert result["a"]["BF01"] > 0


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_bayes_factor_narrow_distribution():
    narrow_posterior = np.random.normal(100, 0.1, (4, 100))
    narrow_prior = np.random.normal(100, 0.1, (4, 100))

    data = azb.from_dict(
        {
            "posterior": {"narrow_var": narrow_posterior},
            "prior": {"narrow_var": narrow_prior},
        }
    )

    with pytest.warns(UserWarning):
        result = bayes_factor(data=data, var_names=["narrow_var"], ref_vals=[0])
        assert result["narrow_var"]["BF10"] > 0
        assert result["narrow_var"]["BF01"] > 0
