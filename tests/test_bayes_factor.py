import numpy as np
import pytest
from arviz_base import from_dict

from arviz_stats.bayes_factor import bayes_factor


def test_bayes_factor_default_prior():
    idata = from_dict(
        {
            "posterior": {
                "a": np.random.normal(1, 0.5, (2, 1000)),
                "b": np.random.normal(2, 0.5, (2, 1000)),
            },
            "prior": {
                "a": np.random.normal(0, 1, (2, 1000)),
                "b": np.random.normal(0, 1, (2, 1000)),
            },
        }
    )

    result = bayes_factor(data=idata, var_names=["a", "b"], ref_vals=[0, 0])

    for var in ["a", "b"]:
        assert "BF10" in result[var]
        assert "BF01" in result[var]
        assert isinstance(result[var]["BF10"], float)
        assert isinstance(result[var]["BF01"], float)
        assert result[var]["BF10"] > 0
        assert result[var]["BF01"] > 0


def test_bayes_factor_with_custom_prior():
    idata = from_dict(
        {
            "posterior": {
                "a": np.random.normal(1, 0.5, (2, 1000)),
                "b": np.random.normal(2, 0.5, (2, 1000)),
            },
            "prior": {
                "a": np.random.normal(0, 1, (2, 1000)),
                "b": np.random.normal(0, 1, (2, 1000)),
            },
        }
    )

    custom_prior = {"a": np.random.normal(1, 2, 5000), "b": np.random.normal(2, 1, 5000)}

    result = bayes_factor(data=idata, var_names=["a", "b"], ref_vals=[1, 2], prior=custom_prior)

    for var in ["a", "b"]:
        assert "BF10" in result[var]
        assert "BF01" in result[var]
        assert isinstance(result[var]["BF10"], float)
        assert isinstance(result[var]["BF01"], float)
        assert result[var]["BF10"] > 0
        assert result[var]["BF01"] > 0


def test_bayes_factor_invalid_ref_vals():
    idata = from_dict(
        {
            "posterior": {
                "a": np.random.normal(1, 0.5, (2, 1000)),
                "b": np.random.normal(2, 0.5, (2, 1000)),
            },
            "prior": {
                "a": np.random.normal(0, 1, (2, 1000)),
                "b": np.random.normal(1, 1, (2, 1000)),
            },
        }
    )

    with pytest.raises(ValueError, match="Reference value for .* must be numerical"):
        bayes_factor(data=idata, var_names=["a", "b"], ref_vals=["invalid", "invalid"])


def test_bayes_factor_multiple_ref_vals():
    idata = from_dict(
        {
            "posterior": {
                "a": np.random.normal(1, 0.5, (2, 1000)),
                "b": np.random.normal(2, 0.5, (2, 1000)),
            },
            "prior": {
                "a": np.random.normal(0, 1, (2, 1000)),
                "b": np.random.normal(1, 1, (2, 1000)),
            },
        }
    )

    ref_vals_list = [[0.5, 1.5], [1, 2], [1.5, 2.5]]

    for ref_vals in ref_vals_list:
        result = bayes_factor(data=idata, var_names=["a", "b"], ref_vals=ref_vals)
        for var in ["a", "b"]:
            assert "BF10" in result[var]
            assert "BF01" in result[var]
            assert result[var]["BF10"] > 0
            assert result[var]["BF01"] > 0


def test_bayes_factor_large_sample():
    posterior_data = {
        "a": np.random.normal(1, 0.5, (4, 100_000)),
        "b": np.random.normal(2, 0.5, (4, 100_000)),
    }
    prior_data = {
        "a": np.random.normal(0, 1, (4, 100_000)),
        "b": np.random.normal(1, 1, (4, 100_000)),
    }

    idata = from_dict({"posterior": posterior_data, "prior": prior_data})

    result = bayes_factor(data=idata, var_names=["a", "b"], ref_vals=[0, 0])

    for var in ["a", "b"]:
        assert "BF10" in result[var]
        assert "BF01" in result[var]
        assert isinstance(result[var]["BF10"], float)
        assert isinstance(result[var]["BF01"], float)
