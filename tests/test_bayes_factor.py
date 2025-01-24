import numpy as np
import pytest
from arviz_base import from_dict

from src.arviz_stats.bayes_factor import bayes_factor


def test_bayes_factor_comparison():
    idata = from_dict(
        {"posterior": {"a": np.random.normal(1, 0.5, (2, 1000))},
         "prior": {"a": np.random.normal(0, 1, (2, 1000))}}
    )
    bf_dict0 = bayes_factor(idata=idata, var_name="a", ref_val=0)
    custom_prior = np.random.normal(1, 2, 5000)
    bf_dict1 = bayes_factor(idata=idata, var_name="a", prior={"a": custom_prior}, ref_val=1)
    assert "BF10" in bf_dict0
    assert "BF01" in bf_dict0
    assert bf_dict0["BF10"] > bf_dict0["BF01"]
    assert "BF10" in bf_dict1
    assert "BF01" in bf_dict1
    assert bf_dict1["BF10"] < bf_dict1["BF01"]

def test_bayes_factor_invalid_ref_val():
    idata = from_dict(
        {"posterior": {"a": np.random.normal(1, 0.5, (2, 1000))},
         "prior": {"a": np.random.normal(0, 1, (2, 1000))}}
    )
    with pytest.raises(ValueError, match="The reference value.*must be a numerical value.*"):
        bayes_factor(idata=idata, var_name="a", ref_val="invalid")

def test_bayes_factor_custom_prior():
    posterior_data = np.random.normal(1, 0.5, (2, 1000))
    prior_data = np.random.normal(0, 1, (2, 1000))
    custom_prior = np.random.normal(0, 10, (2, 1000))
    idata = from_dict(
        {"posterior": {"a": posterior_data},
         "prior": {"a": prior_data}}
    )
    result = bayes_factor(idata=idata, var_name="a", prior={"a": custom_prior}, ref_val=0)
    assert "BF10" in result
    assert "BF01" in result
    assert result["BF10"] > 0
    assert result["BF01"] > 0

def test_bayes_factor_different_ref_vals():
    idata = from_dict(
        {"posterior": {"a": np.random.normal(1, 0.5, (2, 1000))},
         "prior": {"a": np.random.normal(0, 1, (2, 1000))}}
    )
    ref_vals = [-1, 0, 1]
    for ref_val in ref_vals:
        result = bayes_factor(idata=idata, var_name="a", ref_val=ref_val)
        assert "BF10" in result
        assert "BF01" in result
        assert result["BF10"] > 0
        assert result["BF01"] > 0

def test_bayes_factor_large_data():
    posterior_data = np.random.normal(1, 0.5, (2, 1000))
    prior_data = np.random.normal(0, 1, (2, 1000))
    idata = from_dict(
        {"posterior": {"a": posterior_data},
         "prior": {"a": prior_data}}
    )
    result = bayes_factor(idata=idata, var_name="a", ref_val=0)
    assert "BF10" in result
    assert "BF01" in result
    assert result["BF10"] > 0
    assert result["BF01"] > 0
