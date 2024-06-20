"""Test for general computational backend agnostic utilities."""
import inspect

import numpy as np
import pytest
from arviz_base import from_dict, rcParams

from arviz_stats.base.dataarray import dataarray_stats
from arviz_stats.utils import ELPDData, get_function, get_log_likelihood


def test_get_function_str_module():
    rcParams["stats.module"] = "base"
    func = get_function("eti")
    assert inspect.ismethod(func)
    assert func.__self__ is dataarray_stats


def test_get_function_obj_module():
    rcParams["stats.module"] = dataarray_stats
    func = get_function("eti")
    assert inspect.ismethod(func)
    assert func.__self__ is dataarray_stats


def test_get_log_likelihood():
    idata = from_dict(
        {
            "log_likelihood": {
                "y1": np.random.normal(size=(4, 100, 6)),
                "y2": np.random.normal(size=(4, 100, 8)),
            }
        }
    )
    lik1 = get_log_likelihood(idata, "y1")
    lik2 = get_log_likelihood(idata, "y2")
    assert lik1.shape == (4, 100, 6)
    assert lik2.shape == (4, 100, 8)


def test_get_log_likelihood_warning():
    idata = from_dict(
        {
            "sample_stats": {
                "log_likelihood": np.random.normal(size=(4, 100, 6)),
            }
        }
    )
    with pytest.warns(DeprecationWarning):
        get_log_likelihood(idata)


def test_get_log_likelihood_no_var_name():
    idata = from_dict(
        {
            "log_likelihood": {
                "y1": np.random.normal(size=(4, 100, 6)),
                "y2": np.random.normal(size=(4, 100, 8)),
            }
        }
    )
    with pytest.raises(TypeError, match="Found several"):
        get_log_likelihood(idata)


def test_get_log_likelihood_no_group():
    idata = from_dict(
        {
            "posterior": {
                "a": np.random.normal(size=(4, 100)),
                "b": np.random.normal(size=(4, 100)),
            }
        }
    )
    with pytest.raises(TypeError, match="log likelihood not found"):
        get_log_likelihood(idata)


def test_elpddata_base():
    elpddata = ELPDData(
        kind="loo",
        elpd=-20,
        se=2,
        p=5.6,
        n_samples=1000,
        n_data_points=370,
        scale="log",
        warning=False,
    )

    printed = str(elpddata)
    assert "warning" not in printed
    assert "1000 posterior samples" in printed
    assert "370 observations" in printed
