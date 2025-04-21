# pylint: disable=redefined-outer-name
"""Test accessors.

Accessor methods are very short, with the bulk of the computation/processing
handled by private methods. Testing this shared infrastructural methods
is the main goal of this module even if it does so via specific "regular" methods.
"""

import numpy as np
import pytest
from arviz_base import from_dict
from xarray import DataTree

from arviz_stats import accessors  # noqa


@pytest.fixture(scope="module")
def idata():
    return from_dict(
        {
            "posterior": {
                "a": np.random.normal(size=(4, 100)),
                "b": np.random.normal(size=(4, 100, 3)),
            },
            "posterior_predictive": {
                "y": np.random.normal(size=(4, 100, 7)),
            },
        }
    )


def test_accessors_available(idata):
    assert hasattr(idata, "azstats")
    assert hasattr(idata.posterior.ds, "azstats")
    assert hasattr(idata.posterior["a"], "azstats")


def test_datatree_single_group(idata):
    out = idata.azstats.ess(group="posterior")
    assert isinstance(out, DataTree)
    assert not out.children
    assert out.name == "posterior"


def test_datatree_multiple_groups(idata):
    out = idata.azstats.ess(group=["posterior", "posterior_predictive"])
    assert isinstance(out, DataTree)
    assert len(out.children) == 2
    assert "posterior" in out.children
    assert "posterior_predictive" in out.children
