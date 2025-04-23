# pylint: disable=redefined-outer-name
import numpy as np
import pytest

from arviz_stats.base import array_stats


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(9)
    return rng.normal(size=(4, 200, 3))


def test_ess(data):
    ess = array_stats.ess(data, chain_axis=0, draw_axis=1)
    assert ess.shape == (3,)
    assert np.all(ess > 700)


def test_ess_no_chain(data):
    ess = array_stats.ess(data, chain_axis=None, draw_axis=1)
    assert ess.shape == (4, 3)
    assert np.all(ess > 150)
