# pylint: disable=redefined-outer-name
import numpy as np
import pytest

from .helpers import importorskip

azb = importorskip("arviz_base")

from arviz_stats.summary import ci_in_rope


@pytest.fixture
def fake_post():
    return azb.from_dict(
        {
            "posterior": {
                "a": np.random.normal(size=(4, 100)),
                "b": np.random.normal(size=(4, 100)),
            },
        }
    )


def test_rope_single(fake_post):
    result = ci_in_rope(fake_post, var_names=["a", "b"], rope=(-1, 1))
    assert all(result > 60)
    result = ci_in_rope(fake_post, var_names=["a", "b"], rope=(-1, 1), ci_prob=0.5, ci_kind="hdi")
    assert all(result == 100)
    assert "a" in result.data_vars
    assert "b" in result.data_vars


def test_rope_multiple(fake_post):
    result = ci_in_rope(fake_post, rope={"a": (-100, -99), "b": (-2, 2)})
    assert result["a"] == 0
    assert result["b"] > 90
    assert "a" in result.data_vars
    assert "b" in result.data_vars
