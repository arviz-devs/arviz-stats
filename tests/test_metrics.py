# pylint: disable=redefined-outer-name
import numpy as np
import pytest

from .helpers import importorskip

azb = importorskip("arviz_base")

from arviz_stats.base import array_stats
from arviz_stats.metrics import r2_score


@pytest.fixture
def sample_data():
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([[[2.5, 0.0, 2, 8], [3.0, -0.5, 2, 7], [2.8, -0.3, 2.1, 7.2]]])
    return azb.from_dict({"observed_data": {"y": y_true}, "posterior_predictive": {"y": y_pred}})


def test_r2_score_summary(sample_data):
    result = r2_score(sample_data, summary=True, ci_kind="hdi")
    assert isinstance(result, tuple)
    assert hasattr(result, "_fields")
    assert "mean" in result._fields
    assert "hdi_lb" in result._fields
    assert "hdi_ub" in result._fields


def test_r2_score_array(sample_data):
    y_pred = azb.extract(sample_data, group="posterior_predictive").values.T
    result = r2_score(sample_data, summary=False)
    assert isinstance(result, np.ndarray)
    assert result.shape == (y_pred.shape[0],)


def test_r2_score_invalid_shapes():
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([[2.5, 0.0, 2]])
    with pytest.raises(ValueError):
        array_stats.r2_score(y_true, y_pred)
