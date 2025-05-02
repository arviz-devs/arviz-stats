# pylint: disable=redefined-outer-name
import numpy as np
import pandas as pd
import pytest

from arviz_stats.regression import r2_score


@pytest.fixture
def sample_data():
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([[2.5, 0.0, 2, 8], [3.0, -0.5, 2, 7], [2.8, -0.3, 2.1, 7.2]])
    return y_true, y_pred


def test_r2_score_summary(sample_data):
    y_true, y_pred = sample_data
    result = r2_score(y_true, y_pred, summary=True, ci_kind="hdi")
    assert isinstance(result, pd.Series)
    assert "mean" in result.index
    assert "hdi_" in result.index
    assert "hdi^" in result.index


def test_r2_score_array(sample_data):
    y_true, y_pred = sample_data
    result = r2_score(y_true, y_pred, summary=False)
    assert isinstance(result, np.ndarray)
    assert result.shape == (y_pred.shape[0],)


def test_r2_score_invalid_shapes():
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([[2.5, 0.0, 2]])
    with pytest.raises(ValueError):
        r2_score(y_true, y_pred)
