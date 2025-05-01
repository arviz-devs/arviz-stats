# pylint: disable=invalid-name,
# pylint: disable=redefined-outer-name
import numpy as np
import pandas as pd
import pytest

from arviz_stats.regression import R2


@pytest.fixture
def sample_data():
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([[2.5, 0.0, 2, 8], [3.0, -0.5, 2, 7], [2.8, -0.3, 2.1, 7.2]])
    return y_true, y_pred


def test_R2_summary(sample_data):
    y_true, y_pred = sample_data
    result = R2(y_true, y_pred, summary=True, ci_kind="hdi")
    assert isinstance(result, pd.Series)
    assert "mean" in result.index
    assert "hdi_" in result.index
    assert "hdi^" in result.index


def test_R2_array(sample_data):
    y_true, y_pred = sample_data
    result = R2(y_true, y_pred, summary=False)
    assert isinstance(result, np.ndarray)
    assert result.shape == (y_pred.shape[0],)


def test_R2_invalid_shapes():
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([[2.5, 0.0, 2]])
    with pytest.raises(ValueError):
        R2(y_true, y_pred)
