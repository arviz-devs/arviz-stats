# pylint: disable=redefined-outer-name, unused-import
# ruff: noqa: F811
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from .helpers import datatree, datatree_binary, importorskip  # noqa: F401

azb = importorskip("arviz_base")

from arviz_stats.base import array_stats
from arviz_stats.metrics import kl_divergence, metrics, r2_score, wasserstein


def test_r2_score_summary(datatree):
    result = r2_score(datatree, summary=True, ci_kind="hdi")
    assert isinstance(result, tuple)
    assert hasattr(result, "_fields")
    assert "mean" in result._fields
    assert "hdi_lb" in result._fields
    assert "hdi_ub" in result._fields


def test_r2_score_array(datatree):
    y_pred = azb.extract(datatree, group="posterior_predictive").values.T
    result = r2_score(datatree, summary=False)
    assert isinstance(result, np.ndarray)
    assert result.shape == (y_pred.shape[0],)


def test_r2_score_invalid_shapes():
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([[2.5, 0.0, 2]])
    with pytest.raises(ValueError):
        array_stats.r2_score(y_true, y_pred)


@pytest.mark.parametrize(
    "kind, round_to, expected_mean, expected_se",
    [
        ("mae", 2, 2.64, 0.38),
        ("mse", "2g", 11.0, 2.6),
        ("rmse", None, 3.3379, 0.3967),
    ],
)
def test_metrics(datatree, kind, round_to, expected_mean, expected_se):
    result = metrics(datatree, kind=kind, round_to=round_to)
    assert_almost_equal(result.mean, expected_mean, decimal=4)
    assert_almost_equal(result.se, expected_se, decimal=4)


@pytest.mark.parametrize(
    "kind, round_to, expected_mean, expected_se",
    [
        ("acc", 2, 0.86, 0.13),
        ("acc_balanced", "2g", 0.83, 0.021),
    ],
)
def test_metrics_acc(datatree_binary, kind, round_to, expected_mean, expected_se):
    result = metrics(datatree_binary, kind=kind, round_to=round_to)
    assert_almost_equal(result.mean, expected_mean, decimal=4)
    assert_almost_equal(result.se, expected_se, decimal=4)


def test_metrics_invalid_kind(datatree):
    with pytest.raises(ValueError, match="kind must be one of"):
        metrics(datatree, kind="invalid_kind")


@pytest.mark.parametrize("joint", [True, False])
def test_wasserstein(fake_dt, joint):
    result = wasserstein(fake_dt, fake_dt, num_samples=100, joint=joint)
    assert_array_almost_equal(result, 0.0, decimal=5)


def test_wasserstein_not_shared_vars(fake_dt):
    with pytest.raises(ValueError, match="No shared variable names found"):
        wasserstein(fake_dt.posterior["a"], fake_dt.posterior["b"], num_samples=100)


def test_kl_divergence(fake_dt):
    result = kl_divergence(fake_dt, fake_dt, num_samples=100)
    assert_array_almost_equal(result, 0.0, decimal=5)


def test_kl_divergence_not_shared_vars(fake_dt):
    with pytest.raises(ValueError, match="No shared variable names found"):
        kl_divergence(fake_dt.posterior["a"], fake_dt.posterior["b"], num_samples=100)
