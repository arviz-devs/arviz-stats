# pylint: disable=redefined-outer-name
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from .helpers import importorskip

azb = importorskip("arviz_base")

from arviz_stats.base import array_stats
from arviz_stats.metrics import kl_divergence, metrics, r2_score, wasserstein


@pytest.fixture
def sample_data():
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([[[2.5, 0.0, 2, 8], [3.0, -0.5, 2, 7], [2.8, -0.3, 2.1, 7.2]]])
    return azb.from_dict({"observed_data": {"y": y_true}, "posterior_predictive": {"y": y_pred}})


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


@pytest.fixture(name="centered_eight", scope="session")
def fixture_centered_eight():
    return azb.load_arviz_data("centered_eight")


@pytest.fixture(name="anes", scope="session")
def fixture_anes():
    return azb.load_arviz_data("anes")


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


@pytest.mark.parametrize(
    "kind, round_to, expected_mean, expected_se",
    [
        ("mae", 2, 7.42, 2.14),
        ("mse", "2g", 92.0, 51.0),
        ("rmse", None, 9.5757, 2.6479),
    ],
)
def test_metrics(centered_eight, kind, round_to, expected_mean, expected_se):
    result = metrics(centered_eight, kind=kind, round_to=round_to)
    assert_almost_equal(result.mean, expected_mean, decimal=4)
    assert_almost_equal(result.se, expected_se, decimal=4)


@pytest.mark.parametrize(
    "kind, round_to, expected_mean, expected_se",
    [
        ("acc", 2, 0.82, 0.02),
        ("acc_balanced", "2g", 0.82, 0.0039),
    ],
)
def test_metrics_acc(anes, kind, round_to, expected_mean, expected_se):
    result = metrics(anes, kind=kind, round_to=round_to)
    assert_almost_equal(result.mean, expected_mean, decimal=4)
    assert_almost_equal(result.se, expected_se, decimal=4)


def test_metrics_invalid_kind(centered_eight):
    with pytest.raises(ValueError, match="kind must be one of"):
        metrics(centered_eight, kind="invalid_kind")


@pytest.mark.parametrize("joint", [True, False])
def test_wasserstein(fake_post, joint):
    result = wasserstein(fake_post, fake_post, num_samples=100, joint=joint)
    assert_array_almost_equal(result, 0.0, decimal=5)


def test_wasserstein_not_shared_vars(fake_post):
    with pytest.raises(ValueError, match="No shared variable names found"):
        wasserstein(fake_post.posterior["a"], fake_post.posterior["b"], num_samples=100)


def test_kl_divergence(fake_post):
    result = kl_divergence(fake_post, fake_post, num_samples=100)
    assert_array_almost_equal(result, 0.0, decimal=5)


def test_kl_divergence_not_shared_vars(fake_post):
    with pytest.raises(ValueError, match="No shared variable names found"):
        kl_divergence(fake_post.posterior["a"], fake_post.posterior["b"], num_samples=100)
