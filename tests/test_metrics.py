"""Test metrics."""

# pylint: disable=redefined-outer-name
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from .helpers import importorskip

azb = importorskip("arviz_base")

from arviz_stats.base import array_stats
from arviz_stats.metrics import bayesian_r2, kl_divergence, metrics, residual_r2, wasserstein


def test_residual_r2_summary(datatree_regression):
    result = residual_r2(
        datatree_regression, pred_mean="mu", obs_name="y", summary=True, ci_kind="hdi"
    )
    assert isinstance(result, tuple)
    assert hasattr(result, "_fields")
    assert "mean" in result._fields
    assert "hdi_lb" in result._fields
    assert "hdi_ub" in result._fields


def test_residual_r2_array(datatree_regression):
    mu_pred = azb.extract(datatree_regression, group="posterior")["mu"].values.T
    result = residual_r2(datatree_regression, pred_mean="mu", obs_name="y", summary=False)
    assert isinstance(result, np.ndarray)
    assert result.shape == (mu_pred.shape[0],)


def test_residual_r2_circular_summary(datatree_regression):
    result = residual_r2(
        datatree_regression,
        pred_mean="mu",
        obs_name="y",
        summary=True,
        circular=True,
        ci_kind="hdi",
    )
    assert isinstance(result, tuple)
    assert hasattr(result, "_fields")
    assert "mean" in result._fields
    assert "hdi_lb" in result._fields
    assert "hdi_ub" in result._fields


def test_residual_r2_circular_array(datatree_regression):
    mu_pred = azb.extract(datatree_regression, group="posterior")["mu"].values.T
    result = residual_r2(
        datatree_regression, pred_mean="mu", obs_name="y", summary=False, circular=True
    )
    assert isinstance(result, np.ndarray)
    assert result.shape == (mu_pred.shape[0],)


@pytest.mark.parametrize("point_estimate", ["mean", "median"])
def test_residual_r2_point_estimate(datatree_regression, point_estimate):
    result = residual_r2(
        datatree_regression,
        pred_mean="mu",
        obs_name="y",
        summary=True,
        point_estimate=point_estimate,
    )
    assert point_estimate in result._fields


@pytest.mark.parametrize("ci_kind", ["hdi", "eti"])
def test_residual_r2_ci_kind(datatree_regression, ci_kind):
    result = residual_r2(
        datatree_regression, pred_mean="mu", obs_name="y", summary=True, ci_kind=ci_kind
    )
    assert f"{ci_kind}_lb" in result._fields
    assert f"{ci_kind}_ub" in result._fields


@pytest.mark.parametrize("ci_prob", [0.9, 0.95])
def test_residual_r2_ci_prob(datatree_regression, ci_prob):
    result = residual_r2(
        datatree_regression, pred_mean="mu", obs_name="y", summary=True, ci_prob=ci_prob
    )
    assert hasattr(result, "_fields")


def test_residual_r2_no_rounding(datatree_regression):
    result = residual_r2(
        datatree_regression, pred_mean="mu", obs_name="y", summary=True, round_to="None"
    )
    assert isinstance(result.mean, float)


def test_residual_r2_invalid_shapes():
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([[2.5, 0.0, 2]])
    with pytest.raises(ValueError):
        array_stats.residual_r2(y_true, y_pred)


def test_bayesian_r2_summary(datatree_regression):
    result = bayesian_r2(datatree_regression, pred_mean="mu", scale="sigma")
    assert isinstance(result, tuple)
    assert hasattr(result, "_fields")
    assert "mean" in result._fields
    assert "eti_lb" in result._fields
    assert "eti_ub" in result._fields


def test_bayesian_r2_array(datatree_regression):
    mu_pred = azb.extract(datatree_regression, group="posterior")["mu"].values.T
    result = bayesian_r2(datatree_regression, pred_mean="mu", scale="sigma", summary=False)
    assert isinstance(result, np.ndarray)
    assert result.shape == (mu_pred.shape[0],)


def test_bayesian_r2_circular_summary(datatree_regression):
    result = bayesian_r2(
        datatree_regression,
        pred_mean="mu",
        scale="sigma",
        scale_kind="var",
        summary=True,
        circular=True,
        ci_kind="eti",
    )
    assert isinstance(result, tuple)
    assert hasattr(result, "_fields")
    assert "mean" in result._fields
    assert "eti_lb" in result._fields
    assert "eti_ub" in result._fields


def test_bayesian_r2_circular_array(datatree_regression):
    mu_pred = azb.extract(datatree_regression, group="posterior")["mu"].values.T
    result = bayesian_r2(
        datatree_regression,
        pred_mean="mu",
        scale="sigma",
        scale_kind="var",
        summary=False,
        circular=True,
    )
    assert isinstance(result, np.ndarray)
    assert result.shape == (mu_pred.shape[0],)


@pytest.mark.parametrize("point_estimate", ["mean", "median"])
def test_bayesian_r2_point_estimate(datatree_regression, point_estimate):
    result = bayesian_r2(
        datatree_regression,
        pred_mean="mu",
        scale="sigma",
        summary=True,
        point_estimate=point_estimate,
    )
    assert point_estimate in result._fields


def test_bayesian_r2_scale_var(datatree_regression):
    result = bayesian_r2(datatree_regression, pred_mean="mu", scale="sigma")
    datatree_regression.posterior["var"] = datatree_regression.posterior["sigma"] ** 2
    result_var = bayesian_r2(datatree_regression, pred_mean="mu", scale="var", scale_kind="var")
    assert_almost_equal(result.mean, result_var.mean)
    assert_almost_equal(result.eti_lb, result_var.eti_lb)
    assert_almost_equal(result.eti_ub, result_var.eti_ub)


@pytest.mark.parametrize("ci_kind", ["hdi", "eti"])
def test_bayesian_r2_ci_kind(datatree_regression, ci_kind):
    result = bayesian_r2(
        datatree_regression, pred_mean="mu", scale="sigma", summary=True, ci_kind=ci_kind
    )
    assert f"{ci_kind}_lb" in result._fields
    assert f"{ci_kind}_ub" in result._fields


@pytest.mark.parametrize("ci_prob", [0.9, 0.95])
def test_bayesian_r2_ci_prob(datatree_regression, ci_prob):
    result = bayesian_r2(
        datatree_regression, pred_mean="mu", scale="sigma", summary=True, ci_prob=ci_prob
    )
    assert hasattr(result, "_fields")


@pytest.mark.parametrize(
    "kind, round_to, expected_mean, expected_se",
    [
        ("mae", 2, 2.64, 0.38),
        ("mse", "2g", 11.0, 2.6),
        ("rmse", "None", 3.3379, 0.3967),
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


def test_metrics_no_rounding(datatree):
    result = metrics(datatree, kind="rmse", round_to="None")
    assert isinstance(result.mean, float)
    assert isinstance(result.se, float)


@pytest.mark.parametrize("joint", [True, False])
def test_wasserstein(fake_dt, joint):
    result = wasserstein(fake_dt, fake_dt, num_samples=100, joint=joint)
    assert_array_almost_equal(result, 0.0, decimal=5)


def test_wasserstein_var_names(fake_dt):
    result = wasserstein(fake_dt, fake_dt, var_names="a", num_samples=100)
    assert_array_almost_equal(result, 0.0, decimal=5)


def test_wasserstein_no_rounding(fake_dt):
    result = wasserstein(fake_dt, fake_dt, num_samples=100, round_to="None")
    assert isinstance(result, float)


def test_wasserstein_custom_seed(fake_dt):
    result1 = wasserstein(fake_dt, fake_dt, num_samples=100, random_seed=42)
    result2 = wasserstein(fake_dt, fake_dt, num_samples=100, random_seed=42)
    assert_almost_equal(result1, result2)


def test_wasserstein_not_shared_vars(fake_dt):
    with pytest.raises(ValueError, match="No shared variable names found"):
        wasserstein(fake_dt.posterior["a"], fake_dt.posterior["b"], num_samples=100)


def test_kl_divergence(fake_dt):
    result = kl_divergence(fake_dt, fake_dt, num_samples=100)
    assert_array_almost_equal(result, 0.0, decimal=5)


def test_kl_divergence_var_names(fake_dt):
    result = kl_divergence(fake_dt, fake_dt, var_names="a", num_samples=100)
    assert_array_almost_equal(result, 0.0, decimal=5)


def test_kl_divergence_no_rounding(fake_dt):
    result = kl_divergence(fake_dt, fake_dt, num_samples=100, round_to="None")
    assert isinstance(result, float)


def test_kl_divergence_custom_seed(fake_dt):
    result1 = kl_divergence(fake_dt, fake_dt, num_samples=100, random_seed=42)
    result2 = kl_divergence(fake_dt, fake_dt, num_samples=100, random_seed=42)
    assert_almost_equal(result1, result2)


def test_kl_divergence_not_shared_vars(fake_dt):
    with pytest.raises(ValueError, match="No shared variable names found"):
        kl_divergence(fake_dt.posterior["a"], fake_dt.posterior["b"], num_samples=100)
