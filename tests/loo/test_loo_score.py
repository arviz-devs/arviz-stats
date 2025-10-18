# pylint: disable=redefined-outer-name, unused-import, unused-argument
# ruff: noqa: F811, F401
import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from ..helpers import importorskip

azb = importorskip("arviz_base")

from arviz_stats import loo, loo_score


def test_loo_score_invalid_kind(centered_eight):
    with pytest.raises(ValueError, match="kind must be either 'crps' or 'scrps'"):
        loo_score(centered_eight, kind="invalid")


def test_loo_score_invalid_var_name(centered_eight):
    with pytest.raises(TypeError, match="No log likelihood data named nonexistent found"):
        loo_score(centered_eight, var_name="nonexistent")


def test_loo_score_mismatched_log_weights_pareto_k(centered_eight):
    loo_result = loo(centered_eight, pointwise=True)

    with pytest.raises(ValueError, match="Both log_weights and pareto_k must be provided together"):
        loo_score(centered_eight, log_weights=loo_result.log_weights, pareto_k=None)

    with pytest.raises(ValueError, match="Both log_weights and pareto_k must be provided together"):
        loo_score(centered_eight, log_weights=None, pareto_k=loo_result.pareto_k)


@pytest.mark.parametrize("kind", ["crps", "scrps"])
def test_loo_score_basic(centered_eight, kind):
    result = loo_score(centered_eight, kind=kind)

    assert hasattr(result, "mean")
    assert hasattr(result, "se")
    assert np.isfinite(result.mean)
    assert np.isfinite(result.se)


@pytest.mark.parametrize("kind", ["crps", "scrps"])
def test_loo_score_log_weights(centered_eight, kind):
    loo_result = loo(centered_eight, pointwise=True)

    r1 = loo_score(centered_eight, kind=kind)
    r2 = loo_score(
        centered_eight,
        log_weights=loo_result.log_weights,
        pareto_k=loo_result.pareto_k,
        kind=kind,
    )

    assert_almost_equal(r1.mean, r2.mean, decimal=10)
    assert_almost_equal(r1.se, r2.se, decimal=10)


@pytest.mark.parametrize("kind", ["crps", "scrps"])
@pytest.mark.parametrize(
    "scenario, pattern",
    [
        ("missing_sample_dim", "y_pred must include sample dimension 'draw'"),
        ("obs_dim_size_mismatch", "Size mismatch in observation dim 'school'"),
        ("loglik_sample_dim_mismatch", "Size mismatch in sample dimension 'draw'"),
    ],
)
def test_loo_score_validation_errors(centered_eight, kind, scenario, pattern):
    broken = centered_eight.copy()

    if scenario == "missing_sample_dim":
        broken.posterior_predictive["obs"] = broken.posterior_predictive["obs"].mean("draw")
    elif scenario == "obs_dim_size_mismatch":
        broken.observed_data["obs"] = broken.observed_data["obs"].isel(school=slice(0, 7))
    elif scenario == "loglik_sample_dim_mismatch":
        broken.log_likelihood["obs"] = broken.log_likelihood["obs"].isel(draw=slice(0, -1))

    with pytest.raises(ValueError, match=pattern):
        loo_score(broken, kind=kind)


@pytest.mark.parametrize("kind", ["crps", "scrps"])
def test_loo_score_pointwise(centered_eight, kind):
    result = loo_score(centered_eight, kind=kind, pointwise=True)

    assert hasattr(result, "mean")
    assert hasattr(result, "se")
    assert hasattr(result, "pointwise")
    assert np.isfinite(result.mean)
    assert np.isfinite(result.se)
    assert result.pointwise.shape == (8,)
    assert np.all(np.isfinite(result.pointwise.values))


def test_loo_score_namedtuple_names(centered_eight):
    crps_result = loo_score(centered_eight, kind="crps")
    scrps_result = loo_score(centered_eight, kind="scrps")

    assert type(crps_result).__name__ == "CRPS"
    assert type(scrps_result).__name__ == "SCRPS"


def test_loo_score_crps_vs_scrps(centered_eight):
    crps_result = loo_score(centered_eight, kind="crps")
    scrps_result = loo_score(centered_eight, kind="scrps")

    assert crps_result.mean != scrps_result.mean
    assert crps_result.se != scrps_result.se


def test_loo_score_explicit_var_name(centered_eight):
    result_explicit = loo_score(centered_eight, var_name="obs")
    result_auto = loo_score(centered_eight)

    assert_almost_equal(result_explicit.mean, result_auto.mean, decimal=10)
    assert_almost_equal(result_explicit.se, result_auto.se, decimal=10)


@pytest.mark.parametrize("round_to", [2, 3, "2g", "3g", None])
def test_loo_score_round_to(centered_eight, round_to):
    result = loo_score(centered_eight, kind="crps", round_to=round_to)

    assert hasattr(result, "mean")
    assert hasattr(result, "se")

    if round_to is None:
        assert isinstance(result.mean, float)
        assert isinstance(result.se, float)
    else:
        assert isinstance(result.mean, int | float | str)
        assert isinstance(result.se, int | float | str)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_loo_score_multidimensional():
    rng = np.random.default_rng(42)

    multi_dim_data = azb.from_dict(
        {
            "posterior": {"mu": rng.normal(size=(2, 50))},
            "posterior_predictive": {"y": rng.normal(size=(2, 50, 3, 4))},
            "log_likelihood": {"y": rng.normal(size=(2, 50, 3, 4))},
            "observed_data": {"y": rng.normal(size=(3, 4))},
        }
    )

    result = loo_score(multi_dim_data, kind="crps")

    assert hasattr(result, "mean")
    assert hasattr(result, "se")
    assert np.isfinite(result.mean)
    assert np.isfinite(result.se)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_loo_score_pointwise_shape_multidim():
    rng = np.random.default_rng(42)

    multi_var_data = azb.from_dict(
        {
            "posterior": {"mu": rng.normal(size=(2, 50))},
            "posterior_predictive": {"y": rng.normal(size=(2, 50, 5))},
            "log_likelihood": {"y": rng.normal(size=(2, 50, 5))},
            "observed_data": {"y": rng.normal(size=5)},
        }
    )

    result = loo_score(multi_var_data, kind="crps", pointwise=True)

    assert result.pointwise.shape == (5,)
