"""Test score functions for PSIS-LOO-CV."""

# pylint: disable=redefined-outer-name, unused-argument
import pytest

from ..helpers import importorskip

np = importorskip("numpy")
azb = importorskip("arviz_base")

from numpy.testing import assert_almost_equal

from arviz_stats import loo_score
from arviz_stats.loo.helper_loo import _get_r_eff, _prepare_loo_inputs


def test_loo_score_invalid_kind(centered_eight):
    with pytest.raises(ValueError, match="kind must be either 'crps' or 'scrps'"):
        loo_score(centered_eight, kind="invalid")


def test_loo_score_invalid_var_name(centered_eight):
    with pytest.raises(TypeError, match="No log likelihood data named nonexistent found"):
        loo_score(centered_eight, var_name="nonexistent")


@pytest.mark.parametrize("kind", ["crps", "scrps"])
def test_loo_score_basic(centered_eight, kind):
    result = loo_score(centered_eight, kind=kind)

    assert hasattr(result, "mean")
    assert hasattr(result, "se")
    assert np.isfinite(result.mean)
    assert np.isfinite(result.se)
    assert result.se >= 0

    if kind == "crps":
        assert result.mean <= 0


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
    assert hasattr(result, "pareto_k")
    assert np.isfinite(result.mean)
    assert np.isfinite(result.se)
    assert result.pointwise.shape == (8,)
    assert np.all(np.isfinite(result.pointwise.values))
    assert result.pareto_k.shape == (8,)
    assert np.all(np.isfinite(result.pareto_k.values))

    if kind == "crps":
        assert np.all(result.pointwise.values <= 0)


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


@pytest.mark.parametrize("kind", ["crps", "scrps"])
def test_loo_score_precomputed_weights(centered_eight, kind):
    result_auto = loo_score(centered_eight, kind=kind, pointwise=True)

    loo_inputs = _prepare_loo_inputs(centered_eight, None)
    log_likelihood = loo_inputs.log_likelihood

    log_weights_computed, pareto_k_computed = log_likelihood.azstats.psislw(
        dim=["chain", "draw"],
        r_eff=_get_r_eff(centered_eight, loo_inputs.n_samples),
    )

    result_precomputed = loo_score(
        centered_eight,
        kind=kind,
        pointwise=True,
        log_weights=log_weights_computed,
        pareto_k=pareto_k_computed,
    )

    assert_almost_equal(result_precomputed.mean, result_auto.mean, decimal=10)
    assert_almost_equal(result_precomputed.se, result_auto.se, decimal=10)
    assert_almost_equal(
        result_precomputed.pointwise.values, result_auto.pointwise.values, decimal=10
    )
    assert_almost_equal(result_precomputed.pareto_k.values, result_auto.pareto_k.values, decimal=10)
