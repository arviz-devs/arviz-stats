"""Test score functions for PSIS-LOO-CV."""

# pylint: disable=redefined-outer-name, unused-argument, protected-access
import pytest

from ..helpers import importorskip

np = importorskip("numpy")
azb = importorskip("arviz_base")
xr = importorskip("xarray")

from numpy.testing import assert_almost_equal

from arviz_stats import loo_score
from arviz_stats.base import array_stats
from arviz_stats.base.stats_utils import round_num
from arviz_stats.loo.loo_helper import _get_r_eff, _prepare_loo_inputs


def _brute_force_score(values, log_weights, y_obs, kind):
    weights = np.exp(log_weights - log_weights.max())
    weights /= weights.sum()
    e_abs = np.sum(weights * np.abs(values - y_obs))
    delta = np.sum(weights[:, None] * weights[None, :] * np.abs(values[:, None] - values[None, :]))
    if kind == "crps":
        return -(e_abs - 0.5 * delta)
    return -e_abs / delta - 0.5 * np.log(delta)


def test_loo_score_invalid_kind(centered_eight):
    with pytest.raises(ValueError, match="kind must be either 'crps' or 'scrps'"):
        loo_score(centered_eight, kind="invalid")


def test_loo_score_invalid_var_name(centered_eight):
    with pytest.raises(TypeError, match="No log likelihood data named nonexistent found"):
        loo_score(centered_eight, var_name="nonexistent")


@pytest.mark.parametrize(
    "scenario, pattern",
    [
        ("missing_sample_dim", "y_pred must include sample dimension 'draw'"),
        ("obs_dim_size_mismatch", "Size mismatch in observation dim 'school'"),
        ("loglik_sample_dim_mismatch", "Size mismatch in sample dimension 'draw'"),
    ],
)
def test_loo_score_validation_errors(centered_eight, scenario, pattern):
    broken = centered_eight.copy()

    if scenario == "missing_sample_dim":
        broken.posterior_predictive["obs"] = broken.posterior_predictive["obs"].mean("draw")
    elif scenario == "obs_dim_size_mismatch":
        broken.observed_data["obs"] = broken.observed_data["obs"].isel(school=slice(0, 7))
    elif scenario == "loglik_sample_dim_mismatch":
        broken.log_likelihood["obs"] = broken.log_likelihood["obs"].isel(draw=slice(0, -1))

    with pytest.raises(ValueError, match=pattern):
        loo_score(broken)


@pytest.mark.parametrize("pointwise", [False, True])
@pytest.mark.parametrize("kind", ["crps", "scrps"])
def test_loo_score_pointwise(centered_eight, kind, pointwise):
    result = loo_score(centered_eight, kind=kind, pointwise=pointwise)

    assert type(result).__name__ == kind.upper()
    assert np.isfinite(result.mean)
    assert np.isfinite(result.se)
    assert result.se >= 0
    if kind == "crps":
        assert result.mean <= 0

    if pointwise:
        assert result.pointwise.shape == (8,)
        assert np.all(np.isfinite(result.pointwise.values))
        assert result.pareto_k.shape == (8,)
        assert np.all(np.isfinite(result.pareto_k.values))
        assert_almost_equal(result.mean, result.pointwise.values.mean(), decimal=14)
        assert_almost_equal(
            result.se,
            result.pointwise.values.std(ddof=0) / result.pointwise.size**0.5,
            decimal=14,
        )
        if kind == "crps":
            assert np.all(result.pointwise.values <= 0)
    else:
        assert result._fields == ("mean", "se")


def test_loo_score_explicit_var_name(centered_eight):
    result_explicit = loo_score(centered_eight, var_name="obs")
    result_auto = loo_score(centered_eight)

    assert_almost_equal(result_explicit.mean, result_auto.mean, decimal=10)
    assert_almost_equal(result_explicit.se, result_auto.se, decimal=10)


@pytest.mark.parametrize("round_to", [2, "2g"])
def test_loo_score_round_to(centered_eight, round_to):
    base = loo_score(centered_eight, kind="crps")
    result = loo_score(centered_eight, kind="crps", round_to=round_to)

    assert result.mean == round_num(base.mean, round_to)
    assert result.se == round_num(base.se, round_to)


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

    result = loo_score(multi_dim_data, kind="crps", pointwise=True)

    assert np.isfinite(result.mean)
    assert np.isfinite(result.se)
    assert result.pointwise.shape == (3, 4)


@pytest.fixture
def psis_weights(centered_eight):
    loo_inputs = _prepare_loo_inputs(centered_eight, None)
    return loo_inputs.log_likelihood.azstats.psislw(
        dim=["chain", "draw"],
        r_eff=_get_r_eff(centered_eight, loo_inputs.n_samples),
    )


def test_loo_score_precomputed_weights(centered_eight, psis_weights):
    result_auto = loo_score(centered_eight, kind="crps", pointwise=True)
    log_weights, pareto_k = psis_weights

    result_precomputed = loo_score(
        centered_eight,
        kind="crps",
        pointwise=True,
        log_weights=log_weights,
        pareto_k=pareto_k,
    )

    assert_almost_equal(result_precomputed.mean, result_auto.mean, decimal=10)
    assert_almost_equal(result_precomputed.se, result_auto.se, decimal=10)
    assert_almost_equal(
        result_precomputed.pointwise.values, result_auto.pointwise.values, decimal=10
    )
    assert_almost_equal(result_precomputed.pareto_k.values, result_auto.pareto_k.values, decimal=10)


@pytest.mark.parametrize("kind", ["crps", "scrps"])
@pytest.mark.parametrize("center", [-100.0, 0.0])
def test_loo_score_kernel_matches_brute_force(kind, center):
    rng = np.random.default_rng(42)
    values = rng.normal(center, 2.0, size=500)
    log_weights = rng.normal(0, 1.5, size=500)
    y_obs = center + 0.5

    result = array_stats._loo_score(values, y_obs, log_weights, kind)
    expected = _brute_force_score(values, log_weights, y_obs, kind)

    assert_almost_equal(result, expected, decimal=12)


@pytest.mark.parametrize("kind", ["crps", "scrps"])
def test_loo_score_kernel_extreme_weights(kind):
    rng = np.random.default_rng(42)
    values = rng.normal(0.0, 2.0, size=300)
    log_weights = rng.normal(0.0, 1.0, size=300) + 800.0

    result = array_stats._loo_score(values, 0.5, log_weights, kind)
    expected = _brute_force_score(values, log_weights, 0.5, kind)

    assert np.isfinite(result)
    assert_almost_equal(result, expected, decimal=12)


@pytest.mark.parametrize("kind", ["crps", "scrps"])
def test_loo_score_kernel_discrete_ties(kind):
    rng = np.random.default_rng(42)
    values = rng.poisson(3, size=400).astype(float)
    log_weights = rng.normal(0, 1.0, size=400)

    result = array_stats._loo_score(values, 2.0, log_weights, kind)
    expected = _brute_force_score(values, log_weights, 2.0, kind)

    assert_almost_equal(result, expected, decimal=12)


def test_loo_score_kernel_small_samples():
    result = array_stats._loo_score(np.array([2.0]), 3.0, np.array([0.0]), "crps")
    assert_almost_equal(result, -1.0, decimal=14)

    values = np.array([1.0, 3.0])
    log_weights = np.zeros(2)
    result = array_stats._loo_score(values, 2.0, log_weights, "crps")
    expected = _brute_force_score(values, log_weights, 2.0, "crps")
    assert_almost_equal(result, expected, decimal=14)

    result = array_stats._loo_score(values, 2.0, log_weights, "scrps")
    assert_almost_equal(result, -1.0, decimal=14)

    values = np.array([1.0, 5.0])
    result = array_stats._loo_score(values, 2.0, log_weights, "crps")
    assert_almost_equal(result, -1.0, decimal=14)
    result = array_stats._loo_score(values, 2.0, log_weights, "scrps")
    assert_almost_equal(result, -1.0 - 0.5 * np.log(2.0), decimal=14)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("kind", ["crps", "scrps"])
def test_loo_score_shift_invariance(kind):
    rng = np.random.default_rng(42)
    mu = rng.normal(size=(2, 50))
    y_pred = rng.normal(size=(2, 50, 5))
    log_lik = rng.normal(size=(2, 50, 5))
    y = rng.normal(size=5)
    shift = -50.0

    dt = azb.from_dict(
        {
            "posterior": {"mu": mu},
            "posterior_predictive": {"y": y_pred},
            "log_likelihood": {"y": log_lik},
            "observed_data": {"y": y},
        }
    )
    dt_shifted = azb.from_dict(
        {
            "posterior": {"mu": mu},
            "posterior_predictive": {"y": y_pred + shift},
            "log_likelihood": {"y": log_lik},
            "observed_data": {"y": y + shift},
        }
    )

    result = loo_score(dt, kind=kind, pointwise=True)
    result_shifted = loo_score(dt_shifted, kind=kind, pointwise=True)

    assert np.all(np.isfinite(result.pointwise.values))
    assert_almost_equal(result_shifted.pointwise.values, result.pointwise.values, decimal=10)


def test_loo_score_lone_precomputed_arg_raises(centered_eight):
    log_weights = xr.zeros_like(centered_eight.log_likelihood["obs"])
    pareto_k = xr.zeros_like(centered_eight.observed_data["obs"])

    with pytest.raises(ValueError, match="must be provided together"):
        loo_score(centered_eight, log_weights=log_weights)
    with pytest.raises(ValueError, match="must be provided together"):
        loo_score(centered_eight, pareto_k=pareto_k)


@pytest.mark.parametrize("kind", ["crps", "scrps"])
def test_loo_score_precomputed_weights_used(centered_eight, kind):
    result_auto = loo_score(centered_eight, kind=kind, pointwise=True)

    uniform_log_weights = xr.zeros_like(centered_eight.log_likelihood["obs"])
    flat_pareto_k = xr.zeros_like(centered_eight.observed_data["obs"])
    result_uniform = loo_score(
        centered_eight,
        kind=kind,
        pointwise=True,
        log_weights=uniform_log_weights,
        pareto_k=flat_pareto_k,
    )

    assert not np.allclose(result_uniform.pointwise.values, result_auto.pointwise.values)

    y_pred = centered_eight.posterior_predictive["obs"].stack(sample=("chain", "draw"))
    y_obs = centered_eight.observed_data["obs"]
    expected = [
        _brute_force_score(
            y_pred.sel(school=school).values,
            np.zeros(y_pred.sizes["sample"]),
            y_obs.sel(school=school).item(),
            kind,
        )
        for school in y_obs["school"].values
    ]
    np.testing.assert_allclose(result_uniform.pointwise.values, expected, rtol=1e-10)


def test_loo_score_precomputed_no_posterior(centered_eight, psis_weights):
    log_weights, pareto_k = psis_weights
    dt_no_posterior = centered_eight.drop_nodes("posterior")

    result = loo_score(dt_no_posterior, kind="crps", log_weights=log_weights, pareto_k=pareto_k)
    result_full = loo_score(centered_eight, kind="crps", log_weights=log_weights, pareto_k=pareto_k)

    assert_almost_equal(result.mean, result_full.mean, decimal=10)
    assert_almost_equal(result.se, result_full.se, decimal=10)
