"""Tests for LFO-CV sample-based scores."""

# pylint: disable=redefined-outer-name
import numpy as np
import pytest

from ..helpers import importorskip

azb = importorskip("arviz_base")
xr = importorskip("xarray")

from arviz_stats import lfo_score


@pytest.mark.parametrize("horizon", [1, 3])
def test_lfo_score_constant_draws(constant_lfo_wrapper, lfo_constant_data, horizon):
    min_obs = 5
    n_time = lfo_constant_data.log_likelihood["obs"].sizes["time"]
    origins = np.arange(min_obs, n_time - horizon + 1)
    obs = lfo_constant_data.observed_data["obs"].values
    expected = np.array(
        [-np.abs(constant_lfo_wrapper.pred - obs[i : i + horizon]).sum() for i in origins]
    )

    result = lfo_score(
        lfo_constant_data,
        constant_lfo_wrapper,
        min_observations=min_obs,
        forecast_horizon=horizon,
        method="exact",
        pointwise=True,
    )

    assert type(result).__name__ == "CRPS"
    assert result.pointwise.dims == ("time",)
    np.testing.assert_array_equal(result.pointwise.coords["time"].values, origins)
    np.testing.assert_allclose(result.pointwise.values, expected)
    np.testing.assert_allclose(result.mean, expected.mean())
    np.testing.assert_allclose(result.se, expected.std() / np.sqrt(len(expected)))
    assert result.pareto_k is None
    np.testing.assert_array_equal(result.refits, origins)
    assert result.n_refits == len(origins)


def test_lfo_score_exact_uses_uniform_weights(varying_lfo_wrapper, lfo_varying_data):
    min_obs, horizon = 20, 2
    result = lfo_score(
        lfo_varying_data,
        varying_lfo_wrapper,
        min_observations=min_obs,
        forecast_horizon=horizon,
        kind="scrps",
        method="exact",
        pointwise=True,
    )

    cutoff = result.pointwise.coords["time"].values[0]
    _, excluded = varying_lfo_wrapper.sel_observations(np.arange(cutoff, cutoff + horizon))
    idata = varying_lfo_wrapper.get_inference_data(
        varying_lfo_wrapper.sample(varying_lfo_wrapper.sel_observations(np.arange(cutoff, 25))[0])
    )
    draws = varying_lfo_wrapper.posterior_predictive__i(excluded, idata)
    y_obs = lfo_varying_data.observed_data["obs"].isel(time=slice(cutoff, cutoff + horizon))
    scores, _ = draws.azstats.loo_score(y_obs=y_obs, log_weights=xr.zeros_like(draws), kind="scrps")

    assert type(result).__name__ == "SCRPS"
    np.testing.assert_allclose(result.pointwise.values[0], scores.sum().values)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_lfo_score_approx(varying_lfo_wrapper, lfo_varying_data):
    min_obs, horizon = 5, 2
    n_time = lfo_varying_data.log_likelihood["obs"].sizes["time"]
    n_origins = n_time - horizon - min_obs + 1

    result = lfo_score(
        lfo_varying_data,
        varying_lfo_wrapper,
        min_observations=min_obs,
        forecast_horizon=horizon,
        method="approx",
        k_threshold=0.7,
        pointwise=True,
    )

    assert result.pointwise.sizes["time"] == n_origins
    assert np.all(np.isfinite(result.pointwise.values))
    assert result.pareto_k.sizes["time"] == n_origins
    assert np.isnan(result.pareto_k.values[0])
    assert result.n_refits == len(result.refits)
    assert varying_lfo_wrapper.fit_count == result.n_refits + 1


def test_lfo_score_requires_posterior_predictive(custom_dim_lfo_wrapper, lfo_custom_dim_data):
    with pytest.raises(ValueError, match="posterior_predictive__i"):
        lfo_score(
            lfo_custom_dim_data,
            custom_dim_lfo_wrapper,
            min_observations=5,
            forecast_horizon=2,
            time_dim="week",
        )


def test_lfo_score_invalid_kind(constant_lfo_wrapper, lfo_constant_data):
    with pytest.raises(ValueError, match="kind must be"):
        lfo_score(
            lfo_constant_data,
            constant_lfo_wrapper,
            min_observations=5,
            forecast_horizon=2,
            kind="log",
        )
