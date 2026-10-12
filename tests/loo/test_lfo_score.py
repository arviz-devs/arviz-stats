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
    np.testing.assert_array_equal(result.pointwise.coords["time"].values, origins + 2000)
    np.testing.assert_allclose(result.pointwise.values, expected)
    np.testing.assert_allclose(result.mean, expected.mean())
    np.testing.assert_allclose(result.se, expected.std() / np.sqrt(len(expected)))
    assert result.pareto_k is None
    np.testing.assert_array_equal(result.refits, origins)
    assert result.n_refits == len(origins)


def test_lfo_score_exact_refits_before_scoring(varying_lfo_wrapper, lfo_varying_data):
    result = lfo_score(
        lfo_varying_data,
        varying_lfo_wrapper,
        min_observations=20,
        forecast_horizon=2,
        kind="scrps",
        method="exact",
        pointwise=True,
    )

    train, block = varying_lfo_wrapper.sel_observations(np.arange(23, 25))
    idata = varying_lfo_wrapper.get_inference_data(varying_lfo_wrapper.sample(train))
    draws = varying_lfo_wrapper.posterior_predictive__i(block, idata)
    y_obs = lfo_varying_data.observed_data["obs"].isel(time=slice(23, 25))
    scores, _ = draws.azstats.loo_score(y_obs=y_obs, log_weights=xr.zeros_like(draws), kind="scrps")

    assert type(result).__name__ == "SCRPS"
    np.testing.assert_allclose(result.pointwise.values[-1], scores.sum().values)


@pytest.mark.parametrize("kind", ["crps", "scrps"])
def test_lfo_score_approx_weights_draws(varying_lfo_wrapper, lfo_varying_data, kind):
    wrapper = varying_lfo_wrapper
    result = lfo_score(
        lfo_varying_data,
        wrapper,
        min_observations=20,
        forecast_horizon=2,
        kind=kind,
        method="approx",
        pointwise=True,
    )
    assert wrapper.fit_count == 1
    assert result.n_refits == 0

    obs = lfo_varying_data.observed_data["obs"]
    fit = wrapper.get_inference_data(wrapper.sample(obs.isel(time=slice(20))))
    mu = fit.posterior["mu"].values.ravel()
    expected = []
    for cutoff in range(20, obs.size - 1):
        _, block = wrapper.sel_observations(np.arange(cutoff, cutoff + 2))
        draws = wrapper.posterior_predictive__i(block, fit).values[0]
        ratios = np.exp(-0.5 * ((obs.values[20:cutoff, None] - mu) ** 2).sum(axis=0))
        weights = ratios / ratios.sum()
        error = (np.abs(draws - block.values[:, None]) * weights).sum(axis=1)
        pairwise = np.abs(draws[:, :, None] - draws[:, None, :])
        gini = (pairwise * weights[:, None] * weights[None, :]).sum(axis=(1, 2))
        score = -(error - gini / 2) if kind == "crps" else -error / gini - 0.5 * np.log(gini)
        expected.append(score.sum())

    log_ratios = xr.DataArray(-0.5 * (obs.values[20] - mu)[None] ** 2, dims=["chain", "draw"])
    _, khat = (-log_ratios).azstats.psislw(dim=["chain", "draw"], r_eff=1.0)

    np.testing.assert_allclose(result.pointwise.values, expected, atol=0.002, rtol=0)
    np.testing.assert_allclose(result.pareto_k.values[1], khat.values)


def test_lfo_score_rejects_short_observed_data(varying_lfo_wrapper, lfo_varying_data):
    data = lfo_varying_data.copy()
    data["observed_data"] = lfo_varying_data.observed_data.to_dataset().isel(time=slice(0, -1))

    with pytest.raises(ValueError, match="to match the log likelihood"):
        lfo_score(data, varying_lfo_wrapper, min_observations=20, forecast_horizon=2)

    assert varying_lfo_wrapper.fit_count == 0


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
