"""Tests for LFO-CV (leave-future-out cross-validation)."""

# pylint: disable=redefined-outer-name
import numpy as np
import pytest

from ..helpers import importorskip

azb = importorskip("arviz_base")
xr = importorskip("xarray")

from arviz_stats import lfo_cv


@pytest.mark.parametrize("method", ["exact", "approx"])
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_lfo_cv_constant_log_likelihood(constant_lfo_wrapper, lfo_constant_data, method):
    min_obs, horizon = 5, 3
    n_time = lfo_constant_data.log_likelihood["obs"].sizes["time"]
    const = constant_lfo_wrapper.const
    n_origins = n_time - horizon - min_obs + 1

    result = lfo_cv(
        lfo_constant_data,
        constant_lfo_wrapper,
        min_observations=min_obs,
        forecast_horizon=horizon,
        method=method,
        pointwise=True,
    )

    assert result.kind == "lfo_cv"
    assert result.n_data_points == n_origins
    assert result.elpd_i.sizes["time"] == n_origins
    np.testing.assert_allclose(result.elpd_i.values, horizon * const)
    assert np.isclose(result.elpd, n_origins * horizon * const)
    assert np.isclose(result.se, 0.0)
    assert np.isclose(result.p, 0.0)


def test_lfo_cv_exact_refits_each_origin(constant_lfo_wrapper, lfo_constant_data):
    min_obs, horizon = 5, 3
    n_time = lfo_constant_data.log_likelihood["obs"].sizes["time"]
    expected_cutoffs = list(range(min_obs, n_time - horizon + 1))

    lfo_cv(
        lfo_constant_data,
        constant_lfo_wrapper,
        min_observations=min_obs,
        forecast_horizon=horizon,
        method="exact",
    )

    assert constant_lfo_wrapper.fit_count == len(expected_cutoffs)
    assert constant_lfo_wrapper.cutoffs == expected_cutoffs


@pytest.mark.parametrize("horizon", [1, 2, 4])
def test_lfo_cv_forecast_horizon(constant_lfo_wrapper, lfo_constant_data, horizon):
    min_obs = 5
    n_time = lfo_constant_data.log_likelihood["obs"].sizes["time"]

    result = lfo_cv(
        lfo_constant_data,
        constant_lfo_wrapper,
        min_observations=min_obs,
        forecast_horizon=horizon,
        method="exact",
        pointwise=True,
    )

    assert result.n_data_points == n_time - horizon - min_obs + 1
    np.testing.assert_allclose(result.elpd_i.values, horizon * constant_lfo_wrapper.const)


@pytest.mark.parametrize("min_obs", [4, 6, 8])
def test_lfo_cv_min_observations(constant_lfo_wrapper, lfo_constant_data, min_obs):
    horizon = 2
    n_time = lfo_constant_data.log_likelihood["obs"].sizes["time"]

    result = lfo_cv(
        lfo_constant_data,
        constant_lfo_wrapper,
        min_observations=min_obs,
        forecast_horizon=horizon,
        method="exact",
    )

    assert result.n_data_points == n_time - horizon - min_obs + 1


def test_lfo_cv_single_forecast_origin(constant_lfo_wrapper, lfo_constant_data):
    n_time = lfo_constant_data.log_likelihood["obs"].sizes["time"]
    horizon = 3
    min_obs = n_time - horizon

    result = lfo_cv(
        lfo_constant_data,
        constant_lfo_wrapper,
        min_observations=min_obs,
        forecast_horizon=horizon,
        method="exact",
        pointwise=True,
    )

    assert result.n_data_points == 1
    assert result.elpd_i.sizes["time"] == 1
    assert result.se == 0.0
    np.testing.assert_allclose(result.elpd, horizon * constant_lfo_wrapper.const)


def test_lfo_cv_custom_time_dim(custom_dim_lfo_wrapper, lfo_custom_dim_data):
    min_obs, horizon = 5, 2
    n_time = lfo_custom_dim_data.log_likelihood["obs"].sizes["week"]

    result = lfo_cv(
        lfo_custom_dim_data,
        custom_dim_lfo_wrapper,
        min_observations=min_obs,
        forecast_horizon=horizon,
        time_dim="week",
        method="exact",
        pointwise=True,
    )

    assert result.kind == "lfo_cv"
    assert result.n_data_points == n_time - horizon - min_obs + 1
    assert "week" in result.elpd_i.dims


def test_lfo_cv_invalid_time_dim(constant_lfo_wrapper, lfo_constant_data):
    with pytest.raises(ValueError, match="Time dimension 'not_a_dim' not found"):
        lfo_cv(
            lfo_constant_data,
            constant_lfo_wrapper,
            min_observations=5,
            forecast_horizon=3,
            time_dim="not_a_dim",
        )


def test_lfo_cv_elpd_and_se(varying_lfo_wrapper, lfo_varying_data):
    result = lfo_cv(
        lfo_varying_data,
        varying_lfo_wrapper,
        min_observations=8,
        forecast_horizon=2,
        method="exact",
        pointwise=True,
    )

    elpd_i = result.elpd_i.values
    assert np.isclose(elpd_i.sum(), result.elpd)
    assert np.isclose(result.se, np.sqrt(result.n_data_points * np.var(elpd_i)))


def test_lfo_cv_p_nonzero(constant_lfo_wrapper, lfo_constant_data):
    log_lik = lfo_constant_data.log_likelihood["obs"]
    n_chains, n_draws = log_lik.sizes["chain"], log_lik.sizes["draw"]
    n_time = log_lik.sizes["time"]
    data = azb.from_dict(
        {
            "posterior": {"mu": np.zeros((n_chains, n_draws))},
            "log_likelihood": {"obs": np.full((n_chains, n_draws, n_time), -2.0)},
            "observed_data": {"obs": np.arange(n_time, dtype=float)},
        },
        dims={"obs": ["time"]},
        coords={"time": np.arange(n_time)},
    )
    min_obs, horizon = 5, 3
    n_origins = n_time - horizon - min_obs + 1

    result = lfo_cv(
        data,
        constant_lfo_wrapper,
        min_observations=min_obs,
        forecast_horizon=horizon,
        method="exact",
        pointwise=True,
    )

    expected_p = n_origins * horizon * (-2.0 - constant_lfo_wrapper.const)
    assert not np.isclose(result.p, 0.0)
    assert np.isclose(result.p, expected_p)


def test_lfo_cv_approx_tracks_exact(varying_lfo_wrapper, lfo_varying_data):
    exact = lfo_cv(
        lfo_varying_data,
        varying_lfo_wrapper,
        min_observations=8,
        forecast_horizon=2,
        method="exact",
        pointwise=True,
    )
    approx = lfo_cv(
        lfo_varying_data,
        varying_lfo_wrapper,
        min_observations=8,
        forecast_horizon=2,
        method="approx",
        pointwise=True,
    )

    np.testing.assert_allclose(approx.elpd_i.values, exact.elpd_i.values, atol=0.1)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_lfo_cv_forced_refit_matches_exact(varying_lfo_wrapper, lfo_varying_data):
    exact = lfo_cv(
        lfo_varying_data,
        varying_lfo_wrapper,
        min_observations=8,
        forecast_horizon=2,
        method="exact",
        pointwise=True,
    )
    forced = lfo_cv(
        lfo_varying_data,
        varying_lfo_wrapper,
        min_observations=8,
        forecast_horizon=2,
        method="approx",
        k_threshold=-100.0,
        pointwise=True,
    )

    np.testing.assert_allclose(forced.elpd_i.values, exact.elpd_i.values, atol=1e-10)


def test_lfo_cv_approx_skips_refits(varying_lfo_wrapper, lfo_varying_data):
    result = lfo_cv(
        lfo_varying_data,
        varying_lfo_wrapper,
        min_observations=8,
        forecast_horizon=2,
        method="approx",
        k_threshold=0.7,
        pointwise=True,
    )

    assert result.n_refits < result.n_data_points
    assert result.pareto_k.sizes["time"] == result.n_data_points
    np.testing.assert_array_equal(
        result.pareto_k.coords["time"].values, result.elpd_i.coords["time"].values
    )
    finite_k = result.pareto_k.values[np.isfinite(result.pareto_k.values)]
    assert finite_k.size > 0
    assert np.all(finite_k <= 0.7)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_lfo_cv_pointwise_false(constant_lfo_wrapper, lfo_constant_data):
    result = lfo_cv(
        lfo_constant_data,
        constant_lfo_wrapper,
        min_observations=5,
        forecast_horizon=3,
        method="approx",
        pointwise=False,
    )

    assert result.elpd_i is None
    assert result.pareto_k is None


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_lfo_cv_metadata(constant_lfo_wrapper, lfo_constant_data):
    result = lfo_cv(
        lfo_constant_data,
        constant_lfo_wrapper,
        min_observations=5,
        forecast_horizon=3,
        method="approx",
    )

    assert result.forecast_horizon == 3
    assert result.min_observations == 5
    assert result.good_k == 0.7
    assert isinstance(result.refits, np.ndarray)
    assert result.n_refits == len(result.refits)


def test_lfo_cv_exact_has_no_pareto_k(constant_lfo_wrapper, lfo_constant_data):
    result = lfo_cv(
        lfo_constant_data,
        constant_lfo_wrapper,
        min_observations=5,
        forecast_horizon=3,
        method="exact",
        pointwise=True,
    )

    assert result.pareto_k is None
    assert result.good_k is None


def test_lfo_cv_invalid_method(constant_lfo_wrapper, lfo_constant_data):
    with pytest.raises(ValueError, match="method must be 'exact' or 'approx'"):
        lfo_cv(
            lfo_constant_data,
            constant_lfo_wrapper,
            min_observations=5,
            forecast_horizon=3,
            method="bogus",
        )


@pytest.mark.parametrize(
    "min_obs, horizon, match",
    [
        (0, 2, "min_observations must be a positive integer"),
        (5, 0, "forecast_horizon must be a positive integer"),
        (12, 2, "must be less than"),
        (10, 5, "exceeds the number of"),
    ],
)
def test_lfo_cv_invalid_parameters(
    constant_lfo_wrapper, lfo_constant_data, min_obs, horizon, match
):
    with pytest.raises(ValueError, match=match):
        lfo_cv(
            lfo_constant_data,
            constant_lfo_wrapper,
            min_observations=min_obs,
            forecast_horizon=horizon,
        )
