"""Tests for LFO-CV helper functions."""

# pylint: disable=redefined-outer-name
import numpy as np
import pytest

from ..helpers import importorskip

azb = importorskip("arviz_base")
xr = importorskip("xarray")

from arviz_stats.loo.lfo_cv_helper import (
    LFOInputs,
    _prepare_lfo_inputs,
    _validate_lfo_parameters,
)


@pytest.mark.parametrize(
    "min_obs, horizon, n_time, valid",
    [
        (10, 5, 30, True),
        (1, 1, 2, True),
        (0, 5, 30, False),
        (10, 0, 30, False),
        (30, 5, 30, False),
        (28, 5, 30, False),
    ],
)
def test_validate_lfo_parameters(min_obs, horizon, n_time, valid):
    if valid:
        _validate_lfo_parameters(min_obs, horizon, n_time)
    else:
        with pytest.raises(ValueError):
            _validate_lfo_parameters(min_obs, horizon, n_time)


def test_prepare_lfo_inputs(constant_lfo_wrapper, lfo_constant_data):
    log_lik = lfo_constant_data.log_likelihood["obs"]
    inputs = _prepare_lfo_inputs(lfo_constant_data, None, constant_lfo_wrapper, 5, 3, "time")

    assert isinstance(inputs, LFOInputs)
    assert inputs.n_time_points == log_lik.sizes["time"]
    assert inputs.n_samples == log_lik.sizes["chain"] * log_lik.sizes["draw"]
    assert inputs.time_dim == "time"
    assert inputs.sample_dims == ["chain", "draw"]


def test_prepare_lfo_inputs_selects_var_name(constant_lfo_wrapper):
    n_chains, n_draws, n_time = 2, 20, 12
    data = azb.from_dict(
        {
            "posterior": {"mu": np.zeros((n_chains, n_draws))},
            "log_likelihood": {
                "obs": np.full((n_chains, n_draws, n_time), -1.0),
                "decoy": np.zeros((n_chains, n_draws, n_time)),
            },
            "observed_data": {"obs": np.arange(n_time, dtype=float)},
        },
        dims={"obs": ["time"], "decoy": ["time"]},
        coords={"time": np.arange(n_time)},
    )

    inputs = _prepare_lfo_inputs(data, "obs", constant_lfo_wrapper, 5, 3, "time")

    assert inputs.log_likelihood.name == "obs"
    assert inputs.n_time_points == n_time


def test_prepare_lfo_inputs_missing_time_dim(constant_lfo_wrapper, lfo_constant_data):
    with pytest.raises(ValueError, match="Time dimension 'week' not found"):
        _prepare_lfo_inputs(lfo_constant_data, None, constant_lfo_wrapper, 5, 3, "week")


def test_prepare_lfo_inputs_requires_wrapper(lfo_constant_data):
    with pytest.raises(TypeError, match="wrapper must be an instance of SamplingWrapper"):
        _prepare_lfo_inputs(lfo_constant_data, None, object(), 5, 3, "time")


def test_prepare_lfo_inputs_incomplete_wrapper(lfo_constant_data):
    from arviz_stats.loo.wrapper import SamplingWrapper

    bare_wrapper = SamplingWrapper(model=None, idata_orig=lfo_constant_data)
    with pytest.raises(ValueError, match="must be implemented"):
        _prepare_lfo_inputs(lfo_constant_data, None, bare_wrapper, 5, 3, "time")


def test_prepare_lfo_inputs_rejects_extra_obs_dims(constant_lfo_wrapper):
    n_chains, n_draws = 2, 20
    data = azb.from_dict(
        {
            "posterior": {"mu": np.zeros((n_chains, n_draws))},
            "log_likelihood": {"obs": np.zeros((n_chains, n_draws, 6, 2))},
            "observed_data": {"obs": np.zeros((6, 2))},
        },
        dims={"obs": ["time", "group"]},
        coords={"time": np.arange(6), "group": [0, 1]},
    )

    with pytest.raises(ValueError, match="single time dimension"):
        _prepare_lfo_inputs(data, None, constant_lfo_wrapper, 3, 1, "time")
