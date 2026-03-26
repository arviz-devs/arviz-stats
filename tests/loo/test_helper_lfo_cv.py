"""Test helper functions for LFO-CV."""

# pylint: disable=redefined-outer-name
import pytest

from ..helpers import importorskip

azb = importorskip("arviz_base")
xr = importorskip("xarray")

from arviz_stats.loo.helper_lfo_cv import (
    LFOInputs,
    LFOResults,
    LFOStepResult,
    _combine_lfo_elpds,
    _prepare_lfo_inputs,
    _validate_lfo_parameters,
)


# Parameter validation tests
@pytest.mark.parametrize(
    "min_obs,forecast,n_time,valid",
    [
        (10, 5, 30, True),
        (10, 5, 16, True),  # Minimal
        (10, 5, 14, False),  # Too small
        (0, 5, 30, False),  # min_obs not positive
        (10, 0, 30, False),  # forecast not positive
        (-1, 5, 30, False),  # min_obs negative
        (10, -1, 30, False),  # forecast negative
    ],
)
def test_validate_lfo_parameters(min_obs, forecast, n_time, valid):
    """Test LFO parameter validation."""
    if valid:
        _validate_lfo_parameters(min_obs, forecast, n_time)
    else:
        with pytest.raises(ValueError):
            _validate_lfo_parameters(min_obs, forecast, n_time)


def test_validate_lfo_parameters_boundary():
    """Test boundary conditions for parameter validation."""
    # Exactly enough data: min_obs + forecast = n_time (allows exactly 1 step)
    _validate_lfo_parameters(10, 5, 15)  # 10 + 5 = 15 <= 15, valid

    # One less than needed (not enough room for even 1 forecast)
    with pytest.raises(ValueError):
        _validate_lfo_parameters(10, 5, 14)  # 10 + 5 = 15 > 14, invalid


# Input preparation tests
def test_prepare_lfo_inputs(data_with_time, fresh_lfo_wrapper):
    """Test LFO input preparation."""
    result = _prepare_lfo_inputs(
        data=data_with_time,
        var_name=None,
        wrapper=fresh_lfo_wrapper,
        min_observations=10,
        forecast_horizon=5,
        time_dim="time",
    )

    assert isinstance(result, LFOInputs)
    assert result.log_likelihood is not None
    assert result.n_time_points > 0
    assert result.min_observations == 10
    assert result.forecast_horizon == 5
    assert result.time_dim == "time"


def test_prepare_lfo_inputs_with_var_name(data_with_time, fresh_lfo_wrapper):
    """Test LFO input preparation with explicit var_name."""
    result = _prepare_lfo_inputs(
        data=data_with_time,
        var_name="obs",
        wrapper=fresh_lfo_wrapper,
        min_observations=10,
        forecast_horizon=5,
        time_dim="time",
    )

    assert isinstance(result, LFOInputs)
    assert result.var_name == "obs"


def test_prepare_lfo_inputs_invalid_wrapper(data_with_time):
    """Test that invalid wrapper raises error."""
    with pytest.raises(TypeError, match="wrapper must be an instance of SamplingWrapper"):
        _prepare_lfo_inputs(
            data=data_with_time,
            var_name=None,
            wrapper="not_a_wrapper",
            min_observations=10,
            forecast_horizon=5,
            time_dim="time",
        )


def test_prepare_lfo_inputs_invalid_time_dim(data_with_time, fresh_lfo_wrapper):
    """Test that invalid time dimension raises error."""
    with pytest.raises(ValueError, match="not found"):
        _prepare_lfo_inputs(
            data=data_with_time,
            var_name=None,
            wrapper=fresh_lfo_wrapper,
            min_observations=10,
            forecast_horizon=5,
            time_dim="invalid_dim",
        )


# Result combination tests
def test_combine_lfo_elpds_basic():
    """Test basic combining of LFO ELPD results."""
    import xarray as xr

    # Create mock LFOStepResult objects with xarray DataArrays for elpd_i
    step_results = []
    for i in range(5):
        elpd_val = float(i + 1)
        elpd_i = xr.DataArray([elpd_val], dims=["time"])
        step_results.append(
            LFOStepResult(
                elpd_i=elpd_i,
                pareto_k=0.1 * (i + 1),
                refitted=(i == 0),
                time_index=i,
            )
        )

    result = _combine_lfo_elpds(step_results, "time", 10, 1)

    assert "elpd" in result
    assert "se" in result
    assert "refit_indices" in result
    assert result["se"] >= 0


def test_combine_lfo_elpds_negative():
    """Test combining negative ELPD values."""
    import xarray as xr

    step_results = []
    for i in range(3):
        elpd_val = float(-(i + 1))
        elpd_i = xr.DataArray([elpd_val], dims=["time"])
        step_results.append(
            LFOStepResult(
                elpd_i=elpd_i,
                pareto_k=0.1 * (i + 1),
                refitted=(i == 0),
                time_index=i,
            )
        )

    result = _combine_lfo_elpds(step_results, "time", 10, 1)

    assert result["elpd"] < 0
    assert result["se"] >= 0


def test_combine_lfo_elpds_single_value():
    """Test combining single ELPD value."""
    import xarray as xr

    elpd_i = xr.DataArray([5.0], dims=["time"])
    step_results = [
        LFOStepResult(
            elpd_i=elpd_i,
            pareto_k=0.3,
            refitted=True,
            time_index=0,
        )
    ]

    result = _combine_lfo_elpds(step_results, "time", 10, 1)

    assert "elpd" in result


# Named tuple tests
def test_lfo_inputs_namedtuple():
    """Test LFOInputs named tuple structure."""
    assert hasattr(LFOInputs, "_fields")
    expected_fields = {
        "data",
        "log_likelihood",
        "var_name",
        "sample_dims",
        "obs_dims",
        "time_dim",
        "n_data_points",
        "n_samples",
        "n_time_points",
        "min_observations",
        "forecast_horizon",
    }
    assert set(LFOInputs._fields) == expected_fields


def test_lfo_step_result_namedtuple():
    """Test LFOStepResult named tuple structure."""
    assert hasattr(LFOStepResult, "_fields")
    expected_fields = {"elpd_i", "pareto_k", "refitted", "time_index"}
    assert set(LFOStepResult._fields) == expected_fields


def test_lfo_results_namedtuple():
    """Test LFOResults named tuple structure."""
    assert hasattr(LFOResults, "_fields")
    expected_fields = {"elpd_values", "pareto_k_values", "refit_indices", "time_indices"}
    assert set(LFOResults._fields) == expected_fields
