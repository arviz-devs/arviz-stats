"""Test PSIS-LFO-CV (Leave-Future-Out Cross-Validation)."""

# pylint: disable=redefined-outer-name
import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from ..helpers import importorskip

azb = importorskip("arviz_base")
xr = importorskip("xarray")

from arviz_stats import lfo_cv
from arviz_stats.loo.wrapper import SamplingWrapper
from arviz_stats.utils import ELPDData


# Basic functionality tests
@pytest.mark.parametrize("pointwise", [True, False])
def test_lfo_cv_basic(data_with_time, fresh_lfo_wrapper, pointwise):
    """Test basic LFO-CV functionality."""
    result = lfo_cv(
        data=data_with_time,
        wrapper=fresh_lfo_wrapper,
        min_observations=10,
        forecast_horizon=5,
        pointwise=pointwise,
    )

    assert isinstance(result, ELPDData)
    assert result.kind == "lfo_cv"
    assert result.scale == "log"
    assert result.n_data_points > 0
    assert result.n_samples > 0
    assert np.isfinite(result.elpd)
    assert np.isfinite(result.se)
    assert np.isfinite(result.p)
    assert fresh_lfo_wrapper.fit_count >= 1

    if pointwise:
        assert result.elpd_i is not None
        assert result.pareto_k is not None
    else:
        assert result.elpd_i is None
        assert result.pareto_k is None


def test_lfo_cv_pointwise_sum_equals_total(data_with_time, fresh_lfo_wrapper):
    """Test that pointwise ELPD sums to total."""
    result = lfo_cv(
        data=data_with_time,
        wrapper=fresh_lfo_wrapper,
        min_observations=10,
        forecast_horizon=5,
        pointwise=True,
    )

    elpd_sum = np.sum(result.elpd_i.values)
    assert_almost_equal(result.elpd, elpd_sum, decimal=10)


def test_lfo_cv_finite_values(data_with_time, fresh_lfo_wrapper):
    """Test that all returned values are finite."""
    result = lfo_cv(
        data=data_with_time,
        wrapper=fresh_lfo_wrapper,
        min_observations=10,
        forecast_horizon=5,
        pointwise=True,
    )

    assert np.isfinite(result.elpd)
    assert np.isfinite(result.se)
    assert np.isfinite(result.p)
    assert np.all(np.isfinite(result.elpd_i.values))
    if result.pareto_k is not None:
        assert np.all(np.isfinite(result.pareto_k.values))


def test_lfo_cv_se_positive(data_with_time, fresh_lfo_wrapper):
    """Test that standard error is positive."""
    result = lfo_cv(
        data=data_with_time,
        wrapper=fresh_lfo_wrapper,
        min_observations=10,
        forecast_horizon=5,
    )
    assert result.se > 0


# Parameter validation tests
@pytest.mark.parametrize("forecast_horizon", [1, 2, 4, 8])
def test_lfo_cv_forecast_horizon(data_with_time, fresh_lfo_wrapper, forecast_horizon):
    """Test different forecast horizons."""
    result = lfo_cv(
        data=data_with_time,
        wrapper=fresh_lfo_wrapper,
        min_observations=10,
        forecast_horizon=forecast_horizon,
    )

    assert result.kind == "lfo_cv"
    assert hasattr(result, "forecast_horizon")
    assert result.forecast_horizon == forecast_horizon
    assert fresh_lfo_wrapper.fit_count >= 1


@pytest.mark.parametrize("min_observations", [5, 10, 15])
def test_lfo_cv_min_observations(data_with_time, fresh_lfo_wrapper, min_observations):
    """Test different minimum observation requirements."""
    result = lfo_cv(
        data=data_with_time,
        wrapper=fresh_lfo_wrapper,
        min_observations=min_observations,
        forecast_horizon=5,
    )

    assert result.kind == "lfo_cv"
    assert hasattr(result, "min_observations")
    assert result.min_observations == min_observations

    # Verify all training sets have at least min_observations
    for train_indices in fresh_lfo_wrapper.training_indices_history:
        assert len(train_indices) >= min_observations


@pytest.mark.parametrize("k_threshold", [0.5, 0.7, 0.9, 1.0])
def test_lfo_cv_k_threshold(data_with_time, fresh_lfo_wrapper, k_threshold):
    """Test different k-threshold values."""
    result = lfo_cv(
        data=data_with_time,
        wrapper=fresh_lfo_wrapper,
        min_observations=10,
        forecast_horizon=5,
        k_threshold=k_threshold,
        pointwise=True,
    )

    assert result.kind == "lfo_cv"
    assert np.isfinite(result.elpd)


def test_lfo_cv_k_threshold_none(data_with_time, fresh_lfo_wrapper):
    """Test with k_threshold=None (use default)."""
    result = lfo_cv(
        data=data_with_time,
        wrapper=fresh_lfo_wrapper,
        min_observations=10,
        forecast_horizon=5,
        k_threshold=None,
    )

    assert isinstance(result, ELPDData)


# Time dimension tests
def test_lfo_cv_time_dimension_default(data_with_time, fresh_lfo_wrapper):
    """Test with default time dimension."""
    result = lfo_cv(
        data=data_with_time,
        wrapper=fresh_lfo_wrapper,
        min_observations=10,
        forecast_horizon=5,
    )

    assert result.kind == "lfo_cv"


def test_lfo_cv_time_dimension_custom(data_with_time, fresh_lfo_wrapper):
    """Test with explicitly specified time dimension name."""
    result = lfo_cv(
        data=data_with_time,
        wrapper=fresh_lfo_wrapper,
        min_observations=10,
        forecast_horizon=5,
        time_dim="time",
    )

    assert result.kind == "lfo_cv"


def test_lfo_cv_time_dimension_validation(data_with_time, fresh_lfo_wrapper):
    """Test time dimension validation error."""
    with pytest.raises(ValueError, match="[Tt]ime dimension.*not found|not found"):
        lfo_cv(
            data=data_with_time,
            wrapper=fresh_lfo_wrapper,
            min_observations=10,
            forecast_horizon=5,
            time_dim="invalid_dim",
        )


# Method tests (Exact vs Approximate)
@pytest.mark.parametrize("method", ["exact", "approx"])
def test_lfo_cv_methods(data_with_time, fresh_lfo_wrapper, method):
    """Test exact and approximate methods."""
    result = lfo_cv(
        data=data_with_time,
        wrapper=fresh_lfo_wrapper,
        min_observations=10,
        forecast_horizon=5,
        method=method,
    )

    assert result.kind == "lfo_cv"
    assert np.isfinite(result.elpd)


def test_lfo_cv_exact_refits_all(data_with_time, fresh_lfo_wrapper):
    """Test that exact method refits for every time step."""
    n_time = 30  # From data_with_time fixture
    min_obs = 10

    lfo_cv(
        data=data_with_time,
        wrapper=fresh_lfo_wrapper,
        min_observations=min_obs,
        forecast_horizon=5,
        method="exact",
    )

    # Exact method refits for each evaluation point from min_obs to n_time-1
    # (allows reduced forecast horizon for final steps)
    expected_refits = n_time - min_obs  # 30 - 10 = 20
    assert fresh_lfo_wrapper.fit_count == expected_refits


def test_lfo_cv_approx_fewer_refits(data_with_time, fresh_lfo_wrapper):
    """Test that approximate method can have fewer or equal refits."""
    n_time = 30
    min_obs = 10
    forecast_horizon = 5

    lfo_cv(
        data=data_with_time,
        wrapper=fresh_lfo_wrapper,
        min_observations=min_obs,
        forecast_horizon=forecast_horizon,
        method="approx",
        k_threshold=0.7,
    )

    # Approximate should have at most as many refits as exact
    max_refits = n_time - min_obs - forecast_horizon + 1
    assert fresh_lfo_wrapper.fit_count <= max_refits
    assert fresh_lfo_wrapper.fit_count >= 1  # At least initial fit


# Wrapper validation tests
def test_lfo_cv_wrapper_validation(data_with_time):
    """Test wrapper type validation."""
    with pytest.raises(TypeError, match="wrapper must be an instance of SamplingWrapper"):
        lfo_cv(
            data=data_with_time,
            wrapper="not_a_wrapper",
            min_observations=10,
            forecast_horizon=5,
        )


def test_lfo_cv_incomplete_wrapper(data_with_time):
    """Test that incomplete wrapper raises error."""

    class IncompleteWrapper(SamplingWrapper):  # pylint: disable=abstract-method
        def sel_observations(self, idx):
            return None, None

    incomplete = IncompleteWrapper(model=None)
    with pytest.raises(ValueError, match="must be implemented"):
        lfo_cv(
            data=data_with_time,
            wrapper=incomplete,
            min_observations=10,
            forecast_horizon=5,
        )


def test_lfo_cv_wrapper_method_calls(data_with_time, fresh_lfo_wrapper):
    """Test that wrapper methods are called correctly."""
    lfo_cv(
        data=data_with_time,
        wrapper=fresh_lfo_wrapper,
        min_observations=10,
        forecast_horizon=5,
    )

    # Verify methods were called
    assert fresh_lfo_wrapper.fit_count > 0
    assert len(fresh_lfo_wrapper.training_indices_history) == fresh_lfo_wrapper.fit_count
    assert len(fresh_lfo_wrapper.test_indices_history) == fresh_lfo_wrapper.fit_count


# Error condition tests
@pytest.mark.parametrize(
    "error_case,expected_error,expected_msg",
    [
        ("invalid_wrapper", TypeError, "wrapper must be an instance of SamplingWrapper"),
        ("invalid_time_dim", ValueError, "not found"),
        ("min_obs_too_large", ValueError, "min_observations|Insufficient"),
        ("forecast_too_large", ValueError, "forecast_horizon|Insufficient"),
        ("insufficient_data", ValueError, "Insufficient|min_observations|forecast_horizon"),
        ("min_obs_not_positive", ValueError, "min_observations.*positive|must be positive"),
        ("forecast_not_positive", ValueError, "forecast_horizon.*positive|must be positive"),
    ],
)
def test_lfo_cv_errors(data_with_time, fresh_lfo_wrapper, error_case, expected_error, expected_msg):
    """Test error conditions."""
    if error_case == "invalid_wrapper":
        with pytest.raises(expected_error, match=expected_msg):
            lfo_cv(
                data=data_with_time,
                wrapper="not_a_wrapper",
                min_observations=10,
                forecast_horizon=5,
            )
    elif error_case == "invalid_time_dim":
        with pytest.raises(expected_error, match=expected_msg):
            lfo_cv(
                data=data_with_time,
                wrapper=fresh_lfo_wrapper,
                min_observations=10,
                forecast_horizon=5,
                time_dim="invalid",
            )
    elif error_case == "min_obs_too_large":
        with pytest.raises(expected_error, match=expected_msg):
            lfo_cv(
                data=data_with_time,
                wrapper=fresh_lfo_wrapper,
                min_observations=100,  # Larger than data
                forecast_horizon=5,
            )
    elif error_case == "forecast_too_large":
        with pytest.raises(expected_error, match=expected_msg):
            lfo_cv(
                data=data_with_time,
                wrapper=fresh_lfo_wrapper,
                min_observations=10,
                forecast_horizon=100,  # Larger than data
            )
    elif error_case == "insufficient_data":
        with pytest.raises(expected_error, match=expected_msg):
            lfo_cv(
                data=data_with_time,
                wrapper=fresh_lfo_wrapper,
                min_observations=25,
                forecast_horizon=10,  # 25 + 10 > 30
            )
    elif error_case == "min_obs_not_positive":
        with pytest.raises(expected_error, match=expected_msg):
            lfo_cv(
                data=data_with_time,
                wrapper=fresh_lfo_wrapper,
                min_observations=0,
                forecast_horizon=5,
            )
    elif error_case == "forecast_not_positive":
        with pytest.raises(expected_error, match=expected_msg):
            lfo_cv(
                data=data_with_time,
                wrapper=fresh_lfo_wrapper,
                min_observations=10,
                forecast_horizon=0,
            )


# Edge case tests
def test_lfo_cv_single_step_forecast(data_with_time, fresh_lfo_wrapper):
    """Test with forecast_horizon=1 (single step ahead)."""
    result = lfo_cv(
        data=data_with_time,
        wrapper=fresh_lfo_wrapper,
        min_observations=10,
        forecast_horizon=1,
    )

    assert result.kind == "lfo_cv"
    assert result.forecast_horizon == 1


# Integration tests
def test_lfo_cv_elpd_data_fields(data_with_time, fresh_lfo_wrapper):
    """Test that ELPDData has all expected LFO-CV fields."""
    result = lfo_cv(
        data=data_with_time,
        wrapper=fresh_lfo_wrapper,
        min_observations=10,
        forecast_horizon=5,
        pointwise=True,
    )

    # Standard ELPDData fields
    assert hasattr(result, "elpd")
    assert hasattr(result, "se")
    assert hasattr(result, "p")
    assert hasattr(result, "n_samples")
    assert hasattr(result, "n_data_points")
    assert hasattr(result, "warning")
    assert hasattr(result, "kind")
    assert hasattr(result, "scale")
    assert hasattr(result, "elpd_i")
    assert hasattr(result, "pareto_k")

    # LFO-CV specific fields
    assert hasattr(result, "forecast_horizon")
    assert hasattr(result, "min_observations")


@pytest.mark.parametrize("save_fits", [True, False])
def test_lfo_cv_save_fits(data_with_time, fresh_lfo_wrapper, save_fits):
    """Test save_fits parameter."""
    result = lfo_cv(
        data=data_with_time,
        wrapper=fresh_lfo_wrapper,
        min_observations=10,
        forecast_horizon=5,
        save_fits=save_fits,
    )

    if save_fits:
        assert hasattr(result, "refits") or hasattr(result, "fits")
    # When save_fits=False, no fits should be stored
