"""Helper functions for Leave-Future-Out Cross-Validation (LFO-CV)."""

from collections import namedtuple

import numpy as np
import xarray as xr
from arviz_base import convert_to_datatree
from xarray_einstats.stats import logsumexp

from arviz_stats.loo.wrapper import SamplingWrapper
from arviz_stats.utils import get_log_likelihood

__all__ = [
    "_prepare_lfo_inputs",
    "_validate_lfo_parameters",
    "_compute_lfo_step_exact",
    "_compute_lfo_step_approx",
    "_combine_lfo_elpds",
    "_wrap_psislw",
]

# Named tuples for structured data
LFOInputs = namedtuple(
    "LFOInputs",
    [
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
    ],
)

LFOStepResult = namedtuple(
    "LFOStepResult",
    [
        "elpd_i",  # pointwise ELPD values for this step's forecast window
        "pareto_k",  # Pareto k diagnostic (None for exact method)
        "refitted",  # bool indicating if model was refitted
        "time_index",  # the time index where this step starts
    ],
)

LFOResults = namedtuple(
    "LFOResults",
    [
        "elpd_values",  # array of pointwise ELPD values
        "pareto_k_values",  # array of Pareto k values (or None)
        "refit_indices",  # list of time indices where refits occurred
        "time_indices",  # array of time indices evaluated
    ],
)


def _prepare_lfo_inputs(data, var_name, wrapper, min_observations, forecast_horizon, time_dim):
    """Prepare inputs for LFO-CV.

    Parameters
    ----------
    data : DataTree or InferenceData
        Input data containing posterior and log_likelihood groups.
    var_name : str, optional
        Name of the variable in log_likelihood group.
    wrapper : SamplingWrapper
        Wrapper instance for model refitting.
    min_observations : int
        Minimum number of observations required before making predictions.
    forecast_horizon : int
        Number of steps ahead to predict.
    time_dim : str
        Name of the time dimension.

    Returns
    -------
    LfoInputs
        Named tuple with prepared inputs.
    """
    data = convert_to_datatree(data)

    # Validate wrapper
    if not isinstance(wrapper, SamplingWrapper):
        raise TypeError("wrapper must be an instance of SamplingWrapper")

    required_methods = ["sel_observations", "sample", "get_inference_data", "log_likelihood__i"]
    not_implemented = wrapper.check_implemented_methods(required_methods)

    if not_implemented:
        raise ValueError(
            f"The following methods must be implemented in the SamplingWrapper: {not_implemented}"
        )

    # Get log likelihood
    log_likelihood = get_log_likelihood(data, var_name)
    var_name = log_likelihood.name if var_name is None else var_name

    sample_dims = ["chain", "draw"]
    obs_dims = [dim for dim in log_likelihood.dims if dim not in sample_dims]

    # Validate time dimension
    if time_dim not in log_likelihood.dims:
        raise ValueError(
            f"Time dimension '{time_dim}' not found in log_likelihood. "
            f"Available dimensions: {list(log_likelihood.dims)}"
        )

    n_time_points = log_likelihood.sizes[time_dim]
    n_data_points = int(np.prod([log_likelihood.sizes[dim] for dim in obs_dims]))
    n_samples = int(np.prod([log_likelihood.sizes[dim] for dim in sample_dims]))

    # Validate parameters
    _validate_lfo_parameters(min_observations, forecast_horizon, n_time_points)

    return LFOInputs(
        data=data,
        log_likelihood=log_likelihood,
        var_name=var_name,
        sample_dims=sample_dims,
        obs_dims=obs_dims,
        time_dim=time_dim,
        n_data_points=n_data_points,
        n_samples=n_samples,
        n_time_points=n_time_points,
        min_observations=min_observations,
        forecast_horizon=forecast_horizon,
    )


def _validate_lfo_parameters(min_observations, forecast_horizon, n_time_points):
    """Validate LFO-CV parameters.

    Parameters
    ----------
    min_observations : int
        Minimum number of observations required before making predictions.
    forecast_horizon : int
        Number of steps ahead to predict.
    n_time_points : int
        Total number of time points in the data.

    Raises
    ------
    ValueError
        If parameters are invalid.
    """
    if not isinstance(min_observations, int | np.integer) or min_observations < 1:
        raise ValueError(f"min_observations must be a positive integer, got {min_observations}")

    if not isinstance(forecast_horizon, int | np.integer) or forecast_horizon < 1:
        raise ValueError(f"forecast_horizon must be a positive integer, got {forecast_horizon}")

    if min_observations >= n_time_points:
        raise ValueError(
            f"min_observations ({min_observations}) must be less than "
            f"the number of time points ({n_time_points})"
        )

    if min_observations + forecast_horizon > n_time_points:
        raise ValueError(
            f"min_observations ({min_observations}) + forecast_horizon ({forecast_horizon}) "
            f"= {min_observations + forecast_horizon} exceeds the number of "
            f"time points ({n_time_points})"
        )


def _compute_lfo_step_exact(
    lfo_inputs,
    wrapper,
    cutoff,
):
    """Compute LFO-CV for a single time step using exact refitting.

    Parameters
    ----------
    lfo_inputs : LfoInputs
        Prepared inputs from _prepare_lfo_inputs.
    wrapper : SamplingWrapper
        Wrapper instance for model refitting.
    cutoff : int
        Time index cutoff. Observations from cutoff onwards are excluded from training.

    Returns
    -------
    LfoStepResult
        Results for this time step.
    """
    time_dim = lfo_inputs.time_dim
    forecast_horizon = lfo_inputs.forecast_horizon
    n_time_points = lfo_inputs.n_time_points

    # Indices to exclude from training (future observations from cutoff onwards)
    exclude_idx = np.arange(cutoff, n_time_points)

    # Call wrapper to get training and excluded data
    train_data, excluded_data = wrapper.sel_observations(exclude_idx)

    # Refit the model on training data
    fitted_model = wrapper.sample(train_data)
    idata_refitted = wrapper.get_inference_data(fitted_model)

    # Select forecast window from excluded data
    # The forecast window is the first forecast_horizon observations from the excluded data
    actual_forecast_horizon = min(forecast_horizon, n_time_points - cutoff)
    forecast_window = excluded_data.isel({time_dim: slice(0, actual_forecast_horizon)})

    # Compute log-likelihood for forecast window
    log_lik_forecast = wrapper.log_likelihood__i(forecast_window, idata_refitted)

    # Compute ELPD for each forecast point
    if "chain" in log_lik_forecast.dims and "draw" in log_lik_forecast.dims:
        sample_dims_k = ["chain", "draw"]
    else:
        # Handle case where sample dimensions might be different
        sample_dims_k = [dim for dim in log_lik_forecast.dims if dim not in [time_dim]]

    n_samples_k = int(np.prod([log_lik_forecast.sizes[dim] for dim in sample_dims_k]))

    # Compute pointwise ELPD (sum over samples)
    elpd_i = logsumexp(log_lik_forecast, dims=sample_dims_k, b=1 / n_samples_k)

    return LFOStepResult(
        elpd_i=elpd_i,
        pareto_k=None,  # No Pareto k for exact method
        refitted=True,
        time_index=cutoff,
    )


def _compute_lfo_step_approx(
    lfo_inputs,
    wrapper,
    cutoff,
    log_lik_current,
    k_threshold,
    last_refit_cutoff,
):
    """Compute LFO-CV for a single time step using PSIS approximation.

    Uses importance sampling to approximate the leave-future-out predictive
    distribution. If Pareto k exceeds threshold, triggers a refit.

    Parameters
    ----------
    lfo_inputs : LfoInputs
        Prepared inputs from _prepare_lfo_inputs.
    wrapper : SamplingWrapper
        Wrapper instance for model refitting.
    cutoff : int
        Time index cutoff.
    log_lik_current : DataArray
        Log-likelihood from the current posterior (from last refit).
    k_threshold : float
        Pareto k threshold for triggering refit.
    last_refit_cutoff : int
        The cutoff index from the last refit.

    Returns
    -------
    tuple
        (LfoStepResult, needs_refit, new_log_lik)
        - LfoStepResult: Results for this time step
        - needs_refit: bool indicating if a refit is needed
        - new_log_lik: Updated log-likelihood if refitted, else None
    """
    time_dim = lfo_inputs.time_dim
    forecast_horizon = lfo_inputs.forecast_horizon
    n_time_points = lfo_inputs.n_time_points
    sample_dims = lfo_inputs.sample_dims
    n_samples = lfo_inputs.n_samples

    actual_forecast_horizon = min(forecast_horizon, n_time_points - cutoff)

    # Compute importance ratios for observations between last_refit_cutoff and cutoff
    # These are the observations that were in the training set at last refit
    # but are now being "removed" for the LFO calculation
    if last_refit_cutoff < cutoff:
        # Sum log-likelihood of observations that need to be "removed"
        removed_obs_idx = slice(last_refit_cutoff, cutoff)
        log_lik_removed = log_lik_current.isel({time_dim: removed_obs_idx})
        log_lik_removed_sum = log_lik_removed.sum(dim=time_dim)

        # Importance weights: exp(-log_lik_removed) to remove these observations
        log_ratios = -log_lik_removed_sum

        # Apply PSIS
        try:
            log_weights, pareto_k = _wrap_psislw(log_ratios, sample_dims, n_samples)
        except Exception:
            # If PSIS fails, trigger refit
            return None, True, None
    else:
        # No observations to remove, use uniform weights
        log_weights = xr.zeros_like(log_lik_current.isel({time_dim: 0})) - np.log(n_samples)
        pareto_k = 0.0

    # Check if Pareto k exceeds threshold
    # If k_threshold is None, use default of 0.7
    effective_threshold = k_threshold if k_threshold is not None else 0.7
    if pareto_k > effective_threshold:
        return None, True, None

    # Compute ELPD for forecast window using importance weights
    forecast_idx = slice(cutoff, cutoff + actual_forecast_horizon)
    log_lik_forecast = log_lik_current.isel({time_dim: forecast_idx})

    # Weighted ELPD computation
    # elpd_i = logsumexp(log_lik_forecast + log_weights, dims=sample_dims)
    # Need to broadcast log_weights to match log_lik_forecast shape
    log_lik_weighted = log_lik_forecast + log_weights

    elpd_i = logsumexp(log_lik_weighted, dims=sample_dims)

    # Create pareto_k array matching forecast window shape
    pareto_k_array = xr.full_like(elpd_i, pareto_k)

    return (
        LFOStepResult(
            elpd_i=elpd_i,
            pareto_k=pareto_k_array,
            refitted=False,
            time_index=cutoff,
        ),
        False,
        None,
    )


def _combine_lfo_elpds(step_results, time_dim, n_time_points, forecast_horizon):
    """Combine ELPD values from all LFO-CV steps.

    Parameters
    ----------
    step_results : list of LfoStepResult
        Results from each LFO-CV step.
    time_dim : str
        Name of the time dimension.
    n_time_points : int
        Total number of time points.
    forecast_horizon : int
        Forecast horizon.

    Returns
    -------
    dict
        Dictionary containing:
        - elpd: total ELPD
        - se: standard error of ELPD
        - elpd_i: pointwise ELPD values
        - pareto_k: pointwise Pareto k values (or None)
        - refit_indices: array of time indices where refits occurred
        - n_refits: number of refits performed
    """
    # Collect all pointwise ELPD values
    all_elpd_i = []
    all_pareto_k = []
    refit_indices = []

    for result in step_results:
        if result.elpd_i is not None:
            # Flatten if necessary
            elpd_values = result.elpd_i.values.flatten()
            all_elpd_i.extend(elpd_values)

            if result.pareto_k is not None:
                # Handle both float and DataArray pareto_k values
                if hasattr(result.pareto_k, "values"):
                    k_values = result.pareto_k.values.flatten()
                    all_pareto_k.extend(k_values)
                else:
                    # Single float value
                    all_pareto_k.append(float(result.pareto_k))

        if result.refitted:
            refit_indices.append(result.time_index)

    all_elpd_i = np.array(all_elpd_i)
    n_evaluated = len(all_elpd_i)

    # Compute total ELPD and SE
    elpd_total = np.sum(all_elpd_i)
    se_elpd = np.sqrt(n_evaluated * np.var(all_elpd_i, ddof=1)) if n_evaluated > 1 else 0.0

    # Create DataArrays for pointwise values
    elpd_i_da = xr.DataArray(
        all_elpd_i,
        dims=["lfo_point"],
        coords={"lfo_point": np.arange(n_evaluated)},
    )

    pareto_k_da = None
    if all_pareto_k:
        pareto_k_da = xr.DataArray(
            np.array(all_pareto_k),
            dims=["lfo_point"],
            coords={"lfo_point": np.arange(len(all_pareto_k))},
        )

    return {
        "elpd": elpd_total,
        "se": se_elpd,
        "elpd_i": elpd_i_da,
        "pareto_k": pareto_k_da,
        "refit_indices": np.array(refit_indices),
        "n_refits": len(refit_indices),
        "n_data_points": n_evaluated,
    }


def _wrap_psislw(log_ratios, sample_dims, n_samples, r_eff=1.0):
    """Apply PSIS smoothing to log importance ratios.

    Wrapper around the array-layer psislw function for use in LFO-CV.
    Similar to _wrap__psislw in loo_moment_match.py.

    Parameters
    ----------
    log_ratios : DataArray
        Log importance ratios.
    sample_dims : list of str
        Sample dimensions (typically ["chain", "draw"]).
    n_samples : int
        Total number of samples.
    r_eff : float, optional
        Relative effective sample size. Default is 1.0.

    Returns
    -------
    tuple
        (log_weights, pareto_k)
        - log_weights: DataArray with smoothed log importance weights
        - pareto_k: float, Pareto k diagnostic value
    """
    if not isinstance(log_ratios, xr.DataArray):
        raise TypeError("log_ratios must be an xarray.DataArray")

    # Check all sample dimensions are present
    missing_dims = [dim for dim in sample_dims if dim not in log_ratios.dims]
    if missing_dims:
        raise ValueError(
            f"All sample dimensions must be present in the input; missing {missing_dims}."
        )

    # Check for extra dimensions
    other_dims = [dim for dim in log_ratios.dims if dim not in sample_dims]
    if other_dims:
        raise ValueError(
            f"_wrap_psislw expects log_ratios to include only sample dimensions; "
            f"found extra dims {other_dims}."
        )

    # Stack sample dimensions
    stacked = log_ratios.stack(__sample__=sample_dims)

    # Apply PSIS (note: psislw expects negative log-likelihood ratios)
    try:
        lw_stacked, k = stacked.azstats.psislw(dim="__sample__", r_eff=r_eff)
    except ValueError as err:
        err_message = str(err)
        fallback_errors = ("All tail values are the same", "n_draws_tail must be at least 5")
        if not any(msg in err_message for msg in fallback_errors):
            raise

        # Fallback: use normalized weights without PSIS smoothing
        log_norm = logsumexp(stacked, dims="__sample__")
        lw_stacked = stacked - log_norm
        k = np.inf

    # Unstack and return
    log_weights = lw_stacked.unstack("__sample__").transpose(*log_ratios.dims)

    # Extract scalar k value
    if isinstance(k, xr.DataArray):
        k_val = k.item() if k.ndim == 0 else float(k.values)
    elif isinstance(k, np.ndarray):
        k_val = k.item() if k.ndim == 0 else float(k)
    else:
        k_val = float(k)

    return log_weights, k_val
