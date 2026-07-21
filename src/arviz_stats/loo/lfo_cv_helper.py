"""Helper functions for Leave-Future-Out Cross-Validation (LFO-CV)."""

from collections import namedtuple

import numpy as np
import xarray as xr
from arviz_base import convert_to_datatree
from xarray_einstats.stats import logsumexp

from arviz_stats.loo.loo_helper import _get_r_eff
from arviz_stats.loo.wrapper import SamplingWrapper
from arviz_stats.utils import get_log_likelihood

__all__ = [
    "_prepare_lfo_inputs",
    "_compute_lfo_exact",
    "_compute_lfo_approx",
    "_validate_lfo_parameters",
]

LFOInputs = namedtuple(
    "LFOInputs",
    [
        "log_likelihood",
        "sample_dims",
        "time_dim",
        "n_samples",
        "n_time_points",
        "min_observations",
        "forecast_horizon",
    ],
)

LFOResults = namedtuple(
    "LFOResults",
    ["elpd", "se", "p", "n_data_points", "elpd_i", "p_lfo_i", "refits", "pareto_k"],
)


def _prepare_lfo_inputs(data, var_name, wrapper, min_observations, forecast_horizon, time_dim):
    """Validate arguments and collect the quantities shared by every LFO-CV step."""
    data = convert_to_datatree(data)

    if not isinstance(wrapper, SamplingWrapper):
        raise TypeError("wrapper must be an instance of SamplingWrapper")

    required_methods = ["sel_observations", "sample", "get_inference_data", "log_likelihood__i"]
    not_implemented = wrapper.check_implemented_methods(required_methods)
    if not_implemented:
        raise ValueError(
            f"The following methods must be implemented in the SamplingWrapper: {not_implemented}"
        )

    log_likelihood = get_log_likelihood(data, var_name)

    sample_dims = ["chain", "draw"]
    if time_dim not in log_likelihood.dims:
        raise ValueError(
            f"Time dimension '{time_dim}' not found in log_likelihood. "
            f"Available dimensions: {list(log_likelihood.dims)}"
        )

    obs_dims = [dim for dim in log_likelihood.dims if dim not in sample_dims]
    if obs_dims != [time_dim]:
        raise ValueError(
            f"lfo_cv currently supports a single time dimension; found observation "
            f"dimensions {obs_dims}. Expected only '{time_dim}'."
        )

    n_samples = int(np.prod([log_likelihood.sizes[dim] for dim in sample_dims]))

    _validate_lfo_parameters(min_observations, forecast_horizon, log_likelihood.sizes[time_dim])

    return LFOInputs(
        log_likelihood=log_likelihood,
        sample_dims=sample_dims,
        time_dim=time_dim,
        n_samples=n_samples,
        n_time_points=log_likelihood.sizes[time_dim],
        min_observations=min_observations,
        forecast_horizon=forecast_horizon,
    )


def _compute_lfo_exact(lfo_inputs, wrapper):
    """Compute LFO-CV by refitting the model at every forecast origin.

    For each origin ``i`` the model is refit on ``y[:i]`` and scored on the joint predictive
    density of the block ``y[i:i + forecast_horizon]``.

    Parameters
    ----------
    lfo_inputs : LFOInputs
        Prepared inputs from ``_prepare_lfo_inputs``.
    wrapper : SamplingWrapper
        Wrapper instance handling model refitting.

    Returns
    -------
    LFOResults
        A namedtuple containing:

        - elpd: Total expected log pointwise predictive density
        - se: Standard error of the elpd
        - p: Effective number of parameters
        - n_data_points: Number of forecast origins evaluated
        - elpd_i: Per-origin elpd values along the time dimension
        - p_lfo_i: Per-origin effective number of parameters
        - refits: Time indices where refits occurred, every origin for the exact method
        - pareto_k: None for the exact method
    """
    ll_full = lfo_inputs.log_likelihood
    sample_dims = lfo_inputs.sample_dims
    n_samples = lfo_inputs.n_samples
    time_dim = lfo_inputs.time_dim
    horizon = lfo_inputs.forecast_horizon

    origins = np.arange(lfo_inputs.min_observations, lfo_inputs.n_time_points - horizon + 1)
    elpds = np.empty(len(origins))
    lpds = np.empty(len(origins))
    for pos, cutoff in enumerate(origins):
        log_lik, dims, n_refit, _ = _refit_loglik(lfo_inputs, wrapper, cutoff)
        block = log_lik.isel({time_dim: slice(0, horizon)}).sum(time_dim)
        elpds[pos] = logsumexp(block, dims=dims, b=1 / n_refit)

        ll_block = ll_full.isel({time_dim: slice(cutoff, cutoff + horizon)}).sum(time_dim)
        lpds[pos] = logsumexp(ll_block, dims=sample_dims, b=1 / n_samples)

    return _assemble_results(lfo_inputs, origins, elpds, lpds, origins, None)


def _compute_lfo_approx(lfo_inputs, wrapper, k_threshold):
    """Compute LFO-CV with Pareto-smoothed importance sampling between refits.

    Starting from a fit on ``y[:min_observations]``, the posterior is carried forward with
    importance weights over the observations added since the last refit, and the model is only
    refit when the Pareto :math:`k` of those weights exceeds ``k_threshold``.

    Parameters
    ----------
    lfo_inputs : LFOInputs
        Prepared inputs from ``_prepare_lfo_inputs``.
    wrapper : SamplingWrapper
        Wrapper instance handling model refitting.
    k_threshold : float
        Pareto k threshold above which the model is refit.

    Returns
    -------
    LFOResults
        A namedtuple containing:

        - elpd: Total expected log pointwise predictive density
        - se: Standard error of the elpd
        - p: Effective number of parameters
        - n_data_points: Number of forecast origins evaluated
        - elpd_i: Per-origin elpd values along the time dimension
        - p_lfo_i: Per-origin effective number of parameters
        - refits: Time indices where PSIS triggered a refit
        - pareto_k: Per-origin Pareto k values, NaN at the first origin and wherever a
          refit occurred
    """
    ll_full = lfo_inputs.log_likelihood
    sample_dims = lfo_inputs.sample_dims
    n_samples = lfo_inputs.n_samples
    time_dim = lfo_inputs.time_dim
    horizon = lfo_inputs.forecast_horizon

    origins = np.arange(lfo_inputs.min_observations, lfo_inputs.n_time_points - horizon + 1)
    last_refit = origins[0]
    ll_star, star_dims, n_star, idata_star = _refit_loglik(lfo_inputs, wrapper, last_refit)
    r_eff = _get_r_eff(idata_star, n_star)

    elpds = np.empty(len(origins))
    lpds = np.empty(len(origins))
    pareto_ks = np.full(len(origins), np.nan)
    refits = []

    for pos, cutoff in enumerate(origins):
        offset = cutoff - last_refit
        if offset == 0:
            log_weights, pareto_k = None, np.nan
        else:
            log_ratios = ll_star.isel({time_dim: slice(0, offset)}).sum(time_dim)
            try:
                log_weights, pareto_k = (-log_ratios).azstats.psislw(dim=star_dims, r_eff=r_eff)
            except ValueError:
                log_weights, pareto_k = None, np.inf
            if pareto_k > k_threshold:
                ll_star, star_dims, n_star, idata_star = _refit_loglik(lfo_inputs, wrapper, cutoff)
                r_eff = _get_r_eff(idata_star, n_star)
                last_refit = cutoff
                offset = 0
                log_weights, pareto_k = None, np.nan
                refits.append(cutoff)

        block = ll_star.isel({time_dim: slice(offset, offset + horizon)}).sum(time_dim)
        if log_weights is None:
            log_weights = xr.zeros_like(block)
        weighted = logsumexp(log_weights + block, dims=star_dims)
        elpds[pos] = weighted - logsumexp(log_weights, dims=star_dims)

        ll_block = ll_full.isel({time_dim: slice(cutoff, cutoff + horizon)}).sum(time_dim)
        lpds[pos] = logsumexp(ll_block, dims=sample_dims, b=1 / n_samples)
        pareto_ks[pos] = pareto_k

    refits = np.array(refits, dtype=int)
    return _assemble_results(lfo_inputs, origins, elpds, lpds, refits, pareto_ks)


def _refit_loglik(lfo_inputs, wrapper, cutoff):
    """Refit on ``y[:cutoff]`` and return the log likelihood of all remaining observations.

    The returned array is indexed so that position ``0`` corresponds to observation ``cutoff``,
    which lets a caller slice out the importance-ratio observations and the forecast block by
    their offset from the refit.

    Parameters
    ----------
    lfo_inputs : LFOInputs
        Prepared inputs from ``_prepare_lfo_inputs``.
    wrapper : SamplingWrapper
        Wrapper instance handling model refitting.
    cutoff : int
        Number of leading observations to train on.

    Returns
    -------
    log_lik : DataArray
        Log likelihood of observations ``cutoff`` onward, evaluated at the refit
        posterior draws.
    sample_dims : list of str
        Dimensions of ``log_lik`` other than the time dimension.
    n_samples : int
        Number of posterior draws in the refit.
    idata : DataTree
        Inference data of the refit returned by the wrapper.
    """
    exclude_idx = np.arange(cutoff, lfo_inputs.n_time_points)
    train_data, excluded_data = wrapper.sel_observations(exclude_idx)
    idata = wrapper.get_inference_data(wrapper.sample(train_data))

    log_lik = wrapper.log_likelihood__i(excluded_data, idata)
    sample_dims = [dim for dim in log_lik.dims if dim != lfo_inputs.time_dim]
    n_samples = np.prod([log_lik.sizes[dim] for dim in sample_dims])
    return log_lik, sample_dims, n_samples, idata


def _assemble_results(lfo_inputs, origins, elpds, lpds, refits, pareto_k_values):
    """Build the per-origin DataArrays and aggregate totals shared by both methods."""
    time_dim = lfo_inputs.time_dim
    ps = lpds - elpds
    origin_coord = lfo_inputs.log_likelihood.coords[time_dim].isel({time_dim: origins}).values

    elpd_i = xr.DataArray(elpds, dims=[time_dim], coords={time_dim: origin_coord})
    p_lfo_i = xr.DataArray(ps, dims=[time_dim], coords={time_dim: origin_coord})
    pareto_k = None
    if pareto_k_values is not None:
        pareto_k = xr.DataArray(pareto_k_values, dims=[time_dim], coords={time_dim: origin_coord})

    n_data_points = len(elpds)
    se = np.sqrt(n_data_points * np.var(elpds)) if n_data_points > 1 else 0.0

    return LFOResults(
        elpd=np.sum(elpds),
        se=se,
        p=np.sum(ps),
        n_data_points=n_data_points,
        elpd_i=elpd_i,
        p_lfo_i=p_lfo_i,
        refits=refits,
        pareto_k=pareto_k,
    )


def _validate_lfo_parameters(min_observations, forecast_horizon, n_time_points):
    """Check that the LFO-CV window parameters are consistent with the data."""
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
