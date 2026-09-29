"""Helper functions for Leave-Future-Out Cross-Validation (LFO-CV)."""

import warnings
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
    "_compute_lfo",
    "_forecast_origins",
    "_validate_lfo_parameters",
    "_warn_lfo_refits",
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
        "origins",
    ],
)

LFOResults = namedtuple(
    "LFOResults",
    ["elpd", "se", "p", "n_data_points", "elpd_i", "p_lfo_i", "refits", "pareto_k"],
)

LFOFit = namedtuple("LFOFit", ["log_lik", "sample_dims", "n_samples", "idata"])

LFOOrigin = namedtuple(
    "LFOOrigin",
    [
        "pos",
        "cutoff",
        "offset",
        "log_lik",
        "sample_dims",
        "n_samples",
        "idata",
        "log_weights",
        "pareto_k",
        "refit",
    ],
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
            "lfo_cv currently requires one log-likelihood value per time point. "
            f"Found observation dimensions {obs_dims}, expected only '{time_dim}'. "
            "Combine a multivariate likelihood into one joint log-likelihood value per time "
            "point before calling lfo_cv."
        )

    n_samples = int(np.prod([log_likelihood.sizes[dim] for dim in sample_dims]))

    _validate_lfo_parameters(min_observations, forecast_horizon, log_likelihood.sizes[time_dim])
    origins = np.arange(min_observations, log_likelihood.sizes[time_dim] - forecast_horizon + 1)

    return LFOInputs(
        log_likelihood=log_likelihood,
        sample_dims=sample_dims,
        time_dim=time_dim,
        n_samples=n_samples,
        n_time_points=log_likelihood.sizes[time_dim],
        min_observations=min_observations,
        forecast_horizon=forecast_horizon,
        origins=origins,
    )


def _compute_lfo(lfo_inputs, wrapper, method, k_threshold=None):
    """Compute LFO-CV elpd values at every forecast origin.

    For each origin ``i`` the model is scored on the joint predictive density of the block
    ``y[i:i + forecast_horizon]``, refitting at every origin (``method="exact"``) or carrying
    the posterior forward with Pareto-smoothed importance sampling (``method="approx"``).

    Parameters
    ----------
    lfo_inputs : LFOInputs
        Prepared inputs from ``_prepare_lfo_inputs``.
    wrapper : SamplingWrapper
        Wrapper instance handling model refitting.
    method : str
        Either ``"exact"`` or ``"approx"``.
    k_threshold : float, optional
        Pareto k threshold above which the model is refit, only used by ``"approx"``.

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
        - pareto_k: Per-origin Pareto k values for the approximate method, None for the
          exact method. NaN at the first origin. At a refit origin it holds the value that
          triggered the refit
    """
    ll_full = lfo_inputs.log_likelihood
    sample_dims = lfo_inputs.sample_dims
    n_samples = lfo_inputs.n_samples
    time_dim = lfo_inputs.time_dim
    horizon = lfo_inputs.forecast_horizon

    origins = lfo_inputs.origins
    elpds = np.empty(len(origins))
    lpds = np.empty(len(origins))
    pareto_ks = np.full(len(origins), np.nan)
    refits = []

    for origin in _forecast_origins(lfo_inputs, wrapper, method, k_threshold):
        pos, cutoff = origin.pos, origin.cutoff
        block = origin.log_lik.isel({time_dim: slice(origin.offset, origin.offset + horizon)})
        block = block.sum(time_dim)

        if origin.log_weights is None:
            elpds[pos] = logsumexp(block, dims=origin.sample_dims, b=1 / origin.n_samples)
        else:
            weighted = logsumexp(origin.log_weights + block, dims=origin.sample_dims)
            elpds[pos] = weighted - logsumexp(origin.log_weights, dims=origin.sample_dims)

        ll_block = ll_full.isel({time_dim: slice(cutoff, cutoff + horizon)}).sum(time_dim)
        lpds[pos] = logsumexp(ll_block, dims=sample_dims, b=1 / n_samples)
        pareto_ks[pos] = origin.pareto_k
        if origin.refit:
            refits.append(cutoff)

    refits = np.array(refits, dtype=int)
    return _assemble_results(
        lfo_inputs, origins, elpds, lpds, refits, pareto_ks if method == "approx" else None
    )


def _forecast_origins(lfo_inputs, wrapper, method, k_threshold=None):
    """Walk the forecast origins with the fit and importance weights that apply at each one.

    With ``method="exact"`` the model is refit at every origin and the weights are uniform.
    With ``method="approx"`` the fit on ``y[:min_observations]`` is carried forward with
    importance weights over the observations added since the last refit, and the model is
    only refit when the Pareto :math:`k` of those weights exceeds ``k_threshold``.

    Parameters
    ----------
    lfo_inputs : LFOInputs
        Prepared inputs from ``_prepare_lfo_inputs``.
    wrapper : SamplingWrapper
        Wrapper instance handling model refitting.
    method : str
        Either ``"exact"`` or ``"approx"``.
    k_threshold : float, optional
        Pareto k threshold above which the model is refit, only used by ``"approx"``.

    Returns
    -------
    generator of LFOOrigin
        One namedtuple per forecast origin, produced lazily so that each refit happens
        when the origin is reached. Each contains:

        - pos: Position of the origin within ``lfo_inputs.origins``
        - cutoff: Number of observations the current fit conditions on through
          importance weighting, that is, the forecast origin
        - offset: Position of the origin within the fit's ``log_lik`` time dimension
        - log_lik: Log likelihood of the observations from the last refit onward
        - sample_dims: Sample dimensions of ``log_lik``
        - n_samples: Number of posterior draws in the fit
        - idata: Inference data of the fit
        - log_weights: Smoothed log importance weights, None when they are uniform
        - pareto_k: Pareto k of the importance ratios since the last refit. At a refit
          origin it holds the value that triggered the refit. NaN at the first origin and
          for the exact method
        - refit: Whether the model was refit at this origin. The initial fit at the first
          origin only counts as a refit for the exact method
    """
    time_dim = lfo_inputs.time_dim
    origins = lfo_inputs.origins
    last_refit = origins[0]
    fit = _refit_loglik(lfo_inputs, wrapper, last_refit)
    r_eff = _get_r_eff(fit.idata, fit.n_samples) if method == "approx" else None

    for pos, cutoff in enumerate(origins):
        offset = cutoff - last_refit
        log_weights, pareto_k, refit = None, np.nan, method == "exact"
        if method == "exact" and offset > 0:
            fit = _refit_loglik(lfo_inputs, wrapper, cutoff)
            last_refit, offset = cutoff, 0
        elif offset > 0:
            log_ratios = fit.log_lik.isel({time_dim: slice(0, offset)}).sum(time_dim)
            log_weights, pareto_k = _psis_lfo_weights(log_ratios, fit.sample_dims, r_eff)
            if pareto_k > k_threshold:
                fit = _refit_loglik(lfo_inputs, wrapper, cutoff)
                r_eff = _get_r_eff(fit.idata, fit.n_samples)
                last_refit, offset = cutoff, 0
                log_weights, refit = None, True

        yield LFOOrigin(
            pos=pos,
            cutoff=cutoff,
            offset=offset,
            log_lik=fit.log_lik,
            sample_dims=fit.sample_dims,
            n_samples=fit.n_samples,
            idata=fit.idata,
            log_weights=log_weights,
            pareto_k=pareto_k,
            refit=refit,
        )


def _psis_lfo_weights(log_ratios, sample_dims, r_eff):
    """Compute PSIS-LFO weights and flag invalid approximations for exact refitting.

    It returns ``(None, np.inf)`` if the input log ratios are invalid, if PSIS raises
    ``ValueError``, or if PSIS returns invalid weights or Pareto k. The approximate LFO loop
    treats the infinite Pareto k as exceeding ``k_threshold`` and refits the model at the
    current forecast origin.
    """
    ratio_values = np.asarray(log_ratios)

    if (
        np.any(np.isnan(ratio_values))
        or np.any(np.isposinf(ratio_values))
        or np.all(np.isneginf(ratio_values))
    ):
        return None, np.inf

    try:
        log_weights, pareto_k = (-log_ratios).azstats.psislw(dim=sample_dims, r_eff=r_eff)
    except ValueError:
        return None, np.inf

    pareto_k = float(np.asarray(pareto_k))
    weight_values = np.asarray(log_weights)
    if (
        np.isnan(pareto_k)
        or np.isposinf(pareto_k)
        or np.any(np.isnan(weight_values))
        or np.any(np.isposinf(weight_values))
    ):
        return None, np.inf

    return log_weights, pareto_k


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
    LFOFit
        A namedtuple containing:

        - log_lik: Log likelihood of observations ``cutoff`` onward, evaluated at the refit
          posterior draws
        - sample_dims: Dimensions of ``log_lik`` other than the time dimension
        - n_samples: Number of posterior draws in the refit
        - idata: Inference data of the refit returned by the wrapper
    """
    exclude_idx = np.arange(cutoff, lfo_inputs.n_time_points)
    train_data, excluded_data = wrapper.sel_observations(exclude_idx)
    idata = wrapper.get_inference_data(wrapper.sample(train_data))

    # Approximate LFO reuses this fit at later origins, so it needs the likelihood of every
    # remaining observation, including those used to form subsequent importance ratios.
    log_lik = wrapper.log_likelihood__i(excluded_data, idata)
    time_dim = lfo_inputs.time_dim
    if log_lik.sizes.get(time_dim) != len(exclude_idx):
        raise ValueError(
            f"log_likelihood__i must return one value per excluded observation. Expected "
            f"size {len(exclude_idx)} along '{time_dim}', got "
            f"{log_lik.sizes.get(time_dim)} for cutoff {cutoff}."
        )
    if {"chain", "draw"}.issubset(log_lik.dims):
        log_lik = log_lik.transpose("chain", "draw", ...)
    sample_dims = [dim for dim in log_lik.dims if dim != time_dim]
    n_samples = np.prod([log_lik.sizes[dim] for dim in sample_dims])
    return LFOFit(log_lik=log_lik, sample_dims=sample_dims, n_samples=n_samples, idata=idata)


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


def _warn_lfo_refits(method, n_refits, n_data_points):
    """Warn when the PSIS approximation triggered refits at more than half the origins."""
    if method != "approx" or n_refits <= n_data_points / 2:
        return False
    warnings.warn(
        f"LFO-CV triggered {n_refits} refits out of {n_data_points} forecast "
        "origins. The importance sampling approximation may be unreliable. "
        "Consider method='exact'.",
        UserWarning,
    )
    return True
