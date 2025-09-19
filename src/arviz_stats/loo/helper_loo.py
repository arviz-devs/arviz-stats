"""Helper functions for PSIS-LOO-CV."""

import warnings
from collections import namedtuple
from collections.abc import Mapping
from copy import deepcopy

import numpy as np
import xarray as xr
from arviz_base import convert_to_datatree, extract, ndarray_to_dataarray, rcParams
from xarray_einstats.stats import logsumexp

from arviz_stats.utils import ELPDData, get_log_likelihood

__all__ = [
    "_shift",
    "_shift_and_scale",
    "_shift_and_cov",
    "_get_log_likelihood_i",
    "_get_weights_and_k_i",
    "_compute_loo_approximation",
    "_diff_srs_estimator",
    "_srs_estimator",
    "_compute_loo_results",
    "_generate_subsample_indices",
    "_get_r_eff",
    "_prepare_loo_inputs",
    "_extract_loo_data",
    "_check_log_density",
    "_check_log_jacobian",
    "_warn_pareto_k",
    "_warn_pointwise_loo",
    "_prepare_subsample",
    "_prepare_update_subsample",
    "_select_obs_by_indices",
    "_select_obs_by_coords",
    "_prepare_full_arrays",
    "_align_data_to_obs",
    "_validate_crps_input",
    "_log_lik_i",
    "_validate_sample_dims",
    "_get_sample_coords",
]

LooInputs = namedtuple(
    "LooInputs",
    ["log_likelihood", "var_name", "sample_dims", "obs_dims", "n_samples", "n_data_points"],
)

SubsampleData = namedtuple(
    "SubsampleData",
    ["log_likelihood_sample", "lpd_approx_sample", "lpd_approx_all", "indices", "subsample_size"],
)
UpdateSubsampleData = namedtuple(
    "UpdateSubsampleData",
    [
        "log_likelihood_new",
        "lpd_approx_all",
        "old_elpd_i",
        "old_pareto_k",
        "new_indices",
        "old_indices",
        "concat_dim",
        "combined_size",
    ],
)

ShiftResult = namedtuple("ShiftResult", ["upars", "shift"])
ShiftAndScaleResult = namedtuple("ShiftAndScaleResult", ["upars", "shift", "scaling"])
ShiftAndCovResult = namedtuple("ShiftAndCovResult", ["upars", "shift", "mapping"])


def _compute_loo_results(
    log_likelihood,
    var_name,
    sample_dims,
    n_samples,
    n_data_points,
    pointwise=None,
    log_weights=None,
    pareto_k=None,
    reff=None,
    log_p=None,
    log_q=None,
    approx_posterior=False,
    return_pointwise=False,
    log_jacobian=None,
):
    """Compute PSIS-LOO-CV results."""
    if isinstance(log_likelihood, xr.Dataset):
        if var_name is None:
            raise ValueError("var_name must be specified when log_likelihood is a Dataset")
        if var_name not in log_likelihood:
            raise ValueError(f"Variable '{var_name}' not found in log_likelihood Dataset")
        log_likelihood_da = log_likelihood[var_name]
    else:
        log_likelihood_da = log_likelihood

    obs_dims = [dim for dim in log_likelihood_da.dims if dim not in sample_dims]

    if log_p is not None and log_q is not None:
        from arviz_stats.loo.loo_approximate_posterior import loo_approximate_posterior

        data = xr.DataTree()
        data["log_likelihood"] = log_likelihood

        loo_results = loo_approximate_posterior(
            data=data, log_p=log_p, log_q=log_q, pointwise=True, var_name=var_name
        )

        jacobian_da = _check_log_jacobian(log_jacobian, obs_dims)

        if jacobian_da is not None:
            loo_results.elpd_i = loo_results.elpd_i + jacobian_da
            loo_results.elpd = loo_results.elpd_i.sum().item()

        if return_pointwise:
            return loo_results.elpd_i, loo_results.pareto_k, True
        return loo_results

    if log_weights is not None:
        if isinstance(log_weights, xr.Dataset):
            if var_name is None:
                raise ValueError("var_name must be specified when log_weights is a Dataset")
            log_weights = log_weights[var_name]
        for dim in sample_dims:
            if dim not in log_weights.dims:
                raise ValueError(f"log_weights must have sample dimension '{dim}'")

    if pareto_k is not None:
        if isinstance(pareto_k, xr.Dataset):
            if var_name is None:
                raise ValueError("var_name must be specified when pareto_k is a Dataset")
            pareto_k = pareto_k[var_name]
        if set(pareto_k.dims) != set(obs_dims):
            raise ValueError(
                f"pareto_k dimensions {list(pareto_k.dims)} must match "
                f"observation dimensions {obs_dims}"
            )
        for dim in pareto_k.dims:
            if pareto_k.sizes[dim] != log_likelihood_da.sizes[dim]:
                raise ValueError(
                    f"pareto_k size for dimension '{dim}' ({pareto_k.sizes[dim]}) "
                    f"must match log_likelihood size ({log_likelihood_da.sizes[dim]})"
                )

    if log_weights is None or pareto_k is None:
        log_weights, pareto_k = log_likelihood_da.azstats.psislw(r_eff=reff, dim=sample_dims)

    warn_mg, good_k = _warn_pareto_k(pareto_k, n_samples)

    log_weights_sum = log_weights + log_likelihood_da
    elpd_i = logsumexp(log_weights_sum, dims=sample_dims)

    jacobian_da = _check_log_jacobian(log_jacobian, obs_dims)
    if jacobian_da is not None:
        elpd_i = elpd_i + jacobian_da

    elpd = elpd_i.sum().item()

    lppd_da = logsumexp(log_likelihood_da, b=1 / n_samples, dims=sample_dims)
    if jacobian_da is not None:
        lppd_da = lppd_da + jacobian_da
    lppd = lppd_da.sum().item()

    p_loo = lppd - elpd

    if return_pointwise:
        return elpd_i, pareto_k, approx_posterior

    elpd_se = (n_data_points * np.var(elpd_i.values)) ** 0.5

    pointwise = rcParams["stats.ic_pointwise"] if pointwise is None else pointwise
    if pointwise:
        _warn_pointwise_loo(elpd, elpd_i.values)

    return ELPDData(
        "loo",
        elpd,
        elpd_se,
        p_loo,
        n_samples,
        n_data_points,
        "log",
        warn_mg,
        good_k,
        elpd_i if pointwise else None,
        pareto_k if pointwise else None,
        approx_posterior=approx_posterior,
        log_weights=log_weights,
    )


def _prepare_loo_inputs(data, var_name, thin_factor=None):
    """Prepare inputs for PSIS-LOO-CV."""
    data = convert_to_datatree(data)

    log_likelihood = get_log_likelihood(data, var_name=var_name)
    if var_name is None and log_likelihood.name is not None:
        var_name = log_likelihood.name

    if thin_factor is not None:
        # Avoid circular import
        from arviz_stats.manipulation import thin

        log_likelihood = thin(log_likelihood, factor=thin_factor)

    sample_dims = ["chain", "draw"]
    obs_dims = [dim for dim in log_likelihood.dims if dim not in sample_dims]
    n_samples = log_likelihood.chain.size * log_likelihood.draw.size
    n_data_points = np.prod(
        [log_likelihood[dim].size for dim in log_likelihood.dims if dim not in sample_dims]
    )
    return LooInputs(
        log_likelihood,
        var_name,
        sample_dims,
        obs_dims,
        n_samples,
        n_data_points,
    )


def _log_lik_i(i, data, var_name, log_lik_fn):
    """Construct single-observation log-likelihood from a user function."""
    data = convert_to_datatree(data)

    if log_lik_fn is None:
        loo_inputs = _prepare_loo_inputs(data, var_name)
        log_lik_i = _get_log_likelihood_i(loo_inputs.log_likelihood, i, loo_inputs.obs_dims)
        return log_lik_i, loo_inputs.sample_dims, loo_inputs.obs_dims, loo_inputs.n_samples

    if not callable(log_lik_fn):
        raise TypeError("log_lik_fn must be a callable function")
    if not hasattr(data, "observed_data"):
        raise ValueError(
            "Must be able to extract an observed_data group from data when using log_lik_fn."
        )

    try:
        loo_inputs = _prepare_loo_inputs(data, var_name)
        obs_var = loo_inputs.var_name
        sample_dims = loo_inputs.sample_dims
        obs_dims = loo_inputs.obs_dims
    except TypeError as exc:
        obs_vars = list(data.observed_data.data_vars)
        if var_name is None:
            if len(obs_vars) != 1:
                raise ValueError(
                    "Multiple observed variables found; please specify var_name explicitly."
                ) from exc
            obs_var = obs_vars[0]
        else:
            if var_name not in data.observed_data:
                raise ValueError(f"Variable '{var_name}' not found in observed_data") from exc
            obs_var = var_name

        sample_dims = ["chain", "draw"]
        obs_dims = [d for d in data.observed_data[obs_var].dims if d not in sample_dims]
        loo_inputs = None

    observed_i = _get_log_likelihood_i(data.observed_data[obs_var], i, obs_dims)
    data_for_fn = _align_data_to_obs(data, observed_i)

    try:
        log_lik_i = log_lik_fn(observed_i, data_for_fn)
    except Exception as e:
        coord_str = (
            ", ".join(
                f"{dim}={observed_i.coords[dim].values}"
                for dim in obs_dims
                if dim in observed_i.coords
            )
            or f"index {i}"
        )
        raise RuntimeError(f"Error calling log_lik_fn for observation at {coord_str}: {e}") from e

    if not isinstance(log_lik_i, xr.DataArray):
        log_lik_array = np.asarray(log_lik_i)
        coords = _get_sample_coords(sample_dims, loo_inputs, data_for_fn)

        log_lik_i = ndarray_to_dataarray(
            log_lik_array,
            obs_var or observed_i.name or "log_likelihood",
            sample_dims=sample_dims,
            dims=[],
            coords=coords,
        )

    obs_in_result = [d for d in obs_dims if d in log_lik_i.dims]
    if obs_in_result and any(log_lik_i.sizes[d] != 1 for d in obs_in_result):
        log_lik_i = _get_log_likelihood_i(log_lik_i, i, obs_dims)
        obs_in_result = [d for d in obs_dims if d in log_lik_i.dims]
    if obs_in_result:
        log_lik_i = log_lik_i.squeeze(dim=obs_in_result, drop=True)

    try:
        log_lik_i = _validate_sample_dims(
            log_lik_i,
            sample_dims=tuple(sample_dims),
            ref_sizes=loo_inputs.log_likelihood if loo_inputs else None,
            obs_dims=obs_dims,
        )
    except ValueError as e:
        raise ValueError(
            f"log_lik_fn must return DataArray with dimensions {tuple(sample_dims)}, "
            f"got {tuple(log_lik_i.dims)}"
        ) from e

    n_samples = (
        loo_inputs.n_samples
        if loo_inputs
        else int(log_lik_i.sizes["chain"] * log_lik_i.sizes["draw"])
    )

    return log_lik_i, sample_dims, obs_dims, n_samples


def _extract_loo_data(loo_orig):
    """Extract pointwise DataArrays from original PSIS-LOO-CV object."""
    elpd_values = loo_orig.elpd_i
    pareto_values = loo_orig.pareto_k

    valid_mask = ~np.isnan(elpd_values.values.flatten())
    valid_indices = np.where(valid_mask)[0]

    sample_dims = ["chain", "draw"]
    obs_dims = [dim for dim in elpd_values.dims if dim not in sample_dims]
    stacked_obs_dim = "__obs__"

    if len(obs_dims) == 1:
        obs_dim = obs_dims[0]
        extracted_elpd = elpd_values.isel({obs_dim: valid_indices})
    else:
        is_already_stacked = stacked_obs_dim in elpd_values.dims
        stacked_elpd = (
            elpd_values if is_already_stacked else elpd_values.stack({stacked_obs_dim: obs_dims})
        )
        extracted_elpd = stacked_elpd.isel({stacked_obs_dim: valid_indices})

    if pareto_values is not None:
        if len(obs_dims) == 1:
            obs_dim = obs_dims[0]
            extracted_pareto = pareto_values.isel({obs_dim: valid_indices})
        else:
            is_already_stacked = stacked_obs_dim in pareto_values.dims
            stacked_pareto = (
                pareto_values
                if is_already_stacked
                else pareto_values.stack({stacked_obs_dim: obs_dims})
            )
            extracted_pareto = stacked_pareto.isel({stacked_obs_dim: valid_indices})
    else:
        extracted_pareto = xr.full_like(extracted_elpd, np.nan)
    return extracted_elpd, extracted_pareto


def _shift(upars, lwi):
    """Shift a DataArray of parameters to their weighted mean."""
    sample_dims = ["chain", "draw"]
    param_dim = [dim for dim in upars.dims if dim not in sample_dims][0]
    upars_props = _get_upars_info(upars, param_dim)

    upars_stacked = upars.stack(__sample__=sample_dims)
    lwi_stacked = lwi.stack(__sample__=sample_dims)

    upars_values = upars_stacked.transpose("__sample__", param_dim).data
    lwi_values = lwi_stacked.transpose("__sample__").data

    mean_original = np.mean(upars_values, axis=0)
    weights = np.exp(lwi_values)
    mean_weighted = np.sum(weights[:, None] * upars_values, axis=0)

    shift_vec = mean_weighted - mean_original
    upars_new_values = upars_values + shift_vec[None, :]

    upars_new_da = _reconstruct_upars(upars_new_values, upars_props)

    return ShiftResult(upars=upars_new_da, shift=shift_vec)


def _shift_and_scale(upars, lwi):
    """Shift parameters to weighted mean and scale marginal variances."""
    sample_dims = ["chain", "draw"]
    param_dim = [dim for dim in upars.dims if dim not in sample_dims][0]
    upars_props = _get_upars_info(upars, param_dim)

    upars_stacked = upars.stack(__sample__=sample_dims)
    lwi_stacked = lwi.stack(__sample__=sample_dims)

    upars_values = upars_stacked.transpose("__sample__", param_dim).data
    lwi_values = lwi_stacked.transpose("__sample__").data

    samples = upars_values.shape[0]
    mean_original = np.mean(upars_values, axis=0)
    weights = np.exp(lwi_values)

    mean_weighted = np.sum(weights[:, None] * upars_values, axis=0)
    shift_vec = mean_weighted - mean_original

    weighted_second_moment = np.sum(weights[:, None] * upars_values**2, axis=0)
    mii = weighted_second_moment - mean_weighted**2
    mii = mii * samples / (samples - 1)  # Bessel's correction

    var_original = np.var(upars_values, axis=0, ddof=1)

    scaling_vec = np.ones_like(mean_original)
    valid_mask = (var_original > 1e-9) & (mii > 1e-9)
    scaling_vec[valid_mask] = np.sqrt(mii[valid_mask] / var_original[valid_mask])

    upars_new_values = upars_values - mean_original[None, :]
    upars_new_values = upars_new_values * scaling_vec[None, :]
    upars_new_values = upars_new_values + mean_weighted[None, :]

    upars_new_da = _reconstruct_upars(upars_new_values, upars_props)

    return ShiftAndScaleResult(upars=upars_new_da, shift=shift_vec, scaling=scaling_vec)


def _shift_and_cov(upars, lwi):
    """Shift parameters and scale covariance to match weighted covariance."""
    sample_dims = ["chain", "draw"]
    param_dim = [dim for dim in upars.dims if dim not in sample_dims][0]
    upars_props = _get_upars_info(upars, param_dim)

    upars_stacked = upars.stack(__sample__=sample_dims)
    lwi_stacked = lwi.stack(__sample__=sample_dims)

    upars_values = upars_stacked.transpose("__sample__", param_dim).data
    lwi_values = lwi_stacked.transpose("__sample__").data

    mean_original = np.mean(upars_values, axis=0)
    weights = np.exp(lwi_values)

    mean_weighted = np.sum(weights[:, None] * upars_values, axis=0)
    shift_vec = mean_weighted - mean_original

    cov_original = np.cov(upars_values, rowvar=False, ddof=1)
    cov_weighted = np.cov(upars_values, rowvar=False, aweights=weights, ddof=0)

    mapping_mat = np.eye(upars_values.shape[1])
    try:
        min_eig_orig = np.min(np.linalg.eigvalsh(cov_original))
        min_eig_weighted = np.min(np.linalg.eigvalsh(cov_weighted))
        jitter = 1e-6

        if min_eig_orig <= 0:
            cov_original += np.eye(cov_original.shape[0]) * (jitter - min_eig_orig)
        if min_eig_weighted <= 0:
            cov_weighted += np.eye(cov_weighted.shape[0]) * (jitter - min_eig_weighted)

        chol_weighted = np.linalg.cholesky(cov_weighted)
        chol_original = np.linalg.cholesky(cov_original)

        mapping_mat = chol_weighted.T @ np.linalg.inv(chol_original.T)

    except np.linalg.LinAlgError as e:
        warnings.warn(
            f"Cholesky decomposition failed: {e}. Using mean-shift only with identity mapping. "
            "Check Pareto k diagnostics and model specification for highly "
            "influential observations.",
            UserWarning,
            stacklevel=2,
        )

    upars_new_values = upars_values - mean_original[None, :]
    upars_new_values = upars_new_values @ mapping_mat.T
    upars_new_values = upars_new_values + mean_weighted[None, :]

    upars_new_da = _reconstruct_upars(upars_new_values, upars_props)

    return ShiftAndCovResult(upars=upars_new_da, shift=shift_vec, mapping=mapping_mat)


def _get_log_likelihood_i(log_likelihood, i, obs_dims):
    """Extract the log-likelihood for one observation `i`."""
    if not obs_dims:
        raise ValueError("log_likelihood must have observation dimensions.")

    # flattened integer positional selection
    if isinstance(i, int | np.integer):
        if len(obs_dims) == 1:
            dim = obs_dims[0]
            idx = int(i)
            return log_likelihood.isel({dim: slice(idx, idx + 1)}, drop=False)

        # multi-dim: stack, isel one, then unstack to preserve obs dims
        stacked_dim = "__obs__"
        stacked = log_likelihood.stack({stacked_dim: obs_dims})
        idx = int(i)
        selected = stacked.isel({stacked_dim: slice(idx, idx + 1)}, drop=False)
        return selected.unstack(stacked_dim)

    # mapping of labels for all obs dims
    if isinstance(i, Mapping):
        if set(i.keys()) != set(obs_dims):
            raise ValueError(f"Provide selections for all observation dims: {tuple(obs_dims)}")

        da = log_likelihood.sel(i, drop=False)

        remaining_obs_dims = [d for d in obs_dims if d in da.dims]
        if any(da.sizes[d] != 1 for d in remaining_obs_dims):
            raise ValueError("Selection must reduce each observation dimension to length 1.")
        return da

    # single scalar label when there is exactly one obs dim
    if len(obs_dims) == 1:
        obs_dim = obs_dims[0]
        da = log_likelihood.sel({obs_dim: i}, drop=False)
        if obs_dim in da.sizes and da.sizes[obs_dim] != 1:
            raise ValueError("Selection must select exactly one element.")
        return da

    raise TypeError(
        "i must be either a flattened integer index, a mapping of {obs_dim: coord_value} "
        "for all observation dims, or a single scalar label when there is exactly one "
        "observation dimension."
    )


def _get_weights_and_k_i(
    log_weights, pareto_k, i, obs_dims, sample_dims, data, n_samples, reff, log_lik_i, var_name
):
    """Get log weights and Pareto k for a specific observation."""
    if (log_weights is None) != (pareto_k is None):
        raise ValueError(
            "Both log_weights and pareto_k must be provided together or both must be None. "
            "Only one was provided."
        )

    if log_weights is None and pareto_k is None:
        if reff is None:
            reff = _get_r_eff(data, n_samples)
        return log_lik_i.azstats.psislw(r_eff=reff, dim=sample_dims)

    lw = log_weights
    if isinstance(lw, xr.Dataset):
        if var_name is None:
            raise ValueError("var_name must be specified when log_weights is a Dataset")
        lw = lw[var_name]

    if isinstance(lw, xr.DataArray):
        lw_dims = set(lw.dims)
        if any(d in lw_dims for d in obs_dims):
            lw = _get_log_likelihood_i(lw, i, obs_dims)
            squeeze_dims = [d for d in obs_dims if d in lw.dims]

            if squeeze_dims:
                lw = lw.squeeze(squeeze_dims, drop=True)
        for dim in sample_dims:
            if dim not in lw.dims:
                raise ValueError(f"log_weights must have sample dimension '{dim}'")

    pk = pareto_k
    if isinstance(pk, xr.Dataset):
        if var_name is None:
            raise ValueError("var_name must be specified when pareto_k is a Dataset")
        pk = pk[var_name]

    if isinstance(pk, xr.DataArray):
        pk_dims = set(pk.dims)
        if any(d in pk_dims for d in obs_dims):
            pk_i_da = _get_log_likelihood_i(pk, i, obs_dims)
            pk = pk_i_da.squeeze().item()

    return lw, pk


def _prepare_subsample(
    data,
    log_likelihood_da,
    var_name,
    observations,
    seed,
    method,
    log_lik_fn,
    param_names,
    log,
    obs_dims,
    sample_dims,
    n_data_points,
    n_samples,
    thin_factor=None,
):
    """Prepare inputs for PSIS-LOO-CV with sub-sampling."""
    indices, subsample_size = _generate_subsample_indices(n_data_points, observations, seed)

    if thin_factor is not None and log_lik_fn is not None:
        from arviz_stats.manipulation import thin

        data = data.copy(deep=True)
        if hasattr(data, "posterior"):
            thinned_posterior = thin(data.posterior, factor=thin_factor)
            data["posterior"] = thinned_posterior

    if method == "lpd":
        if log_lik_fn is None:
            lpd_approx_all = logsumexp(log_likelihood_da, dims=sample_dims, b=1 / n_samples)
        else:
            lpd_approx_all = _compute_loo_approximation(
                data=data,
                var_name=var_name,
                log_lik_fn=log_lik_fn,
                param_names=param_names,
                method="lpd",
                log=log,
            )
    else:  # method == "plpd"
        lpd_approx_all = _compute_loo_approximation(
            data=data,
            var_name=var_name,
            log_lik_fn=log_lik_fn,
            param_names=param_names,
            method="plpd",
            log=log,
        )

    log_likelihood_sample = _select_obs_by_indices(log_likelihood_da, indices, obs_dims, "__obs__")
    lpd_approx_sample = _select_obs_by_indices(lpd_approx_all, indices, obs_dims, "__obs__")

    return SubsampleData(
        log_likelihood_sample,
        lpd_approx_sample,
        lpd_approx_all,
        indices,
        subsample_size,
    )


def _prepare_update_subsample(
    loo_orig,
    data,
    observations,
    var_name,
    seed,
    method,
    log_lik_fn,
    param_names,
    log,
    thin_factor=None,
):
    """Prepare inputs for updating PSIS-LOO-CV with additional observations."""
    loo_inputs = _prepare_loo_inputs(data, var_name, thin_factor)
    old_elpd_i_da, old_pareto_k_da = _extract_loo_data(loo_orig)

    log_likelihood = loo_inputs.log_likelihood
    concat_dim = old_elpd_i_da.dims[0]
    old_indices = np.where(~np.isnan(loo_orig.elpd_i.values.flatten()))[0]

    if isinstance(observations, int):
        # Filter out existing indices before generating new ones
        available_indices = np.setdiff1d(np.arange(loo_inputs.n_data_points), old_indices)
        if observations > len(available_indices):
            raise ValueError(
                f"Cannot add {observations} observations when only {len(available_indices)} "
                "are available."
            )
        temp_indices, _ = _generate_subsample_indices(len(available_indices), observations, seed)
        new_indices = available_indices[temp_indices]
    else:
        new_indices = np.asarray(observations, dtype=int)
        if np.any(new_indices < 0) or np.any(new_indices >= loo_inputs.n_data_points):
            raise ValueError("New indices contain out-of-bounds values.")
        # Check for overlap with old indices
        overlap = np.intersect1d(new_indices, old_indices)
        if len(overlap) > 0:
            raise ValueError(f"New indices {overlap} overlap with existing indices.")

    if thin_factor is not None and log_lik_fn is not None:
        from arviz_stats.manipulation import thin

        data = data.copy(deep=True)
        if hasattr(data, "posterior"):
            thinned_posterior = thin(data.posterior, factor=thin_factor)
            data["posterior"] = thinned_posterior

    if method == "lpd":
        if log_lik_fn is None:
            lpd_approx_all = logsumexp(
                log_likelihood, dims=loo_inputs.sample_dims, b=1 / loo_inputs.n_samples
            )
        else:
            lpd_approx_all = _compute_loo_approximation(
                data=data,
                var_name=loo_inputs.var_name,
                log_lik_fn=log_lik_fn,
                param_names=param_names,
                method="lpd",
                log=log,
            )
    else:  # method == "plpd"
        lpd_approx_all = _compute_loo_approximation(
            data=data,
            var_name=loo_inputs.var_name,
            log_lik_fn=log_lik_fn,
            param_names=param_names,
            method="plpd",
            log=log,
        )

    log_likelihood_new_da = _select_obs_by_indices(
        log_likelihood, new_indices, loo_inputs.obs_dims, "__obs__"
    )

    log_likelihood_new_ds = xr.Dataset({loo_inputs.var_name: log_likelihood_new_da})
    combined_size = len(old_indices) + len(new_indices)

    return UpdateSubsampleData(
        log_likelihood_new_ds,
        lpd_approx_all,
        old_elpd_i_da,
        old_pareto_k_da,
        new_indices,
        old_indices,
        concat_dim,
        combined_size,
    )


def _compute_loo_approximation(
    data,
    var_name,
    log_lik_fn=None,
    param_names=None,
    method="lpd",
    log=True,
):
    """Compute LOO approximation with LPD or PLPD method."""
    if not hasattr(data, "observed_data"):
        raise ValueError("No observed_data group found in the data")
    if var_name not in data.observed_data:
        raise ValueError(f"Variable {var_name} not found in observed_data")
    if method not in ["lpd", "plpd"]:
        raise ValueError(f"Unknown method: {method}. Must be 'lpd' or 'plpd'")

    observed = data.observed_data[var_name]
    sample_dims = ["chain", "draw"]

    # early return for simple LPD case
    if method == "lpd" and log_lik_fn is None:
        log_likelihood = get_log_likelihood(data, var_name=var_name)
        n_samples = log_likelihood.chain.size * log_likelihood.draw.size
        return logsumexp(log_likelihood, dims=sample_dims, b=1 / n_samples)

    if log_lik_fn is None:
        raise ValueError(f"log_lik_fn required for {method} method")
    if not callable(log_lik_fn):
        raise TypeError("log_lik_fn must be a callable function.")

    posterior = extract(data, group="posterior", combined=False)

    if isinstance(posterior, xr.DataArray):
        if param_names and len(param_names) == 1 and posterior.name != param_names[0]:
            raise ValueError(
                f"Requested parameter '{param_names[0]}' but DataArray has "
                f"name {posterior.name}"
            )
        if param_names and len(param_names) > 1:
            raise ValueError(
                f"Cannot select multiple parameters {param_names} from a single "
                f"DataArray {posterior.name}"
            )
        posterior = xr.Dataset({posterior.name: posterior})
    elif param_names:
        posterior = posterior[param_names]

    data_for_fn = _align_data_to_obs(data, observed)

    if method == "plpd":
        posterior_means = posterior.mean(dim=sample_dims)
        data_for_fn.posterior = xr.Dataset(posterior_means.data_vars)
        available_vars = list(posterior_means.data_vars)
    else:
        data_for_fn.posterior = posterior
        available_vars = list(posterior.data_vars)

    try:
        result = log_lik_fn(observed, data_for_fn)
    except KeyError as e:
        if "missing_param" in str(e) or "No variable named" in str(e):
            raise KeyError(
                f"Variable not found in posterior. Available posterior variables: "
                f"{available_vars}"
            ) from e
        raise
    except Exception as e:
        raise ValueError(f"Error in log_lik_fn: {type(e).__name__}: {str(e)}") from e

    if not isinstance(result, xr.DataArray):
        result = np.asarray(result) if not isinstance(result, np.ndarray) else result

        if method == "lpd":
            expected_shape = (posterior.chain.size, posterior.draw.size) + observed.shape
            if result.shape != expected_shape:
                raise ValueError(
                    f"log_lik_fn returned array with shape {result.shape}, "
                    f"expected {expected_shape}"
                )
            coords = {**posterior.coords, **observed.coords}
        else:  # plpd
            expected_shape = observed.shape
            if result.shape != expected_shape:
                raise ValueError(
                    f"log_lik_fn returned array with shape {result.shape}, "
                    f"expected {expected_shape}"
                )
            coords = observed.coords

        result = ndarray_to_dataarray(
            result,
            var_name,
            sample_dims=sample_dims if method == "lpd" else [],
            dims=list(observed.dims),
            coords=coords,
        )

    if not log:
        result = xr.ufuncs.log(xr.ufuncs.maximum(result, np.finfo(float).tiny))

    # validation
    if method == "lpd":
        expected_dims = set(sample_dims) | set(observed.dims)
        if set(result.dims) != expected_dims:
            raise ValueError(
                f"log_lik_fn must return an object with dims {list(expected_dims)}. "
                f"Got {list(result.dims)}"
            )
        for dim in observed.dims:
            if result.sizes[dim] != observed.sizes[dim]:
                raise ValueError(
                    f"log_lik_fn must return an object with {dim} size {observed.sizes[dim]}. "
                    f"Got {result.sizes[dim]}"
                )
        n_samples = posterior.chain.size * posterior.draw.size
        return logsumexp(result, dims=sample_dims, b=1 / n_samples).rename("lpd")
    # plpd validation
    if set(result.dims) != set(observed.dims) or any(
        result.sizes[dim] != observed.sizes[dim] for dim in observed.dims
    ):
        got_sizes = {dim: result.sizes.get(dim, "missing") for dim in observed.dims}
        exp_sizes = {dim: observed.sizes[dim] for dim in observed.dims}
        raise ValueError(
            f"For method='plpd', log_lik_fn must return an object with dims "
            f"{list(observed.dims)} and matching sizes. Got dims={list(result.dims)}, "
            f"sizes={got_sizes}, expected sizes={exp_sizes}"
        )
    return result.rename("plpd")


def _diff_srs_estimator(
    elpd_loo_i_sample,
    lpd_approx_sample,
    lpd_approx_all,
    n_data_points,
    subsample_size,
):
    """Calculate the difference estimator PSIS-LOO-CV with sub-sampling.

    Parameters
    ----------
    elpd_loo_i_sample : DataArray
        Pointwise ELPD values for the subsample.
    lpd_approx_sample : DataArray
        LPD approximation values for the subsample.
    lpd_approx_all : DataArray
        LPD approximation values for the full dataset.
    n_data_points : int
        Total number of data points (N).
    subsample_size : int
        Number of observations in the subsample (m).

    Returns
    -------
    tuple[float, float, float]
        A tuple containing:

        - elpd_loo_hat: The estimated ELPD using the difference estimator.
        - subsampling_se_val: The standard error due to subsampling uncertainty.
        - se_val: The total standard error (approximation + sampling uncertainty).
    """
    pointwise_diff = elpd_loo_i_sample - lpd_approx_sample
    lpd_approx_sum_all = lpd_approx_all.sum().values.item()
    scaled_mean_pointwise_diff = n_data_points * pointwise_diff.mean().values.item()
    elpd_loo_estimate = lpd_approx_sum_all + scaled_mean_pointwise_diff

    if subsample_size > 1:
        # Subsampling variance
        subsampling_variance = (
            (n_data_points**2)
            * (1 - subsample_size / n_data_points)
            * pointwise_diff.var(ddof=1).values
            / subsample_size
        )
        subsampling_se = np.sqrt(subsampling_variance) if subsampling_variance >= 0 else np.inf

        # Total variance (approximation + sampling)
        lpd_approx_sq_sum_all = (lpd_approx_all**2).sum().values.item()
        mean_sq_diff = ((elpd_loo_i_sample**2) - (lpd_approx_sample**2)).mean().values.item()
        scaled_mean_sq_diff = n_data_points * mean_sq_diff

        if n_data_points > 0:
            total_variance_estimate = (lpd_approx_sq_sum_all + scaled_mean_sq_diff) - (
                1 / n_data_points
            ) * (
                scaled_mean_pointwise_diff**2
                - subsampling_variance
                + 2 * lpd_approx_sum_all * elpd_loo_estimate
                - lpd_approx_sum_all**2
            )
            total_variance_estimate = max(0, total_variance_estimate)
        else:
            total_variance_estimate = np.inf

        total_se = np.sqrt(total_variance_estimate) if total_variance_estimate >= 0 else np.inf
    else:
        subsampling_se = np.inf
        total_se = np.inf
    return elpd_loo_estimate, subsampling_se, total_se


def _srs_estimator(
    y_sample,
    n_data_points,
    subsample_size,
):
    """Calculate the SRS estimator for PSIS-LOO-CV with sub-sampling.

    Parameters
    ----------
    y_sample : DataArray
        Values of the statistic (e.g., p_loo) for the subsample.
    n_data_points : int
        Total number of data points (N).
    subsample_size : int
        Number of observations in the subsample (m).

    Returns
    -------
    tuple[float, float, float]
        A tuple containing:

        - y_hat: The estimated statistic using simple random sampling.
        - var_y_hat: The variance of the estimator (sampling uncertainty).
        - hat_var_y: The estimated variance of the statistic.
    """
    y_sample_mean = y_sample.mean().values.item()
    y_hat = n_data_points * y_sample_mean

    if subsample_size > 1:
        y_sample_var = y_sample.var(ddof=1).values.item()
        var_y_hat = (
            (n_data_points**2)
            * (1 - subsample_size / n_data_points)
            * y_sample_var
            / subsample_size
        )
        hat_var_y = n_data_points * y_sample_var
    else:
        var_y_hat = np.inf
        hat_var_y = np.inf

    return y_hat, var_y_hat, hat_var_y


def _align_group(group, observed, primary_obs_dim):
    """Align all variables in a data group."""
    if primary_obs_dim not in observed.coords:
        return group.copy()

    aligned_vars = {}
    for var_name, var_data in group.data_vars.items():
        if primary_obs_dim in var_data.dims:
            aligned_vars[var_name] = var_data.sel(
                {primary_obs_dim: observed.coords[primary_obs_dim]}
            )
        else:
            aligned_vars[var_name] = var_data
    return xr.Dataset(aligned_vars)


def _align_data_to_obs(data, observed):
    """Align auxiliary data groups to match the subset of observations."""
    obs_dims = [dim for dim in observed.dims if dim not in ["chain", "draw"]]
    if not obs_dims:
        return deepcopy(data)

    primary_obs_dim = obs_dims[0]
    aligned_data = data.copy()

    for group_name in ["constant_data"]:
        if hasattr(data, group_name):
            group = getattr(data, group_name)
            aligned_group = _align_group(group, observed, primary_obs_dim)
            setattr(aligned_data, group_name, aligned_group)

    return aligned_data


def _generate_subsample_indices(n_data_points, observations, seed):
    """Generate subsample indices."""
    if isinstance(observations, int):
        if not 1 <= observations <= n_data_points:
            raise ValueError(
                f"Number of observations must be between 1 and {n_data_points}, "
                f"got {observations}"
            )
        # SRS-WOR
        rng = np.random.default_rng(seed)
        indices = rng.choice(n_data_points, size=observations, replace=False)
        subsample_size = observations
    elif isinstance(observations, np.ndarray):
        if observations.dtype.kind != "i":
            raise TypeError("observations array must contain integers.")
        if n_data_points > 0 and (observations.min() < 0 or observations.max() >= n_data_points):
            raise ValueError(
                f"Observation indices must be between 0 and {n_data_points - 1}, "
                f"got range [{observations.min()}, {observations.max()}]"
            )
        indices = np.unique(observations)
        subsample_size = len(indices)
        if subsample_size == 0:
            raise ValueError("observations array cannot be empty.")
    else:
        raise TypeError("observations must be an integer or a numpy array of integers.")
    return indices, subsample_size


def _get_r_eff(data, n_samples):
    if not hasattr(data, "posterior"):
        raise TypeError("Must be able to extract a posterior group from data.")
    posterior = data.posterior
    n_chains = len(posterior.chain)
    if n_chains == 1:
        reff = 1.0
    else:
        ess_p = posterior.azstats.ess(method="mean")
        # this mean is over all data variables
        reff = np.hstack([ess_p[v].values.flatten() for v in ess_p.data_vars]).mean() / n_samples
    return reff


def _prepare_full_arrays(
    pointwise_values,
    pareto_k_values,
    ref_array,
    indices,
    obs_dims,
    elpd_loo_hat=None,
):
    """Prepare full arrays for pointwise ELPD and Pareto k values."""
    if not obs_dims:
        # If no obs dim, the pointwise results are already the full results
        return pointwise_values, pareto_k_values

    # Only create full arrays and assign if there are obs dims
    elpd_i_full, pareto_k_full = (
        xr.full_like(ref_array, fill_value=np.nan, dtype=np.float64).rename("elpd_i"),
        xr.full_like(ref_array, fill_value=np.nan, dtype=np.float64).rename("pareto_k"),
    )

    if len(obs_dims) > 1:
        obs_shape = [ref_array.sizes[d] for d in obs_dims]
        multi_indices = np.unravel_index(indices, obs_shape)
        indexers = {
            dim: xr.DataArray(idx, dims="__obs__") for dim, idx in zip(obs_dims, multi_indices)
        }
        elpd_i_full.isel(indexers).values[:] = pointwise_values.values
        pareto_k_full.isel(indexers).values[:] = pareto_k_values.values
    else:
        elpd_i_full[{obs_dims[0]: indices}] = pointwise_values.values
        pareto_k_full[{obs_dims[0]: indices}] = pareto_k_values.values

    if elpd_loo_hat is not None:
        _warn_pointwise_loo(elpd_loo_hat, elpd_i_full.values)
    return elpd_i_full, pareto_k_full


def _select_obs_by_indices(data_array, indices, dims, dim_name):
    """Select a sub-sample from a DataArray based on indices."""
    if not dims:
        return data_array

    if len(dims) > 1:
        n_total = np.prod([data_array.sizes[d] for d in dims])
    else:
        n_total = data_array.sizes[dims[0]]

    valid_indices_mask = (indices >= 0) & (indices < n_total)
    valid_indices = indices[valid_indices_mask]

    if len(valid_indices) == 0:
        if len(dims) > 1:
            return data_array.isel({d: [] for d in dims})
        return data_array.isel({dims[0]: []})

    if len(dims) == 1:
        return data_array.isel({dims[0]: valid_indices})

    obs_shape = [data_array.sizes[d] for d in dims]
    multi_indices = np.unravel_index(valid_indices, obs_shape)

    indexers = {dim: xr.DataArray(idx, dims=dim_name) for dim, idx in zip(dims, multi_indices)}

    return data_array.isel(indexers)


def _select_obs_by_coords(data_array, coord_array, dims, dim_name):
    """Select a sub-sample from a DataArray based on coordinate values."""
    if len(dims) > 1:
        stacked_data = data_array.stack({dim_name: dims})
        return stacked_data.sel({dim_name: coord_array[dim_name]})
    return data_array.sel({dims[0]: coord_array[dims[0]]})


def _warn_pareto_k(pareto_k_values, n_samples, suppress=False):
    """Check Pareto k values and issue warnings if necessary."""
    good_k = min(1 - 1 / np.log10(n_samples), 0.7) if n_samples > 1 else 0.7
    warn_mg = False

    if np.any(pareto_k_values > good_k):
        if not suppress:
            warnings.warn(
                f"Estimated shape parameter of Pareto distribution is greater than {good_k:.2f} "
                "for one or more samples. You should consider using a more robust model, this is "
                "because importance sampling is less likely to work well if the marginal posterior "
                "and LOO posterior are very different. This is more likely to happen with a "
                "non-robust model and highly influential observations."
            )
        warn_mg = True
    return warn_mg, good_k


def _warn_pointwise_loo(elpd, elpd_i_values):
    """Check if pointwise LOO values sum to the same as total LOO."""
    if np.equal(elpd, elpd_i_values).all():
        warnings.warn(
            "The point-wise LOO is the same with the sum LOO, please double check "
            "the Observed RV in your model to make sure it returns element-wise logp."
        )


def _check_log_density(log_dens, name, log_likelihood, n_samples, sample_dims):
    """Validate log_p or log_q input for loo_approximate_posterior."""
    if isinstance(log_dens, np.ndarray):
        if log_dens.size != n_samples:
            raise ValueError(
                f"Size of {name} ({log_dens.size}) must match "
                f"the total number of samples in log_likelihood ({n_samples})."
            )
        sample_shape = tuple(log_likelihood[dim].size for dim in sample_dims)
        log_dens_values = log_dens.reshape(sample_shape)
        coords = {dim: log_likelihood[dim] for dim in sample_dims}
        validated_log_dens = xr.DataArray(log_dens_values, dims=sample_dims, coords=coords)

    elif isinstance(log_dens, xr.DataArray):
        validated_log_dens = log_dens
        for dim in sample_dims:
            if dim not in validated_log_dens.dims:
                raise ValueError(f"{name} must have dimension '{dim}'")
            if validated_log_dens[dim].size != log_likelihood[dim].size:
                raise ValueError(
                    f"Size of dimension '{dim}' in {name} ({validated_log_dens[dim].size}) "
                    f"must match the size in log_likelihood ({log_likelihood[dim].size})."
                )
        for dim in sample_dims:
            if dim in validated_log_dens.coords and not np.array_equal(
                validated_log_dens[dim].values, log_likelihood[dim].values
            ):
                warnings.warn(
                    f"Coordinates for dimension '{dim}' in {name} do not match "
                    f"those in log_likelihood. Ensure they correspond to the same samples.",
                    UserWarning,
                )
    else:
        raise TypeError(f"{name} must be a numpy ndarray or xarray DataArray")
    return validated_log_dens


def _check_log_jacobian(log_jacobian, obs_dims):
    """Validate Jacobian adjustment input."""
    if log_jacobian is None:
        return None

    if not isinstance(log_jacobian, xr.DataArray):
        raise TypeError(
            f"log_jacobian must be an xarray.DataArray or None. "
            f"Got type {type(log_jacobian).__name__}."
        )

    if not np.issubdtype(log_jacobian.dtype, np.number):
        raise TypeError(
            f"log_jacobian must contain numeric values, but has dtype {log_jacobian.dtype}"
        )

    if not np.all(np.isfinite(log_jacobian.values)):
        n_inf = np.sum(np.isinf(log_jacobian.values))
        n_nan = np.sum(np.isnan(log_jacobian.values))
        raise ValueError(
            f"log_jacobian must contain only finite values. "
            f"Found {n_nan} NaN and {n_inf} Inf values."
        )

    sample_dims = {"chain", "draw"}
    jacobian_dims = set(log_jacobian.dims)
    if sample_dims.intersection(jacobian_dims):
        raise ValueError(
            f"log_jacobian must not have sample dimensions {sample_dims}. "
            f"Found dimensions: {list(log_jacobian.dims)}"
        )

    obs_dims_set = set(obs_dims) if obs_dims else set()

    if jacobian_dims != obs_dims_set:
        missing_dims = obs_dims_set - jacobian_dims
        extra_dims = jacobian_dims - obs_dims_set
        error_msg = "log_jacobian dimensions must exactly match observation dimensions.\n"

        if missing_dims:
            error_msg += f"  Missing dimensions: {list(missing_dims)}\n"
        if extra_dims:
            error_msg += f"  Extra dimensions: {list(extra_dims)}\n"

        error_msg += f"  Expected dimensions: {obs_dims}\n"
        error_msg += f"  Got dimensions: {list(log_jacobian.dims)}"
        raise ValueError(error_msg)

    return log_jacobian


def _get_upars_info(upars, param_dim):
    """Get original properties from upars DataArray."""
    props = {
        "dims": upars.dims,
        "shape": upars.shape,
        "coords": {
            "chain": upars.coords["chain"],
            "draw": upars.coords["draw"],
        },
    }
    if param_dim in upars.coords:
        props["coords"][param_dim] = upars.coords[param_dim]
    return props


def _reconstruct_upars(upars_new_values, props):
    """Reconstruct upars DataArray from new values."""
    upars_new_values_reshaped = upars_new_values.reshape(props["shape"])
    return xr.DataArray(
        upars_new_values_reshaped,
        dims=props["dims"],
        coords=props["coords"],
    )


def _has_nan_slice(da, dim):
    """Check if DataArray has NaN slices along a dimension."""
    other = tuple(dd for dd in da.dims if dd != dim)
    return bool(da.isnull().all(dim=other).any()) if other else bool(da.isnull().any())


def _validate_crps_input(y_pred, y_obs, log_likelihood, *, sample_dims, obs_dims):
    """Shape and dimension checks."""
    missing_sample = [d for d in sample_dims if d not in y_pred.dims]
    if missing_sample:
        raise ValueError(f"y_pred must include sample dimension '{missing_sample[0]}'")

    missing_obs = [
        d
        for d in obs_dims
        if (d not in y_pred.dims or d not in y_obs.dims or d not in log_likelihood.dims)
    ]
    if missing_obs:
        raise ValueError(f"Missing observation dimension '{missing_obs[0]}' in inputs")

    ypred_obs_aligned, yobs_aligned = xr.align(y_pred, y_obs, join="inner")
    obs_size_mismatch = [
        d
        for d in obs_dims
        if (
            ypred_obs_aligned.sizes[d] != y_pred.sizes[d] or yobs_aligned.sizes[d] != y_obs.sizes[d]
        )
    ]
    if obs_size_mismatch:
        raise ValueError(f"Size mismatch in observation dim '{obs_size_mismatch[0]}'")

    nan_padded_obs = [d for d in obs_dims if _has_nan_slice(y_obs, d)]
    if nan_padded_obs:
        raise ValueError(f"Size mismatch in observation dim '{nan_padded_obs[0]}'")

    ypred_ll_aligned, ll_aligned = xr.align(y_pred, log_likelihood, join="inner")
    dims_to_check = (*sample_dims, *obs_dims)
    ll_size_mismatch = [
        d
        for d in dims_to_check
        if (
            ypred_ll_aligned.sizes[d] != y_pred.sizes[d]
            or ll_aligned.sizes[d] != log_likelihood.sizes[d]
        )
    ]
    if ll_size_mismatch:
        d0 = ll_size_mismatch[0]
        if d0 in sample_dims:
            raise ValueError(f"Size mismatch in sample dimension '{d0}'")
        raise ValueError(f"Size mismatch in observation dim '{d0}'")

    nan_padded_sample = [d for d in sample_dims if _has_nan_slice(log_likelihood, d)]
    if nan_padded_sample:
        raise ValueError(f"Size mismatch in sample dimension '{nan_padded_sample[0]}'")


def _validate_sample_dims(
    data,
    *,
    sample_dims=None,
    ref_sizes=None,
    obs_dims=None,
):
    """Validate sample dimensions."""
    if sample_dims is None:
        sample_dims = ["chain", "draw"]
    if not isinstance(data, xr.DataArray):
        raise TypeError(f"Expected DataArray, got {type(data).__name__}")

    if obs_dims:
        present_obs = [d for d in obs_dims if d in data.dims]
        if present_obs:
            bad = [d for d in present_obs if data.sizes[d] != 1]
            if bad:
                raise ValueError(
                    f"DataArray has observation dimensions with size > 1: {bad}. "
                    "Select a single observation or drop these dimensions before returning."
                )
            data = data.squeeze()

    data_dims = tuple(data.dims)
    if set(data_dims) != set(sample_dims):
        raise ValueError(
            f"DataArray must have exactly sample dimensions {tuple(sample_dims)}, got {data_dims}"
        )

    if ref_sizes is not None:
        sizes = ref_sizes.sizes if hasattr(ref_sizes, "sizes") else ref_sizes
        for d in sample_dims:
            if data.sizes[d] != sizes[d]:
                raise ValueError(
                    f"DataArray has size {data.sizes[d]} for dimension '{d}', expected {sizes[d]}"
                )
    return data


def _get_sample_coords(sample_dims, loo_inputs, data_for_fn):
    """Collect sample dimension coordinates."""
    if loo_inputs is not None:
        source = loo_inputs.log_likelihood
        return {dim: source.coords[dim] for dim in sample_dims if dim in source.coords}

    coords = {}
    posterior = getattr(data_for_fn, "posterior", None)
    if posterior is None:
        return coords

    if isinstance(posterior, xr.Dataset):
        coords.update(
            {dim: posterior.coords[dim] for dim in sample_dims if dim in posterior.coords}
        )
        missing = [dim for dim in sample_dims if dim not in coords]
        if missing:
            for var in posterior.data_vars.values():
                found = [dim for dim in missing if dim in var.coords]
                for dim in found:
                    coords[dim] = var.coords[dim]
                if not found:
                    continue
                missing = [dim for dim in missing if dim not in coords]
                if not missing:
                    break
        return coords

    if isinstance(posterior, xr.DataArray):
        coords.update(
            {dim: posterior.coords[dim] for dim in sample_dims if dim in posterior.coords}
        )
    return coords
