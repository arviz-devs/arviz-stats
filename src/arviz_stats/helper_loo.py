"""Helper functions for PSIS-LOO-CV."""

import warnings

import numpy as np
import xarray as xr
from arviz_base import convert_to_datatree, extract

from arviz_stats.utils import get_log_likelihood_dataset

__all__ = [
    "_get_r_eff",
    "_diff_srs_estimator",
    "_srs_estimator",
    "_generate_subsample_indices",
    "_prepare_loo_inputs",
    "_extract_loo_data",
    "_check_log_density",
    "_warn_pareto_k",
    "_warn_pointwise_loo",
    "_plpd_approx",
    "_select_observation_subsample",
]


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


def _plpd_approx(
    data,
    var_name,
    log_lik_fn,
    param_names=None,
    log=True,
):
    """Compute the Point Log Predictive Density (PLPD) approximation.

    Parameters
    ----------
    data : DataTree or InferenceData
        Input data containing `posterior` and `observed_data` groups.
    var_name : str
        Name of the variable in `observed_data` for which to compute the PLPD approximation.
    log_lik_fn : callable
        A user-provided function that computes the log likelihood. It must accept the
        observed data as its first argument, followed by the *mean* values
        of the posterior parameters specified by `param_names` as subsequent
        positional arguments.
    param_names : list[str], optional
        An ordered list of parameter names from the posterior group whose mean values
        will be passed as positional arguments (after the data point value) to `log_lik_fn`.
        If None, all parameters from the posterior group are used in alphabetical order.
    log : bool, optional
        Whether the log_likelihood_fn returns log-likelihood (True) or likelihood (False).
        Default is True. If False, the output will be log-transformed.

    Returns
    -------
    DataArray
        DataArray containing PLPD values with the same dimensions as observed data.
    """
    if not callable(log_lik_fn):
        raise TypeError("log_lik_fn must be a callable function.")
    if not hasattr(data, "observed_data"):
        raise ValueError("No observed_data group found in the data")
    if var_name not in data.observed_data:
        raise ValueError(f"Variable {var_name} not found in observed_data")

    observed = data.observed_data[var_name]
    posterior = extract(data, group="posterior", combined=True)

    if param_names is None:
        param_keys = sorted(list(posterior.data_vars.keys()))
    else:
        missing_params = [p for p in param_names if p not in posterior.data_vars]
        if missing_params:
            raise ValueError(f"Parameters {missing_params} not found in posterior group.")
        param_keys = param_names

    param_means = [float(posterior[key].mean()) for key in param_keys]
    plpd_values = np.empty_like(observed.values, dtype=float)

    for idx, value in np.ndenumerate(observed.values):
        try:
            result = log_lik_fn(value, *param_means)
            if not log:
                result = np.log(np.maximum(result, np.finfo(float).tiny))

            if not isinstance(result, int | float | np.number):
                raise TypeError(
                    f"log_lik_fn must return a numeric scalar. Got type: {type(result)}"
                )
            plpd_values[idx] = result
        except Exception as e:
            raise RuntimeError(
                f"Error computing log-likelihood with log_lik_fn for data point "
                f"at index {idx} (value: {value}): {e}"
            ) from e

    plpd_da = xr.DataArray(plpd_values, dims=observed.dims, coords=observed.coords, name="plpd")
    return plpd_da


def _diff_srs_estimator(
    elpd_loo_i_sample,
    lpd_approx_sample,
    lpd_approx_all,
    n_data_points,
    subsample_size,
):
    """Calculate the difference estimator for sub-sampling PSIS-LOO-CV.

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
    lpd_approx_sum_all = lpd_approx_all.sum().values
    scaled_mean_pointwise_diff = n_data_points * pointwise_diff.mean().values
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
        lpd_approx_sq_sum_all = (lpd_approx_all**2).sum().values
        mean_sq_diff = ((elpd_loo_i_sample**2) - (lpd_approx_sample**2)).mean().values
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
    y_sample_mean = y_sample.mean().values
    y_hat = n_data_points * y_sample_mean

    if subsample_size > 1:
        y_sample_var = y_sample.var(ddof=1).values
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


def _generate_subsample_indices(n_data_points, observations, seed):
    """Generate subsample indices based on input type."""
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


def _prepare_loo_inputs(data, var_name):
    """Prepare inputs for LOO computation."""
    data = convert_to_datatree(data)

    log_likelihood_ds = get_log_likelihood_dataset(data, var_names=var_name)
    if var_name is None:
        var_name = list(log_likelihood_ds.data_vars.keys())[0]
    log_likelihood = log_likelihood_ds[var_name]

    sample_dims = ["chain", "draw"]
    obs_dims = [dim for dim in log_likelihood.dims if dim not in sample_dims]
    n_samples = log_likelihood.chain.size * log_likelihood.draw.size
    n_data_points = np.prod(
        [log_likelihood[dim].size for dim in log_likelihood.dims if dim not in sample_dims]
    )
    return log_likelihood_ds, var_name, sample_dims, obs_dims, n_samples, n_data_points


def _extract_loo_data(loo_orig):
    """Extract pointwise DataArrays from original LOO object."""
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


def _select_observation_subsample(data_array, indices, obs_dims, dim_name="__obs__"):
    """Select a subsample from a data array based on indices along observation dimensions."""
    if len(obs_dims) > 1:
        stacked_data = data_array.stack({dim_name: obs_dims})
        coord = stacked_data[dim_name]
    else:
        stacked_data = data_array
        coord = stacked_data[obs_dims[0]]

    valid_indices_mask = (indices >= 0) & (indices < coord.size)
    valid_indices = indices[valid_indices_mask]
    subsample_coord_values = coord.values[valid_indices]

    dim = dim_name if len(obs_dims) > 1 else obs_dims[0]
    return stacked_data.sel({dim: subsample_coord_values})


def _warn_pareto_k(pareto_k_values, n_samples):
    """Check Pareto k values and issue warnings if necessary."""
    good_k = min(1 - 1 / np.log10(n_samples), 0.7) if n_samples > 1 else 0.7
    warn_mg = False

    if np.any(pareto_k_values > good_k):
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
