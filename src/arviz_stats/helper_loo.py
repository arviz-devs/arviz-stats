"""Helper functions for PSIS-LOO-CV."""

import warnings

import numpy as np
import xarray as xr
from arviz_base import convert_to_datatree

from arviz_stats.utils import get_log_likelihood_dataset

__all__ = [
    "_get_r_eff",
    "_diff_srs_estimator",
    "_generate_subsample_indices",
    "_prepare_loo_inputs",
    "_extract_loo",
    "_check_log_density",
    "_warn_pareto_k",
    "_warn_pointwise_loo",
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
    elpd_loo_i_sample : ndarray
        Pointwise ELPD values for the subsample.
    lpd_approx_sample : ndarray
        LPD approximation values for the subsample.
    lpd_approx_all : ndarray
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
    lpd_approx_sum_all = np.sum(lpd_approx_all)
    scaled_mean_pointwise_diff = n_data_points * np.mean(pointwise_diff)
    elpd_loo_estimate = lpd_approx_sum_all + scaled_mean_pointwise_diff

    if subsample_size > 1:
        # Subsampling variance
        subsampling_variance = (
            (n_data_points**2)
            * (1 - subsample_size / n_data_points)
            * np.var(pointwise_diff, ddof=1)
            / subsample_size
        )
        subsampling_se = np.sqrt(subsampling_variance) if subsampling_variance >= 0 else np.inf

        # Total variance (approximation + sampling)
        lpd_approx_sq_sum_all = np.sum(lpd_approx_all**2)
        mean_sq_diff = np.mean(elpd_loo_i_sample**2 - lpd_approx_sample**2)
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


def _extract_loo(loo_orig):
    """Extract data from original LOO object."""
    old_elpd_i = loo_orig.elpd_i
    old_elpd_i_flat = (
        old_elpd_i.values.flatten()
        if hasattr(old_elpd_i, "values")
        else np.asarray(old_elpd_i).flatten()
    )
    valid_old_mask = ~np.isnan(old_elpd_i_flat)
    old_indices = np.where(valid_old_mask)[0]
    old_elpd_i_values = old_elpd_i_flat[valid_old_mask]

    old_pareto_k_values = np.array([])
    if loo_orig.pareto_k is not None:
        pareto_k = loo_orig.pareto_k
        if isinstance(pareto_k, xr.DataArray) and "obs_id_subsample" in pareto_k.coords:
            k_dict = dict(zip(pareto_k.coords["obs_id_subsample"].values, pareto_k.values))
            old_pareto_k_values = np.array([k_dict.get(idx, np.nan) for idx in old_indices])
        else:
            k_flat = (
                pareto_k.values.flatten()
                if hasattr(pareto_k, "values")
                else np.asarray(pareto_k).flatten()
            )
            if len(k_flat) == len(old_indices):
                old_pareto_k_values = k_flat
            else:
                old_pareto_k_values = np.full_like(old_elpd_i_values, np.nan)
    return old_indices, old_elpd_i_values, old_pareto_k_values


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
