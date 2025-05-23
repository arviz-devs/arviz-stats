"""Helper functions for PSIS-LOO-CV."""

import warnings
from collections import namedtuple
from numbers import Number

import numpy as np
import xarray as xr
from arviz_base import convert_to_datatree, extract
from xarray_einstats.stats import logsumexp

from arviz_stats.utils import get_log_likelihood

__all__ = [
    "_split_moment_match",
    "_plpd_approx",
    "_diff_srs_estimator",
    "_srs_estimator",
    "_generate_subsample_indices",
    "_get_r_eff",
    "_prepare_loo_inputs",
    "_extract_loo_data",
    "_check_log_density",
    "_warn_pareto_k",
    "_warn_pointwise_loo",
    "_prepare_subsample",
    "_prepare_update_subsample",
    "_select_obs_by_indices",
    "_select_obs_by_coords",
    "_prepare_full_arrays",
    "_shift",
    "_shift_and_scale",
    "_shift_and_cov",
    "_recalculate_weights_k",
    "_update_loo_data_i",
    "_get_log_likelihood_i",
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

SplitMomentMatch = namedtuple("SplitMomentMatch", ["lwi", "lwfi", "log_liki", "reff"])
ShiftResult = namedtuple("ShiftResult", ["upars", "shift"])
ShiftAndScaleResult = namedtuple("ShiftAndScaleResult", ["upars", "shift", "scaling"])
ShiftAndCovResult = namedtuple("ShiftAndCovResult", ["upars", "shift", "mapping"])
RecalculateWeightsResult = namedtuple(
    "RecalculateWeightsResult", ["lwi", "lwfi", "ki", "kfi", "log_liki"]
)


def _split_moment_match(
    upars,
    cov,
    total_shift,
    total_scaling,
    total_mapping,
    i,
    reff,
    log_prob_upars_fn,
    log_lik_i_upars_fn,
):
    r"""Split moment matching importance sampling for PSIS-LOO-CV.

    Applies affine transformations based on the total moment matching transformation
    to half of the posterior draws, leaving the other half unchanged. These approximations
    to the leave-one-out posterior are then combined using multiple importance sampling.

    Based on the implicit adaptive importance sampling algorithm of [1]_ and the
    PSIS-LOO-CV method of [2]_ and [3]_.

    Parameters
    ----------
    upars : DataArray
        A DataArray representing the posterior draws of the model parameters in the
        unconstrained space. Must contain the dimensions `chain` and `draw` and a final
        dimension representing the different unconstrained parameters.
    cov : bool
        Whether to match the full covariance matrix of the samples (True) or just the
        marginal variances (False). Using the full covariance is more computationally
        expensive.
    total_shift : ndarray
        Vector containing the total shift (translation) applied to the parameters. Shape should
        match the parameter dimension of ``upars``.
    total_scaling : ndarray
        Vector containing the total scaling factors for the marginal variances. Shape should
        match the parameter dimension of ``upars``.
    total_mapping : ndarray
        Square matrix representing the linear transformation applied to the covariance matrix.
        Shape should be (d, d) where d is the parameter dimension.
    i : int
        Index of the specific observation to be left out for computing leave-one-out
        likelihood.
    reff : float
        Relative MCMC efficiency, ``ess / n`` i.e. number of effective samples divided by the number
        of actual samples.
    log_prob_upars_fn : Callable[[DataArray], DataArray]
        A function that computes the log probability density of the *full posterior*
        distribution evaluated at given unconstrained parameter values (as a DataArray).
        Input and Output must have dimensions `chain` and `draw`.
    log_lik_i_upars_fn : Callable[[DataArray, int], DataArray]
        A function that computes the log-likelihood of the *left-out observation* `i`
        evaluated at given unconstrained parameter values (as a DataArray).
        Input and Output must have dimensions `chain` and `draw`.

    Returns
    -------
    SplitMomentMatch
        A namedtuple containing:

        - lwi: Updated log importance weights for each sample
        - lwfi: Updated log importance weights for full distribution
        - log_liki: Updated log likelihood values for the specific observation
        - reff: Relative MCMC efficiency

    References
    ----------

    .. [1] Paananen, T., Piironen, J., Buerkner, P.-C., Vehtari, A. (2021). *Implicitly Adaptive
        Importance Sampling*. Statistics and Computing. 31(2) (2021)
        https://doi.org/10.1007/s11222-020-09982-2
        arXiv preprint https://arxiv.org/abs/1906.08850.
    .. [2] Vehtari et al. *Practical Bayesian model evaluation using leave-one-out cross-validation
        and WAIC*. Statistics and Computing. 27(5) (2017) https://doi.org/10.1007/s11222-016-9696-4
        arXiv preprint https://arxiv.org/abs/1507.04544.
    .. [3] Vehtari et al. *Pareto Smoothed Importance Sampling*.
        Journal of Machine Learning Research, 25(72) (2024) https://jmlr.org/papers/v25/19-556.html
        arXiv preprint https://arxiv.org/abs/1507.02646
    """
    if not isinstance(upars, xr.DataArray):
        raise TypeError("upars must be a DataArray.")

    sample_dims = ["chain", "draw"]
    param_dim_list = [dim for dim in upars.dims if dim not in sample_dims]

    if len(param_dim_list) != 1:
        raise ValueError("upars must have exactly one dimension besides chain and draw.")
    param_dim = param_dim_list[0]

    if not all(dim in upars.dims for dim in sample_dims):
        raise ValueError(
            f"Required sample dimensions {sample_dims} not found in upars dimensions {upars.dims}"
        )

    dim = upars.sizes[param_dim]
    n_samples = upars.sizes["chain"] * upars.sizes["draw"]
    n_samples_half = n_samples // 2

    upars_stacked = upars.stack(__sample__=sample_dims).transpose("__sample__", param_dim)
    mean_original = upars_stacked.mean(dim="__sample__")

    if total_shift is None or total_shift.size == 0:
        total_shift = np.zeros(dim)
    if total_scaling is None or total_scaling.size == 0:
        total_scaling = np.ones(dim)
    if total_mapping is None or total_mapping.size == 0:
        total_mapping = np.eye(dim)

    # Forward transformation
    upars_trans = upars_stacked - mean_original
    upars_trans = upars_trans * xr.DataArray(total_scaling, dims=param_dim)
    if cov and dim > 0:
        upars_trans = xr.DataArray(
            np.einsum("sp,pq->sq", upars_trans.data, total_mapping),
            coords=upars_trans.coords,
            dims=upars_trans.dims,
        )
    upars_trans = upars_trans + (xr.DataArray(total_shift, dims=param_dim) + mean_original)

    # Inverse transformation
    upars_trans_inv = upars_stacked - mean_original

    if cov and dim > 0:
        try:
            inv_mapping = np.linalg.inv(total_mapping)
            upars_trans_inv = xr.DataArray(
                np.einsum("sp,pq->sq", upars_trans_inv.data, inv_mapping),
                coords=upars_trans_inv.coords,
                dims=upars_trans_inv.dims,
            )
        except np.linalg.LinAlgError:
            warnings.warn("Could not invert mapping matrix. Using identity.", UserWarning)

    upars_trans_inv = upars_trans_inv / xr.DataArray(total_scaling, dims=param_dim)
    upars_trans_inv = upars_trans_inv + (mean_original - xr.DataArray(total_shift, dims=param_dim))

    upars_trans_half = upars_stacked.copy(deep=True).unstack("__sample__")
    upars_trans_half = upars_trans_half.transpose(*sample_dims, param_dim)
    upars_trans_half.values.reshape(-1, dim)[:n_samples_half] = upars_trans.values.reshape(-1, dim)[
        :n_samples_half
    ]

    upars_trans_half_inv = upars_stacked.copy(deep=True).unstack("__sample__")
    upars_trans_half_inv = upars_trans_half_inv.transpose(*sample_dims, param_dim)
    upars_trans_half_inv.values.reshape(-1, dim)[n_samples_half:] = upars_trans_inv.values.reshape(
        -1, dim
    )[n_samples_half:]

    try:
        log_prob_half_trans = log_prob_upars_fn(upars_trans_half)
        log_prob_half_trans_inv = log_prob_upars_fn(upars_trans_half_inv)
    except Exception as e:
        raise ValueError(
            f"Could not compute log probabilities for transformed parameters: {e}"
        ) from e

    try:
        log_liki_half = log_lik_i_upars_fn(upars_trans_half, i)
        if not all(dim in log_liki_half.dims for dim in sample_dims) or len(
            log_liki_half.dims
        ) != len(sample_dims):
            raise ValueError(
                f"log_lik_i_upars_fn must return a DataArray with dimensions {sample_dims}"
            )
        if (
            log_liki_half.sizes["chain"] != upars.sizes["chain"]
            or log_liki_half.sizes["draw"] != upars.sizes["draw"]
        ):
            raise ValueError(
                "log_lik_i_upars_fn output shape does not match input sample dimensions"
            )
    except Exception as e:
        raise ValueError(f"Could not compute log likelihood for observation {i}: {e}") from e

    # Jacobian adjustment
    log_jacobian_det = 0.0
    if dim > 0:
        log_jacobian_det = -np.sum(np.log(total_scaling))
        if cov:
            try:
                sign, logdet = np.linalg.slogdet(total_mapping)
                if sign <= 0:
                    log_jacobian_det -= np.inf
                else:
                    log_jacobian_det -= logdet
            except np.linalg.LinAlgError:
                log_jacobian_det -= np.inf

    log_prob_half_trans_inv_adj = log_prob_half_trans_inv + log_jacobian_det

    # Multiple importance sampling
    use_forward_log_prob = log_prob_half_trans > log_prob_half_trans_inv_adj
    raw_log_weights_half = -log_liki_half + log_prob_half_trans

    log_sum_terms = xr.where(
        use_forward_log_prob,
        log_prob_half_trans
        + xr.ufuncs.log1p(xr.ufuncs.exp(log_prob_half_trans_inv_adj - log_prob_half_trans)),
        log_prob_half_trans_inv_adj
        + xr.ufuncs.log1p(xr.ufuncs.exp(log_prob_half_trans - log_prob_half_trans_inv_adj)),
    )

    raw_log_weights_half -= log_sum_terms
    raw_log_weights_half = xr.where(np.isnan(raw_log_weights_half), -np.inf, raw_log_weights_half)
    raw_log_weights_half = xr.where(
        np.isposinf(raw_log_weights_half), -np.inf, raw_log_weights_half
    )

    # PSIS smoothing for half posterior
    lwi_psis_da, _ = raw_log_weights_half.azstats.psislw(r_eff=reff, dims=sample_dims)

    lr_full = lwi_psis_da + log_liki_half
    lr_full = xr.where(np.isnan(lr_full) | (np.isinf(lr_full) & (lr_full > 0)), -np.inf, lr_full)

    # PSIS smoothing for full posterior
    lwfi_psis_da, _ = lr_full.azstats.psislw(r_eff=reff, dims=sample_dims)

    return SplitMomentMatch(
        lwi=lwi_psis_da,
        lwfi=lwfi_psis_da,
        log_liki=log_liki_half,
        reff=reff,
    )


def _shift(upars, lwi):
    """Shift a DataArray of parameters to their weighted mean."""
    sample_dims = ["chain", "draw"]
    param_dim = [dim for dim in upars.dims if dim not in sample_dims][0]
    upars_stacked = upars.stack(__sample__=sample_dims)
    lwi_stacked = lwi.stack(__sample__=sample_dims)

    upars_values = upars_stacked.transpose("__sample__", param_dim).data
    lwi_values = lwi_stacked.transpose("__sample__").data

    mean_original = np.mean(upars_values, axis=0)
    weights = np.exp(
        lwi_values - logsumexp(lwi_stacked.transpose("__sample__"), dims="__sample__").data
    )
    mean_weighted = np.sum(weights[:, None] * upars_values, axis=0)
    shift_vec = mean_weighted - mean_original
    upars_new_values = upars_values + shift_vec[None, :]

    upars_new_stacked = xr.DataArray(
        upars_new_values,
        dims=["__sample__", param_dim],
        coords={
            "__sample__": upars_stacked["__sample__"],
            param_dim: upars_stacked[param_dim],
        },
    )
    upars_new_da = upars_new_stacked.unstack("__sample__")
    return ShiftResult(upars=upars_new_da, shift=shift_vec)


def _shift_and_scale(upars, lwi):
    """Shift parameters to weighted mean and scale marginal variances."""
    sample_dims = ["chain", "draw"]
    param_dim = [dim for dim in upars.dims if dim not in sample_dims][0]
    upars_stacked = upars.stack(__sample__=sample_dims)
    lwi_stacked = lwi.stack(__sample__=sample_dims)

    upars_values = upars_stacked.transpose("__sample__", param_dim).data
    lwi_values = lwi_stacked.transpose("__sample__").data

    mean_original = np.mean(upars_values, axis=0)
    weights = np.exp(
        lwi_values - logsumexp(lwi_stacked.transpose("__sample__"), dims="__sample__").data
    )
    mean_weighted = np.sum(weights[:, None] * upars_values, axis=0)
    shift_vec = mean_weighted - mean_original

    var_weighted = np.sum(weights[:, None] * (upars_values - mean_weighted[None, :]) ** 2, axis=0)
    ess_approx = 1.0 / np.sum(weights**2)

    if ess_approx > 1:
        var_weighted *= ess_approx / (ess_approx - 1)
    else:
        var_weighted = np.var(upars_values, axis=0)

    var_original = np.var(upars_values, axis=0, ddof=1)

    scaling_vec = np.ones_like(mean_original)
    valid_mask = (var_original > 1e-9) & (var_weighted > 1e-9)
    scaling_vec[valid_mask] = np.sqrt(var_weighted[valid_mask] / var_original[valid_mask])

    upars_new_values = upars_values - mean_original[None, :]
    upars_new_values = upars_new_values * scaling_vec[None, :]
    upars_new_values = upars_new_values + mean_weighted[None, :]

    upars_new_stacked = xr.DataArray(
        upars_new_values,
        dims=["__sample__", param_dim],
        coords={
            "__sample__": upars_stacked["__sample__"],
            param_dim: upars_stacked[param_dim],
        },
    )
    upars_new_da = upars_new_stacked.unstack("__sample__")
    return ShiftAndScaleResult(upars=upars_new_da, shift=shift_vec, scaling=scaling_vec)


def _shift_and_cov(upars, lwi):
    """Shift parameters and scale covariance to match weighted covariance."""
    sample_dims = ["chain", "draw"]
    param_dim = [dim for dim in upars.dims if dim not in sample_dims][0]
    upars_stacked = upars.stack(__sample__=sample_dims)
    lwi_stacked = lwi.stack(__sample__=sample_dims)

    upars_values = upars_stacked.transpose("__sample__", param_dim).data
    lwi_values = lwi_stacked.transpose("__sample__").data

    mean_original = np.mean(upars_values, axis=0)
    weights = np.exp(
        lwi_values - logsumexp(lwi_stacked.transpose("__sample__"), dims="__sample__").data
    )
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
        mapping_mat = chol_weighted @ np.linalg.inv(chol_original)

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

    upars_new_stacked = xr.DataArray(
        upars_new_values,
        dims=["__sample__", param_dim],
        coords={
            "__sample__": upars_stacked["__sample__"],
            param_dim: upars_stacked[param_dim],
        },
    )
    upars_new_da = upars_new_stacked.unstack("__sample__")
    return ShiftAndCovResult(upars=upars_new_da, shift=shift_vec, mapping=mapping_mat)


def _get_log_likelihood_i(log_likelihood, i, obs_dims):
    """Extract the log likelihood for a specific observation index `i`."""
    if not obs_dims:
        raise ValueError("log_likelihood must have observation dimensions.")

    if len(obs_dims) == 1:
        obs_dim = obs_dims[0]
        if i < 0 or i >= log_likelihood.sizes[obs_dim]:
            raise IndexError(f"Index {i} is out of bounds for dimension '{obs_dim}'.")
        log_lik_i = log_likelihood.isel({obs_dim: i})
    else:
        stacked_obs_dim = "__obs__"
        log_lik_stacked = log_likelihood.stack({stacked_obs_dim: obs_dims})
        if i < 0 or i >= log_lik_stacked.sizes[stacked_obs_dim]:
            raise IndexError(
                f"Index {i} is out of bounds for stacked dimension '{stacked_obs_dim}'."
            )
        log_lik_i = log_lik_stacked.isel({stacked_obs_dim: i})
    return log_lik_i


def _recalculate_weights_k(
    log_liki_new,
    log_prob_new,
    orig_log_prob,
    reff,
    sample_dims,
):
    """Recalculate importance weights and Pareto k after parameter transformation."""
    log_ratio_i = -log_liki_new + log_prob_new - orig_log_prob
    log_ratio_i = xr.where(np.isnan(log_ratio_i), -np.inf, log_ratio_i)

    lwi_new, ki_new = log_ratio_i.azstats.psislw(r_eff=reff, dims=sample_dims)
    ki_new = ki_new[0].item() if isinstance(ki_new, tuple) else ki_new.item()

    log_ratio_full = log_prob_new - orig_log_prob
    log_ratio_full = xr.where(np.isnan(log_ratio_full), -np.inf, log_ratio_full)

    lwfi_new, kfi_new = log_ratio_full.azstats.psislw(r_eff=reff, dims=sample_dims)
    kfi_new = kfi_new[0].item() if isinstance(kfi_new, tuple) else kfi_new.item()

    return RecalculateWeightsResult(
        lwi=lwi_new, lwfi=lwfi_new, ki=ki_new, kfi=kfi_new, log_liki=log_liki_new
    )


def _update_loo_data_i(
    loo_data,
    i,
    new_elpd_i,
    new_pareto_k,
    log_liki,
    sample_dims,
    obs_dims,
    n_samples,
    original_log_liki=None,
):
    """Update the ELPDData object for a single observation."""
    if loo_data.elpd_i is None or loo_data.pareto_k is None:
        raise ValueError("loo_data must contain pointwise elpd_i and pareto_k values.")

    lpd_i_log_lik = original_log_liki if original_log_liki is not None else log_liki
    lpd_i = logsumexp(lpd_i_log_lik, dims=sample_dims, b=1 / n_samples).item()
    p_loo_i = lpd_i - new_elpd_i

    if len(obs_dims) == 1:
        idx_dict = {obs_dims[0]: i}
    else:
        coords = np.unravel_index(i, tuple(loo_data.elpd_i.sizes[d] for d in obs_dims))
        idx_dict = dict(zip(obs_dims, coords))

    loo_data.elpd_i[idx_dict] = new_elpd_i
    loo_data.pareto_k[idx_dict] = new_pareto_k

    if not hasattr(loo_data, "p_loo_i") or loo_data.p_loo_i is None:
        loo_data.p_loo_i = xr.full_like(loo_data.elpd_i, np.nan)

    loo_data.p_loo_i[idx_dict] = p_loo_i

    loo_data.elpd = float(np.nansum(loo_data.elpd_i.values))
    loo_data.p = float(np.nansum(loo_data.p_loo_i.values))
    loo_data.se = float(np.sqrt(loo_data.n_data_points * np.nanvar(loo_data.elpd_i.values)))

    loo_data.warning, loo_data.good_k = _warn_pareto_k(
        loo_data.pareto_k.values[~np.isnan(loo_data.pareto_k.values)], loo_data.n_samples
    )


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
        A function that computes the log-likelihood for a single observation given the
        mean values of posterior parameters. Required only when ``method="plpd"``.
        The function must accept the observed data value for a single point as its
        first argument (scalar). Subsequent arguments must correspond to the mean
        values of the posterior parameters specified by ``param_names``, passed in the
        same order. It should return a single scalar log-likelihood value.
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

            if not isinstance(result, Number):
                raise TypeError(
                    f"log_lik_fn must return a numeric scalar. Got type: {type(result)}"
                )
            plpd_values[idx] = result
        except Exception as e:
            coord_info = ", ".join(
                [f"{dim}: {observed[idx].coords[dim].values.item()}" for dim in observed.coords]
            )
            raise RuntimeError(
                f"Error computing log-likelihood with log_lik_fn for data point "
                f"with coordinates ({coord_info}) "
                f"at index {idx} (value: {value}): {e}"
            ) from e

    plpd_da = xr.DataArray(
        plpd_values,
        dims=observed.dims,
        coords=observed.coords,
        name="plpd",
    )
    return plpd_da


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
):
    """Prepare inputs for PSIS-LOO-CV with sub-sampling."""
    indices, subsample_size = _generate_subsample_indices(n_data_points, observations, seed)

    if method == "lpd":
        lpd_approx_all = logsumexp(log_likelihood_da, dims=sample_dims, b=1 / n_samples)
    else:  # method == "plpd"
        lpd_approx_all = _plpd_approx(
            data=data, var_name=var_name, log_lik_fn=log_lik_fn, param_names=param_names, log=log
        )

    stacked_obs_dim = "__obs__"
    if len(obs_dims) > 1:
        log_likelihood_stacked = log_likelihood_da.stack({stacked_obs_dim: obs_dims})
        lpd_approx_all_stacked = lpd_approx_all.stack({stacked_obs_dim: obs_dims})

        log_likelihood_sample = _select_obs_by_indices(
            log_likelihood_stacked, indices, [stacked_obs_dim], stacked_obs_dim
        )
        lpd_approx_sample = _select_obs_by_indices(
            lpd_approx_all_stacked, indices, [stacked_obs_dim], stacked_obs_dim
        )
    else:
        obs_dim_name = obs_dims[0]
        log_likelihood_sample = _select_obs_by_indices(
            log_likelihood_da, indices, obs_dims, obs_dim_name
        )
        lpd_approx_sample = _select_obs_by_indices(lpd_approx_all, indices, obs_dims, obs_dim_name)

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
):
    """Prepare inputs for updating PSIS-LOO-CV with additional observations."""
    loo_inputs = _prepare_loo_inputs(data, var_name)
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

    if method == "lpd":
        lpd_approx_all = logsumexp(
            log_likelihood, dims=loo_inputs.sample_dims, b=1 / loo_inputs.n_samples
        )
    else:  # method == "plpd"
        lpd_approx_all = _plpd_approx(
            data,
            loo_inputs.var_name,
            log_lik_fn,
            param_names,
            log,
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

    target_dim = "__obs__" if len(obs_dims) > 1 else obs_dims[0]
    elpd_i_full[{target_dim: indices}] = pointwise_values.values
    pareto_k_full[{target_dim: indices}] = pareto_k_values.values

    if elpd_loo_hat is not None:
        _warn_pointwise_loo(elpd_loo_hat, elpd_i_full.values)
    return elpd_i_full, pareto_k_full


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


def _select_obs_by_indices(data_array, indices, dims, dim_name):
    """Select a sub-sample from a DataArray based on indices."""
    if not dims:
        return data_array

    if len(dims) > 1:
        stacked_data = data_array.stack({dim_name: dims})
        coord = stacked_data[dim_name]
    else:
        stacked_data = data_array
        coord = stacked_data[dims[0]]

    valid_indices_mask = (indices >= 0) & (indices < coord.size)
    valid_indices = indices[valid_indices_mask]

    dim = dim_name if len(dims) > 1 else dims[0]
    return stacked_data.isel({dim: valid_indices})


def _select_obs_by_coords(data_array, coord_array, dims, dim_name):
    """Select a sub-sample from a DataArray based on coordinate values."""
    if len(dims) > 1:
        stacked_data = data_array.stack({dim_name: dims})
        return stacked_data.sel({dim_name: coord_array[dim_name]})
    return data_array.sel({dims[0]: coord_array[dims[0]]})


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
