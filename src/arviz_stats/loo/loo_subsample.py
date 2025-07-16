"""Compute PSIS-LOO-CV using sub-sampling."""

import numpy as np
import xarray as xr
from arviz_base import rcParams
from xarray_einstats.stats import logsumexp

from arviz_stats.loo.helper_loo import (
    _compute_loo_results,
    _diff_srs_estimator,
    _get_r_eff,
    _prepare_full_arrays,
    _prepare_loo_inputs,
    _prepare_subsample,
    _prepare_update_subsample,
    _select_obs_by_coords,
    _select_obs_by_indices,
    _srs_estimator,
    _warn_pareto_k,
)
from arviz_stats.loo.loo_approximate_posterior import loo_approximate_posterior
from arviz_stats.utils import ELPDData


def loo_subsample(
    data,
    observations,
    pointwise=None,
    var_name=None,
    reff=None,
    log_weights=None,
    log_p=None,
    log_q=None,
    seed=315,
    method="lpd",
    thin=None,
    log_lik_fn=None,
    param_names=None,
    log=True,
):
    """Compute PSIS-LOO-CV using sub-sampling.

    Estimates the expected log pointwise predictive density (elpd) using Pareto smoothed
    importance sampling leave-one-out cross-validation (PSIS-LOO-CV) with sub-sampling
    for large datasets. Uses either log predictive density (LPD) or point log predictive
    density (PLPD) approximation and applies a difference estimator based on a simple random
    sample without replacement.

    The PSIS-LOO-CV method is described in [1]_, [2]_. The sub-sampling
    method is described in [3]_.

    Parameters
    ----------
    data : DataTree or InferenceData
        Input data. It should contain the posterior and the log_likelihood groups.
    observations : int or ndarray
        The sub-sample observations to use:

        - An integer specifying the number of observations to randomly sub-sample without
          replacement.
        - An array of integer indices specifying the exact observations to use.
    pointwise: bool, optional
        If True the pointwise predictive accuracy will be returned. Defaults to
        ``rcParams["stats.ic_pointwise"]``.
    var_name : str, optional
        The name of the variable in log_likelihood groups storing the pointwise log
        likelihood data to use for loo computation.
    reff: float, optional
        Relative MCMC efficiency, ``ess / n`` i.e. number of effective samples divided by the number
        of actual samples. Computed from trace by default.
    log_weights : DataArray or ELPDData, optional
        Smoothed log weights. Can be either:

        - A DataArray with the same shape as the log likelihood data
        - An ELPDData object from a previous :func:`arviz_stats.loo` call.

        Defaults to None. If not provided, it will be computed using the PSIS-LOO method.
    log_p : ndarray or DataArray, optional
        The (target) log-density evaluated at samples from the target distribution (p).
        If provided along with ``log_q``, approximate posterior correction will be applied.
    log_q : ndarray or DataArray, optional
        The (proposal) log-density evaluated at samples from the proposal distribution (q).
        If provided along with ``log_p``, approximate posterior correction will be applied.
    seed: int, optional
        Seed for random sampling.
    method: str, optional
        Method used for approximating the pointwise log predictive density:

        - 'lpd': Use standard log predictive density approximation (default)
        - 'plpd': Use Point Log Predictive Density approximation which requires a ``log_lik_fn``.
    thin: int, optional
        Thinning factor for posterior draws. If specified, the posterior will be thinned
        by this factor to reduce computation time. For example, using thin=2 will use
        every 2nd draw. If None (default), all posterior draws are used. This value is stored
        in the returned ELPDData object and will be automatically used by ``update_subsample``.
    log_lik_fn : callable, optional
        A function that computes the log-likelihood for a single observation given the
        mean values of posterior parameters. Required only when ``method="plpd"``.
        The function must accept the observed data value for a single point as its
        first argument (scalar). Subsequent arguments must correspond to the mean
        values of the posterior parameters specified by ``param_names``, passed in the
        same order. It should return a single scalar log-likelihood value.
    param_names : list, optional
        Only used when ``method="plpd"``. List of parameter names to extract from
        the posterior. If None, all parameters are used.
    log: bool, optional
        Only used when ``method="plpd"``. Whether the ``log_lik_fn`` returns
        log-likelihood (True) or likelihood (False). Default is True.

    Returns
    -------
    ELPDData
        Object with the following attributes:

        - **elpd**: approximated expected log pointwise predictive density (elpd)
        - **se**: standard error of the elpd (includes approximation and sampling uncertainty)
        - **p**: effective number of parameters
        - **n_samples**: number of samples in the posterior
        - **n_data_points**: total number of data points (N)
        - **warning**: True if the estimated shape parameter k of the Pareto distribution
          is > ``good_k`` for any observation in the subsample.
        - **elpd_i**: :class:`~xarray.DataArray` with pointwise elpd values (filled with NaNs
          for non-subsampled points), only if ``pointwise=True``.
        - **pareto_k**: :class:`~xarray.DataArray` with Pareto shape values for the subsample
          (filled with NaNs for non-subsampled points), only if ``pointwise=True``.
        - **scale**: scale of the elpd results ("log", "negative_log", or "deviance").
        - **good_k**: Threshold for Pareto k warnings.
        - **approx_posterior**: True if approximate posterior was used.
        - **subsampling_se**: Standard error estimate from subsampling uncertainty only.
        - **subsample_size**: Number of observations in the subsample (m).
        - **log_p**: Log density of the target posterior.
        - **log_q**: Log density of the proposal posterior.
        - **thin**: Thinning factor for posterior draws.
        - **log_weights**: Smoothed log weights.

    Examples
    --------
    Calculate sub-sampled PSIS-LOO-CV using 4 random observations:

    .. ipython::

        In [1]: from arviz_stats import loo_subsample
           ...: from arviz_base import load_arviz_data
           ...: data = load_arviz_data("centered_eight")
           ...: loo_results = loo_subsample(data, observations=4, var_name="obs", pointwise=True)
           ...: loo_results

    Return the pointwise values for the sub-sample:

    .. ipython::

        In [2]: loo_results.elpd_i

    We can also use the PLPD approximation method with a custom log-likelihood function.
    We need to define a function that computes the log-likelihood for a single observation
    given the mean values of posterior parameters. For the Eight Schools model, we define a
    function that computes the likelihood for each observation using the *global mean* of the
    parameters (e.g., the overall mean `theta`):

    .. ipython::

        In [1]: import numpy as np
           ...: from arviz_stats import loo_subsample
           ...: from arviz_base import load_arviz_data
           ...: from scipy.stats import norm
           ...: data = load_arviz_data("centered_eight")
           ...:
           ...: def log_lik_fn(y, theta):
           ...:     sigma = 12.5  # Using a fixed sigma for simplicity
           ...:     return norm.logpdf(y, loc=theta, scale=sigma)
           ...:
           ...: loo_results = loo_subsample(
           ...:     data,
           ...:     observations=4,
           ...:     var_name="obs",
           ...:     method="plpd",
           ...:     log_lik_fn=log_lik_fn,
           ...:     param_names=["theta"],
           ...:     pointwise=True
           ...: )
           ...: loo_results

    See Also
    --------
    loo : Standard PSIS-LOO-CV.
    loo_approximate_posterior : PSIS-LOO-CV for approximate posteriors.
    compare : Compare models based on ELPD.
    update_subsample : Update a previously computed sub-sampled LOO-CV.

    References
    ----------

    .. [1] Vehtari et al. *Practical Bayesian model evaluation using leave-one-out cross-validation
        and WAIC*. Statistics and Computing. 27(5) (2017) https://doi.org/10.1007/s11222-016-9696-4
        arXiv preprint https://arxiv.org/abs/1507.04544.

    .. [2] Vehtari et al. *Pareto Smoothed Importance Sampling*.
        Journal of Machine Learning Research, 25(72) (2024) https://jmlr.org/papers/v25/19-556.html
        arXiv preprint https://arxiv.org/abs/1507.02646

    .. [3] Magnusson, M., Riis Andersen, M., Jonasson, J., & Vehtari, A. *Bayesian Leave-One-Out
        Cross-Validation for Large Data.* Proceedings of the 36th International Conference on
        Machine Learning, PMLR 97:4244–4253 (2019)
        https://proceedings.mlr.press/v97/magnusson19a.html
        arXiv preprint https://arxiv.org/abs/1904.10679
    """
    loo_inputs = _prepare_loo_inputs(data, var_name, thin)
    pointwise = rcParams["stats.ic_pointwise"] if pointwise is None else pointwise

    if method not in ["lpd", "plpd"]:
        raise ValueError("Method must be either 'lpd' or 'plpd'")
    if method == "plpd" and log_lik_fn is None:
        raise ValueError("log_lik_fn must be provided when method='plpd'")

    log_likelihood = loo_inputs.log_likelihood
    if reff is None:
        reff = _get_r_eff(data, loo_inputs.n_samples)

    subsample_data = _prepare_subsample(
        data,
        log_likelihood,
        loo_inputs.var_name,
        observations,
        seed,
        method,
        log_lik_fn,
        param_names,
        log,
        loo_inputs.obs_dims,
        loo_inputs.sample_dims,
        loo_inputs.n_data_points,
        loo_inputs.n_samples,
    )

    sample_ds = xr.Dataset({loo_inputs.var_name: subsample_data.log_likelihood_sample})

    if log_p is not None and log_q is not None:
        sample_data = xr.DataTree()
        sample_data["log_likelihood"] = sample_ds

        loo_approx = loo_approximate_posterior(
            sample_data,
            log_p,
            log_q,
            True,
            loo_inputs.var_name,
        )
        elpd_loo_i = loo_approx.elpd_i
        pareto_k_sample_da = loo_approx.pareto_k
        approx_posterior = True
    else:
        if log_weights is not None:
            if isinstance(log_weights, ELPDData):
                if log_weights.log_weights is None:
                    raise ValueError("ELPDData object does not contain log_weights")
                log_weights = log_weights.log_weights
                if loo_inputs.var_name in log_weights:
                    log_weights = log_weights[loo_inputs.var_name]

            if len(loo_inputs.obs_dims) > 1:
                stacked_obs_dim = "__obs__"
                log_weights_stacked = log_weights.stack({stacked_obs_dim: loo_inputs.obs_dims})
                log_weights_sample = _select_obs_by_indices(
                    log_weights_stacked, subsample_data.indices, [stacked_obs_dim], stacked_obs_dim
                )
                log_weights_sample = log_weights_sample.unstack(stacked_obs_dim)
            else:
                obs_dim = loo_inputs.obs_dims[0]
                log_weights_sample = _select_obs_by_indices(
                    log_weights, subsample_data.indices, loo_inputs.obs_dims, obs_dim
                )

            log_weights_sample_ds = xr.Dataset({loo_inputs.var_name: log_weights_sample})

            _, pareto_k_ds = sample_ds.azstats.psislw(r_eff=reff, dim=loo_inputs.sample_dims)

            log_weights_ds = log_weights_sample_ds + sample_ds
        else:
            log_weights_ds, pareto_k_ds = sample_ds.azstats.psislw(
                r_eff=reff, dim=loo_inputs.sample_dims
            )
            log_weights_sample = log_weights_ds[loo_inputs.var_name]
            log_weights_ds += sample_ds

        elpd_loo_i = logsumexp(log_weights_ds, dims=loo_inputs.sample_dims)[loo_inputs.var_name]
        pareto_k_sample_da = pareto_k_ds[loo_inputs.var_name]
        approx_posterior = False

    warn_mg, good_k = _warn_pareto_k(pareto_k_sample_da, loo_inputs.n_samples)

    elpd_loo_hat, subsampling_se, se = _diff_srs_estimator(
        elpd_loo_i,
        subsample_data.lpd_approx_sample,
        subsample_data.lpd_approx_all,
        loo_inputs.n_data_points,
        subsample_data.subsample_size,
    )

    # Calculate p_loo using SRS estimation directly on the p_loo values
    # from the subsample
    p_loo_sample = subsample_data.lpd_approx_sample - elpd_loo_i
    p_loo, _, _ = _srs_estimator(
        p_loo_sample,
        loo_inputs.n_data_points,
        subsample_data.subsample_size,
    )

    if not pointwise:
        stored_log_weights = log_weights_sample if "log_weights_sample" in locals() else None
        return ELPDData(
            "loo",
            elpd_loo_hat,
            se,
            p_loo,
            loo_inputs.n_samples,
            loo_inputs.n_data_points,
            "log",
            warn_mg,
            good_k,
            None,
            None,
            approx_posterior,
            subsampling_se,
            subsample_data.subsample_size,
            log_p,
            log_q,
            thin,
            stored_log_weights,
        )

    elpd_i_full, pareto_k_full = _prepare_full_arrays(
        elpd_loo_i,
        pareto_k_sample_da,
        subsample_data.lpd_approx_all,
        subsample_data.indices,
        loo_inputs.obs_dims,
        elpd_loo_hat,
    )

    if "log_weights_sample" in locals() and log_weights_sample is not None:
        log_weights_full = xr.Dataset({loo_inputs.var_name: log_weights_sample})
    else:
        log_weights_full = None

    return ELPDData(
        "loo",
        elpd_loo_hat,
        se,
        p_loo,
        loo_inputs.n_samples,
        loo_inputs.n_data_points,
        "log",
        warn_mg,
        good_k,
        elpd_i_full,
        pareto_k_full,
        approx_posterior,
        subsampling_se,
        subsample_data.subsample_size,
        log_p,
        log_q,
        thin,
        log_weights_full,
    )


def update_subsample(
    loo_orig,
    data,
    observations=None,
    var_name=None,
    reff=None,
    log_weights=None,
    seed=315,
    method="lpd",
    log_lik_fn=None,
    param_names=None,
    log=True,
):
    """Update a sub-sampled PSIS-LOO-CV object with new observations.

    Extends a sub-sampled PSIS-LOO-CV result by adding new observations to the sub-sample
    without recomputing values for previously sampled observations. This allows for
    incrementally improving the sub-sampled PSIS-LOO-CV estimate with additional observations.

    The sub-sampling method is described in [1]_.

    Parameters
    ----------
    loo_orig : ELPDData
        Original PSIS-LOO-CV result created with ``loo_subsample`` with ``pointwise=True``.
    data : DataTree or InferenceData
        Input data. It should contain the posterior and the log_likelihood groups.
    observations : int or ndarray, optional
        The additional observations to use:

        - An integer specifying the number of new observations to randomly sub-sample
          without replacement.
        - An array of integer indices specifying the exact new observations to use.
        - If None or 0, returns the original PSIS-LOO-CV result unchanged.
    var_name : str, optional
        The name of the variable in log_likelihood groups storing the pointwise log
        likelihood data to use for loo computation.
    reff : float, optional
        Relative MCMC efficiency, ``ess / n`` i.e. number of effective samples divided by the number
        of actual samples. Computed from trace by default.
    log_weights : DataArray or ELPDData, optional
        Smoothed log weights. Can be either:

        - A DataArray with the same shape as the log likelihood data
        - An ELPDData object from a previous :func:`arviz_stats.loo` call.

        Defaults to None. If not provided, it will be computed using the PSIS-LOO method.
    seed : int, optional
        Seed for random sampling.
    method: str, optional
        Method used for approximating the pointwise log predictive density:

        - 'lpd': Use standard log predictive density approximation (default)
        - 'plpd': Use Point Log Predictive Density approximation which requires a ``log_lik_fn``.
    log_lik_fn : callable, optional
        A function that computes the log-likelihood for a single observation given the
        mean values of posterior parameters. Required only when ``method="plpd"``.
        The function must accept the observed data value for a single point as its
        first argument (scalar). Subsequent arguments must correspond to the mean
        values of the posterior parameters specified by ``param_names``, passed in the
        same order. It should return a single scalar log-likelihood value.
    param_names: list, optional
        Only used when ``method="plpd"``. List of parameter names to extract from
        the posterior. If None, all parameters are used.
    log: bool, optional
        Only used when ``method="plpd"``. Whether the ``log_lik_fn`` returns
        log-likelihood (True) or likelihood (False). Default is True.

    Returns
    -------
    ELPDData
        Object with the following attributes:

        - **elpd**: updated approximated expected log pointwise predictive density (elpd)
        - **se**: standard error of the elpd (includes approximation and sampling uncertainty)
        - **p**: effective number of parameters
        - **n_samples**: number of samples in the posterior
        - **n_data_points**: total number of data points (N)
        - **warning**: True if the estimated shape parameter k of the Pareto distribution
          is > ``good_k`` for any observation in the subsample.
        - **elpd_i**: :class:`~xarray.DataArray` with pointwise elpd values (filled with NaNs
          for non-subsampled points), only if ``pointwise=True``.
        - **pareto_k**: :class:`~xarray.DataArray` with Pareto shape values for the subsample
          (filled with NaNs for non-subsampled points), only if ``pointwise=True``.
        - **scale**: scale of the elpd results ("log", "negative_log", or "deviance").
        - **good_k**: Threshold for Pareto k warnings.
        - **approx_posterior**: True if approximate posterior was used.
        - **subsampling_se**: Standard error estimate from subsampling uncertainty only.
        - **subsample_size**: Number of observations in the subsample (original + new).
        - **log_p**: Log density of the target posterior.
        - **log_q**: Log density of the proposal posterior.
        - **thin**: Thinning factor for posterior draws.
        - **log_weights**: Smoothed log weights.

    Examples
    --------
    Calculate initial sub-sampled PSIS-LOO-CV using 4 observations, then update with 4 more:

    .. ipython::
        :okwarning:

        In [1]: from arviz_stats import loo_subsample, update_subsample
           ...: from arviz_base import load_arviz_data
           ...: data = load_arviz_data("non_centered_eight")
           ...: initial_loo = loo_subsample(data, observations=4, var_name="obs", pointwise=True)
           ...: updated_loo = update_subsample(initial_loo, data, observations=2)
           ...: updated_loo

    See Also
    --------
    loo : Exact PSIS-LOO cross-validation.
    loo_subsample : PSIS-LOO-CV with subsampling.
    compare : Compare models based on ELPD.

    References
    ----------

    .. [1] Magnusson, M., Riis Andersen, M., Jonasson, J., & Vehtari, A. *Bayesian Leave-One-Out
        Cross-Validation for Large Data.* Proceedings of the 36th International Conference on
        Machine Learning, PMLR 97:4244–4253 (2019)
        https://proceedings.mlr.press/v97/magnusson19a.html
        arXiv preprint https://arxiv.org/abs/1904.10679
    """
    if observations is None or (isinstance(observations, int) and observations == 0):
        return loo_orig
    if loo_orig.elpd_i is None:
        raise ValueError("Original loo_subsample result must have pointwise=True")
    if method not in ["lpd", "plpd"]:
        raise ValueError("Method must be either 'lpd' or 'plpd'")
    if method == "plpd" and log_lik_fn is None:
        raise ValueError("log_lik_fn must be provided when method='plpd'")

    thin = getattr(loo_orig, "thin_factor", None)
    loo_inputs = _prepare_loo_inputs(data, var_name, thin)
    update_data = _prepare_update_subsample(
        loo_orig, data, observations, var_name, seed, method, log_lik_fn, param_names, log
    )

    if reff is None:
        reff = _get_r_eff(data, loo_inputs.n_samples)

    # Get log densities from original ELPD data if they exist
    log_p = getattr(loo_orig, "log_p", None)
    log_q = getattr(loo_orig, "log_q", None)

    log_weights_new = None
    if log_weights is None:
        log_weights = getattr(loo_orig, "log_weights", None)

    if log_weights is not None:
        if isinstance(log_weights, ELPDData):
            if log_weights.log_weights is None:
                raise ValueError("ELPDData object does not contain log_weights")
            log_weights = log_weights.log_weights
            if loo_inputs.var_name in log_weights:
                log_weights = log_weights[loo_inputs.var_name]

        if len(loo_inputs.obs_dims) > 1:
            stacked_obs_dim = "__obs__"
            log_weights_stacked = log_weights.stack({stacked_obs_dim: loo_inputs.obs_dims})
            log_weights_new = _select_obs_by_indices(
                log_weights_stacked, update_data.new_indices, [stacked_obs_dim], stacked_obs_dim
            )
            log_weights_new = log_weights_new.unstack(stacked_obs_dim)
        else:
            obs_dim = loo_inputs.obs_dims[0]
            log_weights_new = _select_obs_by_indices(
                log_weights, update_data.new_indices, loo_inputs.obs_dims, obs_dim
            )

    if log_weights_new is None:
        log_weights_new_ds, _ = update_data.log_likelihood_new.azstats.psislw(
            r_eff=reff, dim=loo_inputs.sample_dims
        )
        log_weights_new = log_weights_new_ds[loo_inputs.var_name]

    elpd_loo_i_new_da, pareto_k_new_da, approx_posterior = _compute_loo_results(
        log_likelihood=update_data.log_likelihood_new,
        var_name=loo_inputs.var_name,
        sample_dims=loo_inputs.sample_dims,
        n_samples=loo_inputs.n_samples,
        n_data_points=len(update_data.new_indices),
        log_weights=log_weights_new,
        reff=reff,
        log_p=log_p,
        log_q=log_q,
        return_pointwise=True,
    )

    combined_elpd_i_da = xr.concat(
        [update_data.old_elpd_i, elpd_loo_i_new_da], dim=update_data.concat_dim
    )
    combined_pareto_k_da = xr.concat(
        [update_data.old_pareto_k, pareto_k_new_da], dim=update_data.concat_dim
    )

    good_k = loo_orig.good_k
    warn_mg, _ = _warn_pareto_k(combined_pareto_k_da, loo_inputs.n_samples)

    lpd_approx_sample_da = _select_obs_by_coords(
        update_data.lpd_approx_all, combined_elpd_i_da, loo_inputs.obs_dims, "__obs__"
    )

    elpd_loo_hat, subsampling_se, se = _diff_srs_estimator(
        combined_elpd_i_da,
        lpd_approx_sample_da,
        update_data.lpd_approx_all,
        loo_inputs.n_data_points,
        update_data.combined_size,
    )

    # Calculate p_loo using SRS estimation directly on the p_loo values
    # from the subsample
    p_loo_sample = lpd_approx_sample_da - combined_elpd_i_da
    p_loo, _, _ = _srs_estimator(
        p_loo_sample,
        loo_inputs.n_data_points,
        update_data.combined_size,
    )

    combined_indices = np.concatenate((update_data.old_indices, update_data.new_indices))
    elpd_i_full, pareto_k_full = _prepare_full_arrays(
        combined_elpd_i_da,
        combined_pareto_k_da,
        update_data.lpd_approx_all,
        combined_indices,
        loo_inputs.obs_dims,
        elpd_loo_hat,
    )

    if loo_orig.log_weights is not None and log_weights_new is not None:
        old_log_weights = loo_orig.log_weights
        if isinstance(old_log_weights, xr.Dataset):
            old_log_weights = old_log_weights[loo_inputs.var_name]

        if isinstance(log_weights_new, xr.Dataset):
            log_weights_new = log_weights_new[loo_inputs.var_name]
        combined_log_weights = xr.concat(
            [old_log_weights, log_weights_new], dim=update_data.concat_dim
        )
        log_weights_full = xr.Dataset({loo_inputs.var_name: combined_log_weights})
    else:
        log_weights_full = None

    return ELPDData(
        "loo",
        elpd_loo_hat,
        se,
        p_loo,
        loo_inputs.n_samples,
        loo_inputs.n_data_points,
        "log",
        warn_mg,
        good_k,
        elpd_i_full,
        pareto_k_full,
        approx_posterior,
        subsampling_se,
        update_data.combined_size,
        log_p,
        log_q,
        thin,
        log_weights_full,
    )
