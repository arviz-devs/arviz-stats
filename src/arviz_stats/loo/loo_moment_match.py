"""Compute moment matching for problematic observations in PSIS-LOO-CV."""

import warnings
from collections import namedtuple
from copy import deepcopy

import numpy as np
import xarray as xr
from arviz_base import rcParams
from xarray_einstats.stats import logsumexp

from arviz_stats.loo.helper_loo import (
    _get_log_likelihood_i,
    _get_r_eff,
    _prepare_loo_inputs,
    _recalculate_weights_k,
    _shift,
    _shift_and_cov,
    _shift_and_scale,
    _update_loo_data_i,
)
from arviz_stats.sampling_diagnostics import ess
from arviz_stats.utils import ELPDData

SplitMomentMatch = namedtuple("SplitMomentMatch", ["lwi", "lwfi", "log_liki", "reff"])


def loo_moment_match(
    data,
    loo_orig,
    upars,
    log_prob_upars_fn,
    log_lik_i_upars_fn,
    var_name=None,
    reff=None,
    max_iters=30,
    k_threshold=None,
    split=True,
    cov=False,
    pointwise=None,
):
    r"""Compute moment matching for problematic observations in PSIS-LOO-CV.

    Adjusts the results of a previously computed Pareto smoothed importance sampling leave-one-out
    cross-validation (PSIS-LOO-CV) object by applying a moment matching algorithm to
    observations with high Pareto k diagnostic values. The moment matching algorithm iteratively
    adjusts the posterior draws in the unconstrained parameter space to better approximate the
    leave-one-out posterior.

    The moment matching algorithm is described in [1]_ and the PSIS-LOO-CV method is described in
    [2]_ and [3]_.

    Parameters
    ----------
    data : DataTree or InferenceData
        Input data. It should contain the posterior and the log_likelihood groups.
    loo_orig : ELPDData
        An existing ELPDData object from a previous `loo` result. Must contain
        pointwise Pareto k values (`pointwise=True` must have been used).
    upars : DataArray
        Posterior draws transformed to the unconstrained parameter space. Must have
        `chain` and `draw` dimensions, plus one additional dimension containing all
        parameters. Parameter names can be provided as coordinate values on this
        dimension.
    log_prob_upars_fn : Callable[[DataArray], DataArray]
        A function that takes the unconstrained parameter draws and returns a
        :class:`~xarray.DataArray` containing the log probability density of the full posterior
        distribution evaluated at each unconstrained parameter draw. The returned DataArray must
        have dimensions `chain`, `draw`.
    log_lik_i_upars_fn : Callable[[DataArray, int], DataArray]
        A function that takes the unconstrained parameter draws and the integer index `i`
        of the left-out observation. It should return a :class:`~xarray.DataArray` containing the
        log-likelihood of the left-out observation `i` evaluated at each unconstrained parameter
        draw. The returned DataArray must have dimensions `chain`, `draw`.
    var_name : str, optional
        The name of the variable in log_likelihood groups storing the pointwise log
        likelihood data to use for loo computation.
    reff: float, optional
        Relative MCMC efficiency, ``ess / n`` i.e. number of effective samples divided by the number
        of actual samples. Computed from trace by default.
    max_iters : int, default 30
        Maximum number of moment matching iterations for each problematic observation.
    k_threshold : float, optional
        Threshold value for Pareto k values above which moment matching is applied.
        Defaults to :math:`\min(1 - 1/\log_{10}(S), 0.7)`, where S is the number of samples.
    split : bool, default True
        If True, only transform half of the draws and use multiple importance sampling to combine
        them with untransformed draws.
    cov : bool, default False
        If True, match the covariance structure during the transformation, in addition
        to the mean and marginal variances. Ignored if ``split=False``.
    pointwise: bool, optional
        If True, the pointwise predictive accuracy will be returned. Defaults to
        ``rcParams["stats.ic_pointwise"]``. Moment matching always requires
        pointwise data from `loo_orig`. This argument controls whether the returned
        object includes pointwise data.

    Returns
    -------
    ELPDData
        Object with the following attributes:

        - **elpd**: expected log pointwise predictive density
        - **se**: standard error of the elpd
        - **p**: effective number of parameters
        - **n_samples**: number of samples
        - **n_data_points**: number of data points
        - **warning**: True if the estimated shape parameter of Pareto distribution is greater
          than ``good_k``.
        - **elp_i**: :class:`~xarray.DataArray` with the pointwise predictive accuracy, only if
          ``pointwise=True``
        - **pareto_k**: array of Pareto shape values, only if ``pointwise=True``
        - **good_k**: For a sample size S, the threshold is computed as
          ``min(1 - 1/log10(S), 0.7)``
        - **approx_posterior**: True if approximate posterior was used.

    Examples
    --------
    Moment matching can improve PSIS-LOO-CV estimates for observations with high Pareto k values
    without having to refit the model for each problematic observation. We will use the non-centered
    eight schools data which has 1 problematic observation:

    .. ipython::
        :okwarning:

        In [1]: import arviz_base as az
           ...: from arviz_stats import loo, loo_moment_match
           ...: import numpy as np
           ...: import xarray as xr
           ...: from scipy import stats
           ...:
           ...: idata = az.load_arviz_data("non_centered_eight")
           ...: loo_orig = loo(idata, pointwise=True, var_name="obs")
           ...: loo_orig

    For moment matching, we need the unconstrained parameters and two functions
    for the log probability and pointwise log-likelihood computations:

    .. ipython::

        In [3]: posterior = idata.posterior
           ...: theta_t = posterior.theta_t.values
           ...: mu = posterior.mu.values[:, :, np.newaxis]
           ...: log_tau = np.log(posterior.tau.values)[:, :, np.newaxis]
           ...:
           ...: upars = np.concatenate([theta_t, mu, log_tau], axis=2)
           ...: param_names = [f"theta_t_{i}" for i in range(8)] + ["mu", "log_tau"]
           ...:
           ...: upars = xr.DataArray(
           ...:     upars,
           ...:     dims=["chain", "draw", "upars_dim"],
           ...:     coords={
           ...:         "chain": posterior.chain,
           ...:         "draw": posterior.draw,
           ...:         "upars_dim": param_names
           ...:     }
           ...: )
           ...:
           ...: def log_prob_upars(upars):
           ...:     theta_tilde = upars.sel(upars_dim=[f"theta_t_{i}" for i in range(8)])
           ...:     mu = upars.sel(upars_dim="mu")
           ...:     log_tau = upars.sel(upars_dim="log_tau")
           ...:     tau = np.exp(log_tau)
           ...:
           ...:     log_prob = stats.norm(0, 5).logpdf(mu.values)
           ...:     log_prob += stats.halfcauchy(0, 5).logpdf(tau.values)
           ...:     log_prob += log_tau.values
           ...:     log_prob += stats.norm(0, 1).logpdf(theta_tilde.values).sum(axis=-1)
           ...:
           ...:     return xr.DataArray(
           ...:         log_prob,
           ...:         dims=["chain", "draw"],
           ...:         coords={"chain": upars.chain, "draw": upars.draw}
           ...:     )
           ...:
           ...: def log_lik_i_upars(upars, i):
           ...:     sigmas = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]
           ...:     theta_tilde_i = upars.sel(upars_dim=f"theta_t_{i}")
           ...:     mu = upars.sel(upars_dim="mu")
           ...:     tau = np.exp(upars.sel(upars_dim="log_tau"))
           ...:     theta_i = mu + tau * theta_tilde_i
           ...:     y_i = idata.observed_data.obs.values[i]
           ...:     log_lik = stats.norm(theta_i.values, sigmas[i]).logpdf(y_i)
           ...:
           ...:     return xr.DataArray(
           ...:         log_lik,
           ...:         dims=["chain", "draw"],
           ...:         coords={"chain": upars.chain, "draw": upars.draw}
           ...:     )

    We can now apply moment matching using the split transformation and covariance matching.
    We can see that all Pareto :math:`k` values are now below the threshold and the ELPD is slightly
    improved:

    .. ipython::
        :okwarning:

        In [4]: loo_mm = loo_moment_match(
           ...:     idata,
           ...:     loo_orig,
           ...:     upars=upars,
           ...:     log_prob_upars_fn=log_prob_upars,
           ...:     log_lik_i_upars_fn=log_lik_i_upars,
           ...:     var_name="obs",
           ...:     k_threshold=0.7,
           ...:     split=True,
           ...:     cov=True,
           ...: )
           ...: loo_mm

    Notes
    -----
    The moment matching algorithm considers three affine transformations of the posterior draws.
    For a specific draw :math:`\theta^{(s)}`, a generic affine transformation includes a square
    matrix :math:`\mathbf{A}` representing a linear map and a vector :math:`\mathbf{b}`
    representing a translation such that

    .. math::
        T : \theta^{(s)} \mapsto \mathbf{A}\theta^{(s)} + \mathbf{b}
        =: \theta^{*{(s)}}.

    The first transformation, :math:`T_1`, is a translation that matches the mean of the sample
    to its importance weighted mean given by

    .. math::
        \mathbf{\theta^{*{(s)}}} = T_1(\mathbf{\theta^{(s)}}) =
        \mathbf{\theta^{(s)}} - \bar{\theta} + \bar{\theta}_w,

    where :math:`\bar{\theta}` is the mean of the sample and :math:`\bar{\theta}_w` is the
    importance weighted mean of the sample. The second transformation, :math:`T_2`, is a scaling
    that matches the marginal variances in addition to the means given by

    .. math::
        \mathbf{\theta^{*{(s)}}} = T_2(\mathbf{\theta^{(s)}}) =
        \mathbf{v}^{1/2}_w \circ \mathbf{v}^{-1/2} \circ (\mathbf{\theta^{(s)}} - \bar{\theta}) +
        \bar{\theta}_w,

    where :math:`\mathbf{v}` and :math:`\mathbf{v}_w` are the sample and weighted variances, and
    :math:`\circ` denotes the pointwise product of the elements of two vectors. The third
    transformation, :math:`T_3`, is a covariance transformation that matches the covariance matrix
    of the sample to its importance weighted covariance matrix given by

    .. math::
        \mathbf{\theta^{*{(s)}}} = T_3(\mathbf{\theta^{(s)}}) =
        \mathbf{L}_w \mathbf{L}^{-1} (\mathbf{\theta^{(s)}} - \bar{\theta}) + \bar{\theta}_w,

    where :math:`\mathbf{L}` and :math:`\mathbf{L}_w` are the Cholesky decompositions of the
    covariance matrix and the weighted covariance matrix, respectively, e.g.,

    .. math::
        \mathbf{LL}^T = \mathbf{\Sigma} = \frac{1}{S} \sum_{s=1}^S (\mathbf{\theta^{(s)}} -
        \bar{\theta}) (\mathbf{\theta^{(s)}} - \bar{\theta})^T

    and

    .. math::
        \mathbf{L}_w \mathbf{L}_w^T = \mathbf{\Sigma}_w = \frac{\frac{1}{S} \sum_{s=1}^S
        w^{(s)} (\mathbf{\theta^{(s)}} - \bar{\theta}_w) (\mathbf{\theta^{(s)}} -
        \bar{\theta}_w)^T}{\sum_{s=1}^S w^{(s)}}.

    We iterate on :math:`T_1` repeatedly and move onto :math:`T_2` and :math:`T_3` only
    if :math:`T_1` fails to yield a Pareto-k statistic below the threshold.

    See Also
    --------
    loo : Standard PSIS-LOO-CV.
    loo_approximate_posterior : Approximate posterior PSIS-LOO-CV.
    loo_subsample : Sub-sampled PSIS-LOO-CV.

    References
    ----------

    .. [1] Paananen, T., Piironen, J., Buerkner, P.-C., Vehtari, A. (2021). Implicitly Adaptive
        Importance Sampling. Statistics and Computing. 31(2) (2021)
        https://doi.org/10.1007/s11222-020-09982-2
        arXiv preprint https://arxiv.org/abs/1906.08850.
    .. [2] Vehtari et al. *Practical Bayesian model evaluation using leave-one-out cross-validation
        and WAIC*. Statistics and Computing. 27(5) (2017) https://doi.org/10.1007/s11222-016-9696-4
        arXiv preprint https://arxiv.org/abs/1507.04544.
    .. [3] Vehtari et al. *Pareto Smoothed Importance Sampling*.
        Journal of Machine Learning Research, 25(72) (2024) https://jmlr.org/papers/v25/19-556.html
        arXiv preprint https://arxiv.org/abs/1507.02646
    """
    if not isinstance(loo_orig, ELPDData):
        raise TypeError("loo_orig must be an ELPDData object.")
    if loo_orig.pareto_k is None or loo_orig.elpd_i is None:
        raise ValueError(
            "Moment matching requires pointwise LOO results with Pareto k values. "
            "Please compute the initial LOO with pointwise=True."
        )

    sample_dims = ["chain", "draw"]

    if not isinstance(upars, xr.DataArray):
        raise TypeError("upars must be a DataArray.")
    if not all(dim_name in upars.dims for dim_name in sample_dims):
        raise ValueError(f"upars must have dimensions {sample_dims}.")

    param_dim_list = [dim for dim in upars.dims if dim not in sample_dims]

    if len(param_dim_list) == 0:
        param_dim_name = "upars_dim"
        upars = upars.expand_dims(dim={param_dim_name: 1})
    elif len(param_dim_list) == 1:
        param_dim_name = param_dim_list[0]
    else:
        raise ValueError("upars must have at most one dimension besides 'chain' and 'draw'.")

    loo_data = deepcopy(loo_orig)
    loo_data.method = "loo_moment_match"
    pointwise = rcParams["stats.ic_pointwise"] if pointwise is None else pointwise

    loo_inputs = _prepare_loo_inputs(data, var_name)
    log_likelihood = loo_inputs.log_likelihood
    obs_dims = loo_inputs.obs_dims
    n_samples = loo_inputs.n_samples
    var_name = loo_inputs.var_name
    n_params = upars.sizes[param_dim_name]

    if reff is None:
        reff = _get_r_eff(data, n_samples)

    try:
        orig_log_prob = log_prob_upars_fn(upars)
        if not isinstance(orig_log_prob, xr.DataArray):
            raise TypeError("log_prob_upars_fn must return a DataArray.")
        if not all(dim in orig_log_prob.dims for dim in sample_dims):
            raise ValueError(f"Original log probability must have dimensions {sample_dims}.")
        if len(orig_log_prob.dims) != len(sample_dims):
            raise ValueError(
                f"Original log probability should only have dimensions {sample_dims}, "
                f"found {orig_log_prob.dims}"
            )
    except Exception as e:
        raise ValueError(f"Error executing log_prob_upars_fn: {e}") from e

    if k_threshold is None:
        k_threshold = min(1 - 1 / np.log10(n_samples), 0.7) if n_samples > 1 else 0.7

    ks = loo_data.pareto_k.stack(__obs__=obs_dims).transpose("__obs__").values
    bad_obs_indices = np.where(ks > k_threshold)[0]

    if len(bad_obs_indices) == 0:
        warnings.warn("No Pareto k values exceed the threshold. Returning original LOO data.")
        if not pointwise:
            loo_data.elpd_i = None
            loo_data.pareto_k = None
            if hasattr(loo_data, "p_loo_i"):
                loo_data.p_loo_i = None
        return loo_data

    loo_data.p_loo_i = xr.full_like(loo_data.elpd_i, np.nan)

    # Moment matching algorithm
    for i in bad_obs_indices:
        log_liki = _get_log_likelihood_i(log_likelihood, i, obs_dims)
        log_ratio_i_init = -log_liki
        lwi, ki_tuple = log_ratio_i_init.azstats.psislw(r_eff=reff, dim=sample_dims)

        ki = ki_tuple[0].item() if isinstance(ki_tuple, tuple) else ki_tuple.item()
        ess_val = ess(log_liki.values.reshape(-1, 1), method="mean").item()
        reff_i = ess_val / n_samples if n_samples > 0 else 1.0

        upars_i = upars.copy(deep=True)
        total_shift = np.zeros(upars_i.sizes[param_dim_name])
        total_scaling = np.ones(upars_i.sizes[param_dim_name])
        total_mapping = np.eye(upars_i.sizes[param_dim_name])

        iterind = 1
        transformations_applied = False

        while ki > k_threshold:
            if iterind > max_iters:
                warnings.warn(
                    f"Maximum number of moment matching iterations ({max_iters}) reached "
                    f"for observation {i}. Final Pareto k is {ki:.2f}.",
                    UserWarning,
                    stacklevel=2,
                )
                break

            # Try Mean Shift
            try:
                shift_res = _shift(upars_i, lwi)
                log_prob_shifted = log_prob_upars_fn(shift_res.upars)
                log_liki_shifted = log_lik_i_upars_fn(shift_res.upars, i)
                weights_k_res = _recalculate_weights_k(
                    log_liki_shifted, log_prob_shifted, orig_log_prob, reff_i, sample_dims
                )
                if weights_k_res.ki < ki:
                    ki = weights_k_res.ki
                    lwi = weights_k_res.lwi
                    log_liki = weights_k_res.log_liki
                    upars_i = shift_res.upars
                    total_shift += shift_res.shift
                    transformations_applied = True
                    iterind += 1
                    continue  # Restart, try mean shift again

            except RuntimeError as e:
                warnings.warn(
                    f"Error during mean shift calculation for observation {i}: {e}. "
                    "Stopping moment matching for this observation.",
                    UserWarning,
                    stacklevel=2,
                )
                break

            # Try Scale Shift (only if mean shift didn't improve)
            try:
                scale_res = _shift_and_scale(upars_i, lwi)
                log_prob_scaled = log_prob_upars_fn(scale_res.upars)
                log_liki_scaled = log_lik_i_upars_fn(scale_res.upars, i)
                weights_k_res_scale = _recalculate_weights_k(
                    log_liki_scaled, log_prob_scaled, orig_log_prob, reff_i, sample_dims
                )
                if weights_k_res_scale.ki < ki:
                    ki = weights_k_res_scale.ki
                    lwi = weights_k_res_scale.lwi
                    log_liki = weights_k_res_scale.log_liki
                    upars_i = scale_res.upars
                    total_shift = scale_res.shift + total_shift * scale_res.scaling
                    total_scaling *= scale_res.scaling
                    transformations_applied = True
                    iterind += 1
                    continue  # Restart, try mean shift again

            except RuntimeError as e:
                warnings.warn(
                    f"Error during scale shift calculation for observation {i}: {e}. "
                    "Stopping moment matching for this observation.",
                    UserWarning,
                    stacklevel=2,
                )
                break

            # Try Covariance Shift (only if mean and scale shift didn't improve, cov=True,
            # and S >= 10 * npars)
            if cov and n_samples >= 10 * n_params:
                try:
                    cov_res = _shift_and_cov(upars_i, lwi)
                    log_prob_cov = log_prob_upars_fn(cov_res.upars)
                    log_liki_cov = log_lik_i_upars_fn(cov_res.upars, i)
                    weights_k_res_cov = _recalculate_weights_k(
                        log_liki_cov, log_prob_cov, orig_log_prob, reff_i, sample_dims
                    )
                    if weights_k_res_cov.ki < ki:
                        ki = weights_k_res_cov.ki
                        lwi = weights_k_res_cov.lwi
                        log_liki = weights_k_res_cov.log_liki
                        upars_i = cov_res.upars
                        total_shift = cov_res.shift + total_shift @ cov_res.mapping.T
                        total_mapping = cov_res.mapping @ total_mapping
                        transformations_applied = True
                        iterind += 1
                        continue  # Restart, try mean shift again

                except RuntimeError as e:
                    warnings.warn(
                        f"Error during covariance shift calculation for observation {i}: {e}. "
                        "Stopping moment matching for this observation.",
                        UserWarning,
                        stacklevel=2,
                    )
                    break

            # If none of the transformations in this pass improved ki, break.
            break

        if split and transformations_applied:
            try:
                split_res = _split_moment_match(
                    upars=upars,
                    cov=cov,
                    total_shift=total_shift,
                    total_scaling=total_scaling,
                    total_mapping=total_mapping,
                    i=i,
                    reff=reff_i,
                    log_prob_upars_fn=log_prob_upars_fn,
                    log_lik_i_upars_fn=log_lik_i_upars_fn,
                )

                final_log_liki = split_res.log_liki
                final_lwi = split_res.lwi
                _, ki_split_tuple = split_res.lwi.azstats.psislw(
                    r_eff=split_res.reff, dim=sample_dims
                )
                ki_split = (
                    ki_split_tuple[0].item()
                    if isinstance(ki_split_tuple, tuple)
                    else ki_split_tuple.item()
                )
                final_ki = ki_split

                if ki_split > ki and ki <= k_threshold:
                    warnings.warn(
                        f"Split transformation increased Pareto k for observation {i} "
                        f"({ki:.2f} -> {ki_split:.2f}). This may indicate numerical issues.",
                        UserWarning,
                        stacklevel=2,
                    )

            except RuntimeError as e:
                warnings.warn(
                    f"Error during split moment matching for observation {i}: {e}. "
                    "Using non-split transformation result.",
                    UserWarning,
                    stacklevel=2,
                )
                # On error, keep the non-split results
                final_log_liki = log_liki
                final_lwi = lwi
                final_ki = ki
        else:
            # No split transformation
            final_log_liki = log_liki
            final_lwi = lwi
            final_ki = ki

        new_elpd_i = logsumexp(final_log_liki + final_lwi, dims=sample_dims).item()
        original_log_liki = _get_log_likelihood_i(log_likelihood, i, obs_dims)

        _update_loo_data_i(
            loo_data,
            i,
            new_elpd_i,
            final_ki,
            final_log_liki,
            sample_dims,
            obs_dims,
            n_samples,
            original_log_liki=original_log_liki,
        )

    final_ks = loo_data.pareto_k.stack(__obs__=obs_dims).transpose("__obs__").values
    if np.any(final_ks[bad_obs_indices] > k_threshold):
        warnings.warn(
            f"After Moment Matching, {np.sum(final_ks > k_threshold)} observations still have "
            f"Pareto k > {k_threshold:.2f}.",
            UserWarning,
            stacklevel=2,
        )

    # p_loo for good observations
    if hasattr(loo_data, "p_loo_i") and loo_data.p_loo_i is not None:
        p_loo_i_flat = loo_data.p_loo_i.values.flatten()
        nan_mask = np.isnan(p_loo_i_flat)

        if np.any(nan_mask):
            lpd_all = logsumexp(log_likelihood, b=1 / n_samples, dims=sample_dims)
            lpd_flat = lpd_all.values.flatten()
            elpd_i_flat = loo_data.elpd_i.values.flatten()

            p_loo_i_flat[nan_mask] = lpd_flat[nan_mask] - elpd_i_flat[nan_mask]
            loo_data.p_loo_i.values = p_loo_i_flat.reshape(loo_data.p_loo_i.shape)

        loo_data.p = np.nansum(loo_data.p_loo_i.values)
    else:
        elpd_raw = logsumexp(log_likelihood, b=1 / n_samples, dims=sample_dims).sum().values
        loo_data.p = elpd_raw - loo_data.elpd

    if not pointwise:
        loo_data.elpd_i = None
        loo_data.pareto_k = None
        if hasattr(loo_data, "p_loo_i"):
            loo_data.p_loo_i = None
    return loo_data


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
    lwi_psis_da, _ = raw_log_weights_half.azstats.psislw(r_eff=reff, dim=sample_dims)

    lr_full = lwi_psis_da + log_liki_half
    lr_full = xr.where(np.isnan(lr_full) | (np.isinf(lr_full) & (lr_full > 0)), -np.inf, lr_full)

    # PSIS smoothing for full posterior
    lwfi_psis_da, _ = lr_full.azstats.psislw(r_eff=reff, dim=sample_dims)

    return SplitMomentMatch(
        lwi=lwi_psis_da,
        lwfi=lwfi_psis_da,
        log_liki=log_liki_half,
        reff=reff,
    )
