"""Compute PSIS-LOO-CV for approximate posteriors."""

from arviz_base import rcParams

from arviz_stats.loo.helper_loo import (
    _check_log_density,
    _compute_loo_results,
    _prepare_loo_inputs,
)


def loo_approximate_posterior(data, log_p, log_q, pointwise=None, var_name=None):
    """Compute PSIS-LOO-CV for approximate posteriors.

    Estimates the expected log pointwise predictive density (elpd) using Pareto-smoothed
    importance sampling leave-one-out cross-validation (PSIS-LOO-CV) for approximate
    posteriors (e.g., from variational inference). Requires log-densities of the target (log_p)
    and proposal (log_q) distributions.

    The PSIS-LOO-CV method is described in [1]_ and [2]_. The approximate posterior correction
    is computed using the method described in [3]_.

    Parameters
    ----------
    data : DataTree or InferenceData
        Input data. It should contain the log_likelihood group corresponding to samples
        drawn from the proposal distribution (q).
    log_p : ndarray or DataArray
        The (target) log-density evaluated at S samples from the target distribution (p).
        If ndarray, should be a vector of length S where S is the number of samples.
        If DataArray, should have dimensions matching the sample dimensions
        ("chain", "draw").
    log_q : ndarray or DataArray
        The (proposal) log-density evaluated at S samples from the proposal distribution (q).
        If ndarray, should be a vector of length S where S is the number of samples.
        If DataArray, should have dimensions matching the sample dimensions
        ("chain", "draw").
    pointwise : bool, optional
        If True, returns pointwise values. Defaults to rcParams["stats.ic_pointwise"].
    var_name : str, optional
        The name of the variable in log_likelihood groups storing the pointwise log
        likelihood data to use for loo computation.

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
        - **elpd_i**: :class:`~xarray.DataArray` with the pointwise predictive accuracy, only if
          ``pointwise=True``
        - **pareto_k**: array of Pareto shape values, only if ``pointwise=True``
        - **good_k**: For a sample size S, the threshold is computed as
          ``min(1 - 1/log10(S), 0.7)``
        - **approx_posterior**: True if approximate posterior was used.

    Examples
    --------
    Calculate LOO for posterior approximations. The following example is intentionally minimal
    to demonstrate basic usage. The approximate posterior created below may not accurately
    represent the data and lead to less meaningful LOO results.

    Create dummy log-densities:

    .. ipython::

        In [1]: import numpy as np
           ...: import xarray as xr
           ...: from arviz_stats import loo_approximate_posterior
           ...: from arviz_base import load_arviz_data, extract
           ...:
           ...: data = load_arviz_data("centered_eight")
           ...: log_lik = extract(data, group="log_likelihood", var_names="obs", combined=False)
           ...: rng = np.random.default_rng(214)
           ...:
           ...: values_p = rng.normal(loc=0, scale=1, size=(log_lik.chain.size, log_lik.draw.size))
           ...: log_p = xr.DataArray(
           ...:     values_p,
           ...:     dims=["chain", "draw"],
           ...:     coords={"chain": log_lik.chain, "draw": log_lik.draw}
           ...: )
           ...:
           ...: values_q = rng.normal(loc=-1, scale=1, size=(log_lik.chain.size, log_lik.draw.size))
           ...: log_q = xr.DataArray(
           ...:     values_q,
           ...:     dims=["chain", "draw"],
           ...:     coords={"chain": log_lik.chain, "draw": log_lik.draw}
           ...: )

    Calculate approximate pointwise LOO:

    .. ipython::

        In [2]: loo_approx = loo_approximate_posterior(
           ...:     data,
           ...:     log_p=log_p,
           ...:     log_q=log_q,
           ...:     var_name="obs",
           ...:     pointwise=True
           ...: )
           ...: loo_approx

    See Also
    --------
    loo : Standard PSIS-LOO-CV.
    loo_subsample : Sub-sampled PSIS-LOO-CV.
    compare : Compare models based on their ELPD.

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
    loo_inputs = _prepare_loo_inputs(data, var_name)
    pointwise = rcParams["stats.ic_pointwise"] if pointwise is None else pointwise

    log_likelihood = loo_inputs.log_likelihood
    log_p = _check_log_density(
        log_p, "log_p", log_likelihood, loo_inputs.n_samples, loo_inputs.sample_dims
    )
    log_q = _check_log_density(
        log_q, "log_q", log_likelihood, loo_inputs.n_samples, loo_inputs.sample_dims
    )

    approx_correction = log_p - log_q

    # Handle underflow/overflow
    approx_correction = approx_correction - approx_correction.max()

    corrected_log_ratios = -log_likelihood.copy()
    corrected_log_ratios = corrected_log_ratios + approx_correction

    # Handle underflow/overflow
    log_ratio_max = corrected_log_ratios.max(dim=loo_inputs.sample_dims)
    corrected_log_ratios = corrected_log_ratios - log_ratio_max

    # ignore r_eff here, set to r_eff=1.0
    log_weights, pareto_k = corrected_log_ratios.azstats.psislw(
        r_eff=1.0, dims=loo_inputs.sample_dims
    )

    return _compute_loo_results(
        log_likelihood=loo_inputs.log_likelihood,
        var_name=loo_inputs.var_name,
        pointwise=pointwise,
        sample_dims=loo_inputs.sample_dims,
        n_samples=loo_inputs.n_samples,
        n_data_points=loo_inputs.n_data_points,
        log_weights=log_weights,
        pareto_k=pareto_k,
        approx_posterior=True,
    )
