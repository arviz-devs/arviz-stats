"""Pareto-smoothed importance sampling LOO (PSIS-LOO-CV) related functions."""

import itertools
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import xarray as xr
from arviz_base import convert_to_datatree, extract, rcParams
from scipy.optimize import minimize
from scipy.stats import dirichlet
from xarray_einstats.stats import logsumexp

from arviz_stats.helper_loo import (
    _check_log_density,
    _diff_srs_estimator,
    _get_log_likelihood_i,
    _get_r_eff,
    _prepare_full_arrays,
    _prepare_loo_inputs,
    _prepare_subsample,
    _prepare_update_subsample,
    _recalculate_weights_k,
    _select_obs_by_coords,
    _shift,
    _shift_and_cov,
    _shift_and_scale,
    _split_moment_match,
    _srs_estimator,
    _update_loo_data_i,
    _warn_pareto_k,
    _warn_pointwise_loo,
)
from arviz_stats.metrics import _metrics
from arviz_stats.sampling_diagnostics import ess
from arviz_stats.utils import ELPDData, get_log_likelihood_dataset


def loo(data, pointwise=None, var_name=None, reff=None):
    """Compute Pareto-smoothed importance sampling leave-one-out cross-validation (PSIS-LOO-CV).

    Estimates the expected log pointwise predictive density (elpd) using Pareto-smoothed
    importance sampling leave-one-out cross-validation (PSIS-LOO-CV). Also calculates LOO's
    standard error and the effective number of parameters. The method is described in [1]_
    and [2]_.

    Parameters
    ----------
    data : DataTree or InferenceData
        Input data. It should contain the posterior and the log_likelihood groups.
    pointwise: bool, optional
        If True the pointwise predictive accuracy will be returned. Defaults to
        ``rcParams["stats.ic_pointwise"]``.
    var_name : str, optional
        The name of the variable in log_likelihood groups storing the pointwise log
        likelihood data to use for loo computation.
    reff: float, optional
        Relative MCMC efficiency, ``ess / n`` i.e. number of effective samples divided by the number
        of actual samples. Computed from trace by default.

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
    Calculate LOO of a model:

    .. ipython::

        In [1]: from arviz_stats import loo
           ...: from arviz_base import load_arviz_data
           ...: data = load_arviz_data("centered_eight")
           ...: loo_data = loo(data)
           ...: loo_data

    Return the pointwise values:

    .. ipython::

        In [2]: loo_data.elpd_i

    See Also
    --------
    :func:`compare` : Compare models based on their ELPD.
    :func:`arviz_plots.plot_compare`: Summary plot for model comparison.

    References
    ----------

    .. [1] Vehtari et al. *Practical Bayesian model evaluation using leave-one-out cross-validation
        and WAIC*. Statistics and Computing. 27(5) (2017) https://doi.org/10.1007/s11222-016-9696-4
        arXiv preprint https://arxiv.org/abs/1507.04544.

    .. [2] Vehtari et al. *Pareto Smoothed Importance Sampling*.
        Journal of Machine Learning Research, 25(72) (2024) https://jmlr.org/papers/v25/19-556.html
        arXiv preprint https://arxiv.org/abs/1507.02646
    """
    loo_inputs = _prepare_loo_inputs(data, var_name)
    pointwise = rcParams["stats.ic_pointwise"] if pointwise is None else pointwise

    if reff is None:
        reff = _get_r_eff(data, loo_inputs.n_samples)

    log_weights, pareto_k = loo_inputs.log_likelihood.azstats.psislw(
        r_eff=reff, dims=loo_inputs.sample_dims
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
        approx_posterior=False,
    )


def loo_expectations(data, var_name=None, kind="mean", probs=None):
    """
    Compute weighted expectations using the PSIS-LOO-CV method.

    The expectations assume that the PSIS approximation is working well.
    The PSIS-LOO-CV method is described in [1]_ and [2]_.

    Parameters
    ----------
    data: DataTree or InferenceData
        It should contain the groups `posterior_predictive` and `log_likelihood`.
    var_name: str, optional
        The name of the variable in log_likelihood groups storing the pointwise log
        likelihood data to use for loo computation.
    kind: str, optional
        The kind of expectation to compute. Available options are:

        - 'mean': the mean of the posterior predictive distribution. Default.
        - 'median': the median of the posterior predictive distribution.
        - 'var': the variance of the posterior predictive distribution.
        - 'sd': the standard deviation of the posterior predictive distribution.
        - 'quantile': the quantile of the posterior predictive distribution.
    probs: float or list of float, optional
        The quantile(s) to compute when kind is 'quantile'.

    Returns
    -------
    loo_expec : DataArray
        The weighted expectations.

    Examples
    --------
    Calculate predictive 0.25 and 0.75 quantiles

    .. ipython::

        In [1]: from arviz_stats import loo_expectations
           ...: from arviz_base import load_arviz_data
           ...: dt = load_arviz_data("radon")
           ...: loo_expectations(dt, kind="quantile", probs=[0.25, 0.75])

    References
    ----------

    .. [1] Vehtari et al. *Practical Bayesian model evaluation using leave-one-out cross-validation
        and WAIC*. Statistics and Computing. 27(5) (2017) https://doi.org/10.1007/s11222-016-9696-4
        arXiv preprint https://arxiv.org/abs/1507.04544.

    .. [2] Vehtari et al. *Pareto Smoothed Importance Sampling*.
        Journal of Machine Learning Research, 25(72) (2024) https://jmlr.org/papers/v25/19-556.html
        arXiv preprint https://arxiv.org/abs/1507.02646
    """
    if kind not in ["mean", "median", "var", "sd", "quantile"]:
        raise ValueError("kind must be either 'mean', 'median', 'var', 'sd' or 'quantile'")

    if kind == "quantile" and probs is None:
        raise ValueError("probs must be provided when kind is 'quantile'")

    dims = ("chain", "draw")
    if var_name is None:
        var_name = list(data.observed_data.data_vars.keys())[0]

    # Should we store the log_weights in the datatree when computing LOO?
    # Then we should be able to use the same log_weights for different variables

    data = convert_to_datatree(data)

    log_likelihood = get_log_likelihood_dataset(data, var_names=var_name)
    log_weights, _ = log_likelihood.azstats.psislw()
    weights = np.exp(log_weights)

    weighted_predictions = extract(
        data, group="posterior_predictive", var_names=var_name, combined=False
    ).weighted(weights[var_name])

    if kind == "mean":
        loo_expec = weighted_predictions.mean(dim=dims)

    elif kind == "median":
        loo_expec = weighted_predictions.quantile([0.5], dim=dims)

    elif kind == "var":
        # We use a Bessel's like correction term
        # instead of n/(n-1) we use ESS/(ESS-1)
        # where ESS/(ESS-1) = 1/(1-sum(weights**2))
        loo_expec = weighted_predictions.var(dim=dims) / (1 - np.sum(weights**2))

    elif kind == "sd":
        loo_expec = (weighted_predictions.var(dim=dims) / (1 - np.sum(weights**2))) ** 0.5

    else:
        loo_expec = weighted_predictions.quantile(probs, dim=dims)

    # Computation of specific khat should go here
    # log_ratios = -log_likelihood
    # khat = get_khat(loo_exp, ...)

    return loo_expec  # , khat


def loo_metrics(data, kind="rmse", var_name=None, round_to="2g"):
    """
    Compute predictive metrics using the PSIS-LOO-CV method.

    Currently supported metrics are mean absolute error, mean squared error and
    root mean squared error.
    For classification problems, accuracy and balanced accuracy are also supported.

    The PSIS-LOO-CV method is described in [1]_ and [2]_.

    Parameters
    ----------
    data: DataTree or InferenceData
        It should contain groups `observed_data`, `posterior_predictive` and  `log_likelihood`.
    kind: str
        The kind of metric to compute. Available options are:

        - 'mae': mean absolute error.
        - 'mse': mean squared error.
        - 'rmse': root mean squared error. Default.
        - 'acc': classification accuracy.
        - 'acc_balanced': balanced classification accuracy.

    var_name: str, optional
        The name of the variable in log_likelihood groups storing the pointwise log
        likelihood data to use for loo computation.
    round_to: int or str, optional
        If integer, number of decimal places to round the result. If string of the
        form '2g' number of significant digits to round the result. Defaults to '2g'.

    Returns
    -------
    estimate: namedtuple
        A namedtuple with the mean of the computed metric and its standard error.

    Examples
    --------
    Calculate predictive root mean squared error

    .. ipython::

        In [1]: from arviz_stats import loo_metrics
           ...: from arviz_base import load_arviz_data
           ...: dt = load_arviz_data("radon")
           ...: loo_metrics(dt, kind="rmse")

    Calculate accuracy of a logistic regression model

    .. ipython::

        In [1]: dt = load_arviz_data("anes")
           ...: loo_metrics(dt, kind="acc")

    References
    ----------

    .. [1] Vehtari et al. *Practical Bayesian model evaluation using leave-one-out cross-validation
        and WAIC*. Statistics and Computing. 27(5) (2017) https://doi.org/10.1007/s11222-016-9696-4
        arXiv preprint https://arxiv.org/abs/1507.04544.

    .. [2] Vehtari et al. *Pareto Smoothed Importance Sampling*.
        Journal of Machine Learning Research, 25(72) (2024) https://jmlr.org/papers/v25/19-556.html
        arXiv preprint https://arxiv.org/abs/1507.02646
    """
    if var_name is None:
        var_name = list(data.observed_data.data_vars.keys())[0]

    observed = data.observed_data[var_name]
    predicted = loo_expectations(data, kind="mean", var_name=var_name)

    return _metrics(observed, predicted, kind, round_to)


def loo_pit(
    data,
    var_names=None,
    log_weights=None,
):
    r"""Compute leave one out (PSIS-LOO) probability integral transform (PIT) values.

    The LOO-PIT values are :math:`p(\tilde{y}_i \le y_i \mid y_{-i})`, where :math:`y_i`
    represents the observed data for index :math:`i` and :math:`\tilde y_i` represents the
    posterior predictive sample at index :math:`i`. Note that :math:`y_{-i}` indicates we have
    left out the :math:`i`-th observation. LOO-PIT values are computed using the PSIS-LOO-CV
    method described in [1]_ and [2]_.

    Parameters
    ----------
    data : DataTree or InferenceData
        It should contain posterior, posterior_predictive and log_likelihood groups.
    var_names : str or list of str, optional
        Names of the variables to be used to compute the LOO-PIT values. If None, all
        variables are used. The function assumes that the observed and log_likelihood
        variables share the same names.
    log_weights: DataArray
        Smoothed log_weights. It must have the same shape as ``y_pred``
        Defaults to None, it will be computed using the PSIS-LOO method.

    Returns
    -------
    loo_pit: array or DataArray
        Value of the LOO-PIT at each observed data point.

    Examples
    --------
    Calculate LOO-PIT values using as test quantity the observed values themselves.

    .. ipython::

        In [1]: from arviz_stats import loo_pit
           ...: from arviz_base import load_arviz_data
           ...: dt = load_arviz_data("centered_eight")
           ...: loo_pit(dt)

    Calculate LOO-PIT values using as test quantity the square of the difference between
    each observation and `mu`. For this we create a new DataTree, copying the posterior and
    log_likelihood groups and creating new observed and posterior_predictive groups.

    .. ipython::

        In [1]: from arviz_base import from_dict
           ...: new_dt = from_dict({"posterior": dt.posterior,
           ...:                 "log_likelihood": dt.log_likelihood,
           ...:                 "observed_data": {
           ...:                     "obs": (dt.observed_data.obs
           ...:                            - dt.posterior.mu.median(dim=("chain", "draw")))**2},
           ...:                 "posterior_predictive": {
           ...:                     "obs": (dt.posterior_predictive.obs - dt.posterior.mu)**2}})
           ...: loo_pit(new_dt)

    References
    ----------

    .. [1] Vehtari et al. *Practical Bayesian model evaluation using leave-one-out cross-validation
        and WAIC*. Statistics and Computing. 27(5) (2017) https://doi.org/10.1007/s11222-016-9696-4
        arXiv preprint https://arxiv.org/abs/1507.04544.

    .. [2] Vehtari et al. *Pareto Smoothed Importance Sampling*.
        Journal of Machine Learning Research, 25(72) (2024) https://jmlr.org/papers/v25/19-556.html
        arXiv preprint https://arxiv.org/abs/1507.02646
    """
    data = convert_to_datatree(data)
    rng = np.random.default_rng(214)

    if var_names is None:
        var_names = list(data.observed_data.data_vars.keys())
    elif isinstance(var_names, str):
        var_names = [var_names]

    log_likelihood = get_log_likelihood_dataset(data, var_names=var_names)

    if log_weights is None:
        n_samples = log_likelihood.chain.size * log_likelihood.draw.size
        reff = _get_r_eff(data, n_samples)
        log_weights, _ = log_likelihood.azstats.psislw(r_eff=reff)

    posterior_predictive = extract(
        data,
        group="posterior_predictive",
        combined=False,
        var_names=var_names,
        keep_dataset=True,
    )
    observed_data = extract(
        data,
        group="observed_data",
        combined=False,
        var_names=var_names,
        keep_dataset=True,
    )

    sel_min = {}
    sel_sup = {}
    for var in var_names:
        pred = posterior_predictive[var]
        obs = observed_data[var]
        sel_min[var] = pred < obs
        sel_sup[var] = pred == obs

    sel_min = xr.Dataset(sel_min)
    sel_sup = xr.Dataset(sel_sup)

    pit = np.exp(logsumexp(log_weights.where(sel_min, -np.inf), dims=["chain", "draw"]))

    loo_pit_values = xr.Dataset(coords=observed_data.coords)
    for var in var_names:
        pit_lower = pit[var].values

        if sel_sup[var].any():
            pit_sup_addition = np.exp(
                logsumexp(log_weights.where(sel_sup[var], -np.inf), dims=["chain", "draw"])
            )

            pit_upper = pit_lower + pit_sup_addition[var].values
            random_value = rng.uniform(pit_lower, pit_upper)
            loo_pit_values[var] = observed_data[var].copy(data=random_value)
        else:
            loo_pit_values[var] = observed_data[var].copy(data=pit_lower)

    return loo_pit_values


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


def loo_subsample(
    data,
    observations,
    pointwise=None,
    var_name=None,
    reff=None,
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
        log_weights_ds, pareto_k_ds = sample_ds.azstats.psislw(
            r_eff=reff, dims=loo_inputs.sample_dims
        )
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
        )

    elpd_i_full, pareto_k_full = _prepare_full_arrays(
        elpd_loo_i,
        pareto_k_sample_da,
        subsample_data.lpd_approx_all,
        subsample_data.indices,
        loo_inputs.obs_dims,
        elpd_loo_hat,
    )

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
    )


def update_subsample(
    loo_orig,
    data,
    observations=None,
    var_name=None,
    reff=None,
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

    elpd_loo_i_new_da, pareto_k_new_da, approx_posterior = _compute_loo_results(
        log_likelihood=update_data.log_likelihood_new,
        var_name=loo_inputs.var_name,
        sample_dims=loo_inputs.sample_dims,
        n_samples=loo_inputs.n_samples,
        n_data_points=len(update_data.new_indices),
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
    )


def loo_moment_match(
    data,
    loo_orig,
    upars,
    log_prob_upars_fn,
    log_lik_i_upars_fn,
    max_iters=30,
    k_threshold=None,
    split=True,
    cov=False,
    pointwise=None,
    var_name=None,
    reff=None,
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
        `chain` and `draw` dimensions.
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
    var_name : str, optional
        The name of the variable in log_likelihood groups storing the pointwise log
        likelihood data to use for loo computation.
    reff: float, optional
        Relative MCMC efficiency, ``ess / n`` i.e. number of effective samples divided by the number
        of actual samples. Computed from trace by default.

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
    We will use the radon dataset to demonstrate how moment matching works.

    .. ipython::

        In [1]: import arviz_base as az
           ...: from arviz_stats import loo, loo_moment_match
           ...: import numpy as np
           ...: import xarray as xr
           ...: import pymc as pm
           ...: idata = az.load_arviz_data("radon")

    Compute the initial pointwise LOO:

    .. ipython::

        In [2]: loo_orig = loo(idata, pointwise=True, var_name="y")
           ...: loo_orig

    We need to inflate some Pareto k values since the original Pareto k values are
    all below the threshold:

    .. ipython::

        In [3]: rng = np.random.default_rng(214)
           ...: n_obs = len(loo_orig.pareto_k.values.flatten())
           ...: problematic_indices = rng.choice(n_obs, size=18, replace=False)
           ...:
           ...: k_values = loo_orig.pareto_k.values.copy()
           ...: k_flat = k_values.flatten()
           ...: k_flat[problematic_indices] = rng.uniform(0.8, 3.0, size=18)
           ...:
           ...: loo_orig.pareto_k.values = k_flat.reshape(k_values.shape)
           ...: loo_orig

    Reconstruct the model from the data:

    .. ipython::

        In [4]: county_idx = idata.constant_data.county_idx.values
           ...: floor_idx = idata.constant_data.floor_idx.values
           ...: uranium = idata.constant_data.uranium.values
           ...: y_obs = idata.observed_data.y.values
           ...:
           ...: with pm.Model() as radon_model:
           ...:     # Hyperpriors
           ...:     g = pm.Normal("g", mu=0.0, sigma=10.0, shape=2)
           ...:     sigma_a = pm.Exponential("sigma_a", 1.0)
           ...:
           ...:     # County effects
           ...:     a = pm.Deterministic("a", g[0] + g[1] * uranium)
           ...:     za_county = pm.Normal("za_county", mu=0.0, sigma=1.0, shape=85)
           ...:     a_county = pm.Deterministic("a_county", a + za_county * sigma_a)
           ...:
           ...:     # Observation model
           ...:     b = pm.Normal("b", mu=0.0, sigma=1.0)
           ...:     theta = a_county[county_idx] + b * floor_idx
           ...:     sigma = pm.Exponential("sigma", 1.0)
           ...:     y = pm.Normal("y", theta, sigma=sigma, observed=y_obs)
           ...:
           ...:     idata_with_transforms = pm.sample(
           ...:         draws=1000, tune=1000, chains=4,
           ...:         idata_kwargs={"include_transformed": True},
           ...:         random_seed=42
           ...:     )

    We need to extract the unconstrained parameters from the posterior. Recall that moment matching
    requires the parameters to be in the unconstrained space. The parameters in this model
    have different shapes, so we need to flatten into a single dimension before stacking them:


    .. ipython::

        In [5]: posterior = idata_with_transforms.posterior
           ...:
           ...: upars = xr.DataArray(
           ...:     np.concatenate([
           ...:         posterior['g'].values,
           ...:         posterior['sigma_a_log__'].values[..., None],
           ...:         posterior['za_county'].values,
           ...:         posterior['b'].values[..., None],
           ...:         posterior['sigma_log__'].values[..., None]
           ...:     ], axis=-1),
           ...:     dims=['chain', 'draw', 'param'],
           ...:     coords={'chain': posterior.chain, 'draw': posterior.draw}
           ...: )

    Now we need to create the log density function that takes the unconstrained parameters and
    returns the log density. We will use the compiled logp function from PyMC to do this.

    .. ipython::

        In [6]: compiled_logp = radon_model.compile_logp()
           ...:
           ...: def log_prob_upars(upars):
           ...:     posterior = idata_with_transforms.posterior
           ...:     logp_values = np.empty(upars.shape[:2])
           ...:
           ...:     for chain_idx, draw_idx in np.ndindex(*upars.shape[:2]):
           ...:         point_dict = {
           ...:             'g': posterior['g'].isel(chain=chain_idx, draw=draw_idx).values,
           ...:             'sigma_a_log__': posterior['sigma_a_log__'].isel(
           ...:                 chain=chain_idx, draw=draw_idx).item(),
           ...:             'za_county': posterior['za_county'].isel(
           ...:                 chain=chain_idx, draw=draw_idx).values,
           ...:             'b': posterior['b'].isel(chain=chain_idx, draw=draw_idx).item(),
           ...:             'sigma_log__': posterior['sigma_log__'].isel(
           ...:                 chain=chain_idx, draw=draw_idx).item()
           ...:         }
           ...:         logp_values[chain_idx, draw_idx] = compiled_logp(point_dict)
           ...:
           ...:     return xr.DataArray(
           ...:         logp_values,
           ...:         dims=["chain", "draw"],
           ...:         coords={"chain": upars.chain, "draw": upars.draw}
           ...:     )

    Next, we need to create the pointwise log likelihood function that takes the unconstrained
    parameters and returns the pointwise log likelihood. We will use the compiled logp function
    again to do this:

    .. ipython::

        In [7]: def log_lik_i_upars(upars, i):
           ...:     posterior = idata_with_transforms.posterior
           ...:     obs_data = idata.constant_data
           ...:
           ...:     c_idx = obs_data.county_idx.values[i]
           ...:     a_county = posterior['a_county'].isel(a_county_dim_0=c_idx)
           ...:     mu_i = a_county + posterior['b'] * obs_data.floor_idx.values[i]
           ...:
           ...:     y_i = idata.observed_data.y.values[i]
           ...:     sigma = np.exp(posterior['sigma_log__'])
           ...:     loglik = pm.logp(pm.Normal.dist(mu_i, sigma), y_i).eval()
           ...:
           ...:     return xr.DataArray(
           ...:         loglik,
           ...:         dims=["chain", "draw"],
           ...:         coords={"chain": upars.chain, "draw": upars.draw}
           ...:     )

    Now we can apply moment matching with covariance matching and split transformation:

    .. ipython::
        :okwarning:

        In [8]: loo_mm = loo_moment_match(
           ...:     idata,
           ...:     loo_orig,
           ...:     upars=upars,
           ...:     log_prob_upars_fn=log_prob_upars,
           ...:     log_lik_i_upars_fn=log_lik_i_upars,
           ...:     var_name="y",
           ...:     split=True,
           ...:     cov=True,
           ...: )
           ...: loo_mm

    All the Pareto k values are now below the threshold, and the PSIS-LOO-CV is improved.

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
        lwi, ki_tuple = log_ratio_i_init.azstats.psislw(r_eff=reff, dims=sample_dims)

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
                    r_eff=split_res.reff, dims=sample_dims
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


def compare(
    compare_dict,
    method="stacking",
    var_name=None,
):
    r"""Compare models based on their expected log pointwise predictive density (ELPD).

    The ELPD is estimated by Pareto smoothed importance sampling leave-one-out
    cross-validation, the same method used by ``func:arviz_stats.loo``.
    The method is described in [1]_ and [2]_.
    By default, the weights are estimated using ``"stacking"`` as described in [3]_.

    Parameters
    ----------
    compare_dict: dict of {str: DataTree or ELPDData}
        A dictionary of model names and :class:`xr.DataTree` or ``ELPDData``.
    method: str, optional
        Method used to estimate the weights for each model. Available options are:

        - 'stacking' : stacking of predictive distributions.
        - 'BB-pseudo-BMA' : pseudo-Bayesian Model averaging using Akaike-type
          weighting. The weights are stabilized using the Bayesian bootstrap.
        - 'pseudo-BMA': pseudo-Bayesian Model averaging using Akaike-type
          weighting, without Bootstrap stabilization (not recommended).

        For more information read https://arxiv.org/abs/1704.02030
    var_name: str, optional
        If there is more than a single observed variable in the ``InferenceData``, which
        should be used as the basis for comparison.

    Returns
    -------
    A DataFrame, ordered from best to worst model (measured by the ELPD).
    The index reflects the key with which the models are passed to this function. The columns are:
    rank: The rank-order of the models. 0 is the best.
    elpd: ELPD estimated either using (PSIS-LOO-CV `elpd_loo` or WAIC `elpd_waic`).
        Higher ELPD indicates higher out-of-sample predictive fit ("better" model).
    pIC: Estimated effective number of parameters.
    elpd_diff: The difference in ELPD between two models.
        If more than two models are compared, the difference is computed relative to the
        top-ranked model, that always has a elpd_diff of 0.
    weight: Relative weight for each model.
        This can be loosely interpreted as the probability of each model (among the compared model)
        given the data. By default the uncertainty in the weights estimation is considered using
        Bayesian bootstrap.
    SE: Standard error of the ELPD estimate.
        If method = BB-pseudo-BMA these values are estimated using Bayesian bootstrap.
    dSE: Standard error of the difference in ELPD between each model and the top-ranked model.
        It's always 0 for the top-ranked model.
    warning: A value of 1 indicates that the computation of the ELPD may not be reliable.
        This could be indication of WAIC/LOO starting to fail see
        http://arxiv.org/abs/1507.04544 for details.

    Examples
    --------
    Compare the centered and non centered models of the eight school problem:

    .. ipython::  python
        :okwarning:

        In [1]: from arviz_stats import compare
           ...: from arviz_base import load_arviz_data
           ...: data1 = load_arviz_data("non_centered_eight")
           ...: data2 = load_arviz_data("centered_eight")
           ...: compare_dict = {"non centered": data1, "centered": data2}
           ...: compare(compare_dict)

    See Also
    --------
    :func:`loo` : Compute the ELPD using the Pareto smoothed importance sampling Leave-one-out
        cross-validation method.
    :func:`arviz_plots.plot_compare`: Summary plot for model comparison.

    References
    ----------

    .. [1] Vehtari et al. *Practical Bayesian model evaluation using leave-one-out cross-validation
        and WAIC*. Statistics and Computing. 27(5) (2017) https://doi.org/10.1007/s11222-016-9696-4
        arXiv preprint https://arxiv.org/abs/1507.04544.

    .. [2] Vehtari et al. *Pareto Smoothed Importance Sampling*.
        Journal of Machine Learning Research, 25(72) (2024) https://jmlr.org/papers/v25/19-556.html
        arXiv preprint https://arxiv.org/abs/1507.02646

    .. [3] Yao et al. *Using stacking to average Bayesian predictive distributions*
        Bayesian Analysis, 13, 3 (2018). https://doi.org/10.1214/17-BA1091
        arXiv preprint https://arxiv.org/abs/1704.02030.
    """
    ics_dict = _calculate_ics(compare_dict, var_name=var_name)
    names = list(ics_dict.keys())

    df_comp = pd.DataFrame(
        {
            "rank": pd.Series(index=names, dtype="int"),
            "elpd": pd.Series(index=names, dtype="float"),
            "p": pd.Series(index=names, dtype="float"),
            "elpd_diff": pd.Series(index=names, dtype="float"),
            "weight": pd.Series(index=names, dtype="float"),
            "se": pd.Series(index=names, dtype="float"),
            "dse": pd.Series(index=names, dtype="float"),
            "warning": pd.Series(index=names, dtype="boolean"),
        }
    )

    method = rcParams["stats.ic_compare_method"] if method is None else method
    if method.lower() not in ["stacking", "bb-pseudo-bma", "pseudo-bma"]:
        raise ValueError(f"The method {method}, to compute weights, is not supported.")

    ics = pd.DataFrame.from_dict(ics_dict, orient="index")
    ics.sort_values(by="elpd", inplace=True, ascending=False)
    ics["elpd_i"] = ics["elpd_i"].apply(lambda x: x.values.flatten())
    ses = ics["se"]

    if method.lower() == "stacking":
        rows, cols, ic_i_val = _ic_matrix(ics)
        exp_ic_i = np.exp(ic_i_val)
        km1 = cols - 1

        def w_fuller(weights):
            return np.concatenate((weights, [max(1.0 - np.sum(weights), 0.0)]))

        def log_score(weights):
            w_full = w_fuller(weights)
            score = 0.0
            for i in range(rows):
                score += np.log(np.dot(exp_ic_i[i], w_full))
            return -score

        def gradient(weights):
            w_full = w_fuller(weights)
            grad = np.zeros(km1)
            for k, i in itertools.product(range(km1), range(rows)):
                grad[k] += (exp_ic_i[i, k] - exp_ic_i[i, km1]) / np.dot(exp_ic_i[i], w_full)
            return -grad

        theta = np.full(km1, 1.0 / cols)
        bounds = [(0.0, 1.0) for _ in range(km1)]
        constraints = [
            {"type": "ineq", "fun": lambda x: -np.sum(x) + 1.0},
            {"type": "ineq", "fun": np.sum},
        ]

        weights = minimize(
            fun=log_score, x0=theta, jac=gradient, bounds=bounds, constraints=constraints
        )

        weights = w_fuller(weights["x"])

    elif method.lower() == "bb-pseudo-bma":
        b_samples = 1000
        rows, cols, ic_i_val = _ic_matrix(ics)
        ic_i_val = ic_i_val * rows

        b_weighting = dirichlet.rvs(alpha=[1] * rows, size=b_samples, random_state=124)
        weights = np.zeros((b_samples, cols))
        z_bs = np.zeros_like(weights)
        for i in range(b_samples):
            z_b = np.dot(b_weighting[i], ic_i_val)
            u_weights = np.exp(z_b - np.max(z_b))
            z_bs[i] = z_b  # pylint: disable=unsupported-assignment-operation
            weights[i] = u_weights / np.sum(u_weights)

        weights = weights.mean(axis=0)
        ses = pd.Series(z_bs.std(axis=0), index=ics.index)  # pylint: disable=no-member

    elif method.lower() == "pseudo-bma":
        min_ic = ics.iloc[0]["elpd"]
        z_rv = np.exp(ics["elpd"] - min_ic)
        weights = (z_rv / np.sum(z_rv)).to_numpy()

    if np.any(weights):
        min_ic_i_val = ics["elpd_i"].iloc[0]
        for idx, val in enumerate(ics.index):
            res = ics.loc[val]
            diff = min_ic_i_val - res["elpd_i"]
            d_ic = np.sum(diff)
            d_std_err = np.sqrt(len(diff) * np.var(diff))
            std_err = ses.loc[val]
            weight = weights[idx]
            df_comp.loc[val] = (
                idx,
                res["elpd"],
                res["p"],
                d_ic,
                weight,
                std_err,
                d_std_err,
                res["warning"],
            )

    df_comp["rank"] = df_comp["rank"].astype(int)
    df_comp["warning"] = df_comp["warning"].astype(bool)
    return df_comp.sort_values(by="elpd", ascending=False)


def _ic_matrix(ics):
    """Store the previously computed pointwise predictive accuracy values (ics) in a 2D matrix."""
    cols, _ = ics.shape
    rows = len(ics["elpd_i"].iloc[0])
    ic_i_val = np.zeros((rows, cols))

    for idx, val in enumerate(ics.index):
        ic = ics.loc[val]["elpd_i"]

        if len(ic) != rows:
            raise ValueError("The number of observations should be the same across all models")

        ic_i_val[:, idx] = ic

    return rows, cols, ic_i_val


def _calculate_ics(
    compare_dict,
    var_name=None,
):
    """Calculate LOO only if necessary.

    It always calls LOO with ``pointwise=True``.

    Parameters
    ----------
    compare_dict :  dict of {str : DataTree or ELPDData}
        A dictionary of model names and DataTree or ELPDData objects.
    var_name : str, optional
        Name of the variable storing pointwise log likelihood values in ``log_likelihood`` group.


    Returns
    -------
    compare_dict : dict of ELPDData
    """
    precomputed_elpds = {
        name: elpd_data
        for name, elpd_data in compare_dict.items()
        if isinstance(elpd_data, ELPDData)
    }
    if precomputed_elpds:
        first_kind = list(precomputed_elpds.values())[0].kind
        for _, elpd_data in precomputed_elpds.items():
            if elpd_data.elpd_i is None:
                raise ValueError(
                    "All provided ELPDData should have been calculated with pointwise=True"
                )
            if elpd_data.kind != first_kind:
                raise ValueError("All elpd values should be computed using the same method")

    compare_dict = deepcopy(compare_dict)
    for name, dataset in compare_dict.items():
        if not isinstance(dataset, ELPDData):
            try:
                compare_dict[name] = loo(
                    dataset,
                    pointwise=True,
                    var_name=var_name,
                )
            except Exception as e:
                raise e.__class__(
                    f"Encountered error trying to compute elpd from model {name}."
                ) from e
    return compare_dict


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
):
    """Compute PSIS-LOO-CV results."""
    if log_p is not None and log_q is not None:
        data = xr.DataTree()
        data["log_likelihood"] = log_likelihood

        loo_results = loo_approximate_posterior(
            data=data, log_p=log_p, log_q=log_q, pointwise=True, var_name=var_name
        )

        if return_pointwise:
            return loo_results.elpd_i, loo_results.pareto_k, True
        return loo_results

    if log_weights is None or pareto_k is None:
        log_weights, pareto_k = log_likelihood.azstats.psislw(r_eff=reff, dims=sample_dims)

    log_weights += log_likelihood
    pareto_k_da = pareto_k

    warn_mg, good_k = _warn_pareto_k(pareto_k_da, n_samples)
    elpd_i = logsumexp(log_weights, dims=sample_dims)

    if return_pointwise:
        if isinstance(elpd_i, xr.Dataset) and var_name in elpd_i:
            elpd_i = elpd_i[var_name]
        if isinstance(pareto_k_da, xr.Dataset) and var_name in pareto_k_da:
            pareto_k_da = pareto_k_da[var_name]
        return elpd_i, pareto_k_da, approx_posterior

    elpd_i_values = elpd_i.values if hasattr(elpd_i, "values") else np.asarray(elpd_i)
    elpd_raw = logsumexp(log_likelihood, b=1 / n_samples, dims=sample_dims).sum().values
    elpd = np.sum(elpd_i_values)
    elpd_se = (n_data_points * np.var(elpd_i_values)) ** 0.5
    p_loo = elpd_raw - elpd

    pointwise = rcParams["stats.ic_pointwise"] if pointwise is None else pointwise
    if not pointwise:
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
            approx_posterior=approx_posterior,
        )

    _warn_pointwise_loo(elpd, elpd_i_values)

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
        elpd_i,
        pareto_k_da,
        approx_posterior=approx_posterior,
    )
