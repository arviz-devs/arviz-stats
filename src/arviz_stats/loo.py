"""Pareto-smoothed importance sampling LOO (PSIS-LOO-CV) related functions."""

import itertools
from collections import namedtuple
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
    _get_r_eff,
    _prepare_full_arrays,
    _prepare_loo_inputs,
    _prepare_subsample,
    _prepare_update_subsample,
    _select_obs_by_coords,
    _srs_estimator,
    _warn_pareto_k,
    _warn_pointwise_loo,
)
from arviz_stats.utils import ELPDData, get_log_likelihood_dataset, round_num


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
    valid_kind = ["mae", "rmse", "mse", "acc", "acc_balanced"]
    if kind not in valid_kind:
        raise ValueError(f"kind must be one of {valid_kind}")

    if var_name is None:
        var_name = list(data.observed_data.data_vars.keys())[0]

    estimate = namedtuple(kind, ["mean", "se"])
    observed = data.observed_data[var_name]
    elpd_pred = loo_expectations(data, kind="mean", var_name=var_name)

    n_obs = len(observed)

    if kind == "mae":
        abs_e = np.abs(observed - elpd_pred)
        mean = np.mean(abs_e)
        std_error = np.std(abs_e) / n_obs**0.5

    elif kind == "mse":
        sq_e = (observed - elpd_pred) ** 2
        mean = np.mean(sq_e)
        std_error = np.std(sq_e) / n_obs**0.5

    elif kind == "rmse":
        sq_e = (observed - elpd_pred) ** 2
        mean_mse = np.mean(sq_e)
        var_mse = np.var(sq_e) / n_obs
        var_rmse = var_mse / mean_mse / 4  # Comes from the first order Taylor approx.
        mean = mean_mse**0.5
        std_error = var_rmse**0.5

    elif kind == "acc":
        yhat = elpd_pred > 0.5
        acc = yhat == observed
        mean = np.mean(acc)
        std_error = (mean * (1 - mean) / n_obs) ** 0.5

    else:
        yhat = elpd_pred > 0.5
        mask = observed == 0
        true_neg = np.mean(yhat[mask] == observed[mask])
        true_pos = np.mean(yhat[~mask] == observed[~mask])
        mean = (true_pos + true_neg) / 2
        # This approximation has quite large bias for small samples
        bls_acc_var = (true_pos * (1 - true_pos) + true_neg * (1 - true_neg)) / 4
        std_error = bls_acc_var / n_obs**0.5

    return estimate(round_num(mean, round_to), round_num(std_error, round_to))


def loo_pit(
    data,
    var_names=None,
    data_pairs=None,
    log_weights=None,
    randomize=False,
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
    data_pairs : dict, optional
        Dictionary mapping log_likelihood and observed variable names (keys) to posterior
        predictive variable names (values). This allows different naming between groups.
        If None, or if a variable from `var_names` is not a key in `data_pairs`,
        it's assumed that the corresponding posterior predictive and observed data variables
        share the same name as the log_likelihood variable.
    log_weights: DataArray
        Smoothed log_weights. It must have the same shape as ``y_pred``
        Defaults to None, it will be computed using the PSIS-LOO method.
    randomize: bool
        Whether to randomize the PIT values for discrete variables. Randomization is needed for
        discrete data. This function assumes discrete variables if the observed or predicted
        data are stored as integers.

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

    Calculate LOO-PIT values with different variable names between observed and
    posterior predictive groups.

    .. ipython::

        In [1]: from arviz_base import from_dict
           ...: new_dt = from_dict({
           ...:     "posterior": dt.posterior,
           ...:     "log_likelihood": dt.log_likelihood,
           ...:     "observed_data": dt.observed_data,
           ...:     "posterior_predictive": {"y_pred": dt.posterior_predictive.obs}
           ...: })
           ...: loo_pit(new_dt, var_names="obs", data_pairs={"obs": "y_pred"})

    Calculate LOO-PIT values using as test quantity the square of the difference between
    each observation and `mu`. For this we create a new DataTree, copying the posterior and
    log_likelihood groups and creating new observed and posterior_predictive groups.

    .. ipython::

        In [1]: new_dt = from_dict({"posterior": dt.posterior,
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

    if var_names is None:
        var_names = list(data.observed_data.data_vars.keys())
    elif isinstance(var_names, str):
        var_names = [var_names]

    if data_pairs is None:
        data_pairs = {}

    pp_var_names = [data_pairs.get(vn, vn) for vn in var_names]
    log_likelihood = get_log_likelihood_dataset(data, var_names=var_names)

    if log_weights is None:
        n_samples = log_likelihood.chain.size * log_likelihood.draw.size
        reff = _get_r_eff(data, n_samples)
        log_weights, _ = log_likelihood.azstats.psislw(r_eff=reff)

    posterior_predictive = extract(
        data,
        group="posterior_predictive",
        combined=False,
        var_names=pp_var_names,
        keep_dataset=True,
    )
    observed_data = extract(
        data,
        group="observed_data",
        combined=False,
        var_names=var_names,
        keep_dataset=True,
    )

    type_vars = {}
    for i, var in enumerate(var_names):
        obs_var = var
        pp_var = pp_var_names[i]

        is_discrete = (posterior_predictive[pp_var].values.dtype.kind == "i") or (
            observed_data[obs_var].values.dtype.kind == "i"
        )
        type_vars[var] = "discrete" if is_discrete else "continuous"

    pit_vals = {}
    if randomize and "discrete" in type_vars.values():
        rng = np.random.default_rng(214)
        for i, var in enumerate(var_names):
            obs_var = var
            pp_var = pp_var_names[i]

            if type_vars[var] == "discrete":
                vals = posterior_predictive[pp_var] < observed_data[obs_var]
                urvs = rng.uniform(size=vals.values.shape)
                pit_vals[var] = urvs * vals + (1 - urvs) * vals

                # val_less_eq = posterior_predictive[pp_var] <= observed_data[obs_var]
                # vals_less = posterior_predictive[pp_var] < observed_data[obs_var]
                # urvs = rng.uniform(size=vals.values.shape)
                # pit_vals[var] = urvs * vals_less + (1 - urvs) * val_less_eq

            else:
                pit_vals[var] = posterior_predictive[pp_var] <= observed_data[obs_var]
    else:
        for i, var in enumerate(var_names):
            obs_var = var
            pp_var = pp_var_names[i]
            pit_vals[var] = posterior_predictive[pp_var] <= observed_data[obs_var]

    pit_vals = xr.Dataset(pit_vals)
    loo_pit_values = np.exp(
        logsumexp(log_weights.where(pit_vals, -np.inf), dims=["chain", "draw"])
        - logsumexp(log_weights, dims=["chain", "draw"])
    )
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
