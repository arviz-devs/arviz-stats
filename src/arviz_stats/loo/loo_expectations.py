"""Compute weighted expectations and predictive metrics using PSIS-LOO-CV."""

import numpy as np
import xarray as xr
from arviz_base import convert_to_datatree, extract, rcParams
from xarray import apply_ufunc

from arviz_stats.loo.helper_loo import _get_r_eff, _warn_pareto_k
from arviz_stats.metrics import _metrics, _summary_r2
from arviz_stats.utils import get_log_likelihood_dataset


def loo_expectations(
    data,
    var_name=None,
    kind="mean",
    probs=None,
    log_weights=None,
    pareto_k=None,
):
    """Compute weighted expectations using the PSIS-LOO-CV method.

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

        - 'mean'. Default.
        - 'median'.
        - 'var'.
        - 'sd'.
        - 'quantile'.
        - 'circular_mean'.
        - 'circular_var'.
        - 'circular_sd'.
    probs: float or list of float, optional
        The quantile(s) to compute when kind is 'quantile'.
    log_weights : DataArray, optional
        Pre-computed smoothed log weights from PSIS. Must be provided together with pareto_k.
        If not provided, PSIS will be computed internally.
    pareto_k : DataArray, optional
        Pre-computed Pareto k-hat diagnostic values. Must be provided together with log_weights.

    Returns
    -------
    loo_expec : DataArray
        The weighted expectations.
    khat : DataArray
        Function-specific Pareto k-hat diagnostics for each observation.

    Examples
    --------
    Calculate predictive 0.25 and 0.75 quantiles and the function-specific Pareto k-hat diagnostics

    .. ipython::

        In [1]: from arviz_stats import loo_expectations
           ...: from arviz_base import load_arviz_data
           ...: dt = load_arviz_data("radon")
           ...: loo_expec, khat = loo_expectations(dt, kind="quantile", probs=[0.25, 0.75])
           ...: loo_expec

    .. ipython::

        In [2]: khat

    References
    ----------

    .. [1] Vehtari et al. *Practical Bayesian model evaluation using leave-one-out cross-validation
        and WAIC*. Statistics and Computing. 27(5) (2017) https://doi.org/10.1007/s11222-016-9696-4
        arXiv preprint https://arxiv.org/abs/1507.04544.

    .. [2] Vehtari et al. *Pareto Smoothed Importance Sampling*.
        Journal of Machine Learning Research, 25(72) (2024) https://jmlr.org/papers/v25/19-556.html
        arXiv preprint https://arxiv.org/abs/1507.02646
    """
    if kind not in [
        "mean",
        "median",
        "var",
        "sd",
        "quantile",
        "circular_mean",
        "circular_var",
        "circular_sd",
    ]:
        raise ValueError(
            """kind must be either 'mean', 'median', 'var', 'sd', 'quantile',
            'circular_mean', 'circular_var', or 'circular_sd'"""
        )

    if kind == "quantile" and probs is None:
        raise ValueError("probs must be provided when kind is 'quantile'")

    dims = ("chain", "draw")
    if var_name is None:
        var_name = list(data.observed_data.data_vars.keys())[0]

    data = convert_to_datatree(data)
    log_likelihood = get_log_likelihood_dataset(data, var_names=var_name)
    n_samples = log_likelihood[var_name].sizes["chain"] * log_likelihood[var_name].sizes["draw"]

    r_eff = _get_r_eff(data, n_samples)

    posterior_predictive = extract(
        data, group="posterior_predictive", var_names=var_name, combined=False
    )

    sample_dims = list(dims)

    if log_weights is not None and pareto_k is not None:
        if kind in ("mean", "median", "var", "sd", "circular_mean", "circular_var", "circular_sd"):
            loo_expec, _ = posterior_predictive.azstats.loo_expectation(
                log_weights=log_weights,
                pareto_k=pareto_k,
                kind=kind,
                r_eff=r_eff,
                sample_dims=sample_dims,
            )
        else:
            loo_expec, _ = posterior_predictive.azstats.loo_quantile(
                log_weights=log_weights,
                pareto_k=pareto_k,
                probs=probs,
                r_eff=r_eff,
                sample_dims=sample_dims,
            )
        log_ratios_for_khat = log_weights
    else:
        log_ratios = -log_likelihood[var_name]
        if kind in ("mean", "median", "var", "sd", "circular_mean", "circular_var", "circular_sd"):
            loo_expec, _ = posterior_predictive.azstats.loo_expectation(
                log_ratios=log_ratios,
                kind=kind,
                r_eff=r_eff,
                sample_dims=sample_dims,
            )
        else:
            loo_expec, _ = posterior_predictive.azstats.loo_quantile(
                log_ratios=log_ratios,
                probs=probs,
                r_eff=r_eff,
                sample_dims=sample_dims,
            )
        log_ratios_for_khat = log_ratios

    khat = apply_ufunc(
        _get_function_khat,
        posterior_predictive,
        log_ratios_for_khat,
        input_core_dims=[dims, dims],
        output_core_dims=[[]],
        exclude_dims=set(dims),
        kwargs={"kind": kind},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    _warn_pareto_k(khat.values, n_samples)

    return loo_expec, khat


def loo_metrics(data, kind="rmse", var_name=None, round_to=None):
    """Compute predictive metrics using the PSIS-LOO-CV method.

    Currently supported metrics are mean absolute error, mean squared error and
    root mean squared error. For classification problems, accuracy and balanced
    accuracy are also supported.

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
    round_to: int or str or None, optional
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
    if round_to is None:
        round_to = rcParams["stats.round_to"]

    if var_name is None:
        var_name = list(data.observed_data.data_vars.keys())[0]

    observed = data.observed_data[var_name]
    predicted, _ = loo_expectations(data, kind="mean", var_name=var_name)

    return _metrics(observed, predicted, kind, round_to)


def loo_r2(
    data,
    var_name,
    n_simulations=4000,
    summary=True,
    point_estimate=None,
    ci_kind=None,
    ci_prob=None,
    circular=False,
    round_to=None,
):
    """Compute LOO-adjusted :math:`R^2`.

    Parameters
    ----------
    data: DataTree or InferenceData
        It should contain groups `observed_data`, `posterior_predictive` and `log_likelihood`.
    var_name : str
        Name of the observed variable
    n_simulations : int, default 4000
        Number of Dirichlet-weighted bootstrap samples for variance estimation.
    circular : bool, default False
        Whether the variable is circular (angles in radians, range [-π, π]).
    summary: bool
        Whether to return a summary (default) or an array of :math:`R^2` samples.
        The summary is a named tuple with a point estimate and a credible interval
    point_estimate: str
        The point estimate to compute. If None, the default value is used.
        Defaults values are defined in rcParams["stats.point_estimate"]. Ignored if
        summary is False.
    ci_kind: str
        The kind of credible interval to compute. If None, the default value is used.
        Defaults values are defined in rcParams["stats.ci_kind"]. Ignored if
        summary is False.
    ci_prob: float
        The probability for the credible interval. If None, the default value is used.
        Defaults values are defined in rcParams["stats.ci_prob"]. Ignored if
        summary is False.
    circular: bool
        Whether to compute the Bayesian :math:`R^2` for circular data. Defaults to False.
        It's assumed that the circular data is in radians and ranges from -π to π.
        We use the same definition of :math:`R^2` for circular data as in the linear case,
        but using circular variance instead of regular variance.
    round_to: int or str or None, optional
     If integer, number of decimal places to round the result. Integers can be negative.
        If string of the form '2g' number of significant digits to round the result.
        Defaults to rcParams["stats.round_to"] if None. Use the string "None" or "none" to
        return raw numbers.

    Returns
    -------
    Namedtuple or array

    See Also
    --------
    arviz_stats.bayesian_r2 : Bayesian :math:`R^2`.
    arviz_stats.residual_r2 : Residual :math:`R^2`.

    Examples
    --------
    Calculate LOO-adjusted :math:`R^2` for Bayesian logistic regression:

    .. ipython::
        :okwarning:

        In [1]: from arviz_stats import loo_r2
           ...: from arviz_base import load_arviz_data
           ...: data = load_arviz_data('anes')
           ...: loo_r2(data, var_name="vote")

    Calculate LOO-adjusted :math:`R^2` for circular regression:

    .. ipython::
        :okwarning:

        In [1]: data = load_arviz_data('periwinkles')
           ...: loo_r2(data, var_name='direction', circular=True)
    """
    if point_estimate is None:
        point_estimate = rcParams["stats.point_estimate"]
    if ci_kind is None:
        ci_kind = rcParams["stats.ci_kind"]
    if ci_prob is None:
        ci_prob = rcParams["stats.ci_prob"]
    if round_to is None:
        round_to = rcParams["stats.round_to"]

    y_obs = data.observed_data[var_name]

    if circular:
        kind = "circular_mean"
    else:
        kind = "mean"

    ypred_loo = loo_expectations(data, var_name=var_name, kind=kind)[0]

    loo_r_squared = y_obs.azstats.loo_r2(
        ypred_loo=ypred_loo,
        n_simulations=n_simulations,
        circular=circular,
        random_state=42,
    )

    if summary:
        return _summary_r2("loo", loo_r_squared, point_estimate, ci_kind, ci_prob, round_to)
    return loo_r_squared


def _get_function_khat(
    values,
    log_weights,
    kind,
):
    """Compute function-specific k-hat diagnostics for LOO expectations.

    Parameters
    ----------
    values : ndarray
        Values of the posterior predictive distribution, raveled across sample dimensions.
    log_weights : ndarray
        Raw log weights from PSIS, raveled across sample dimensions.
    kind : str
        Type of expectation being computed ('mean', 'median', 'var', 'sd', 'quantile').

    Returns
    -------
    khat : float
        Function-specific k-hat estimate.
    """
    r_theta_da = xr.DataArray(
        np.exp(log_weights.ravel() - np.max(log_weights.ravel())), dims=["sample"]
    )

    # Get right tail khat
    try:
        khat_r_da = r_theta_da.azstats.pareto_khat(
            sample_dims="sample", tail="right", log_weights=False
        )
        khat_r = khat_r_da.item()
    except ValueError:
        khat_r = np.nan

    # For quantile/median, only need khat_r
    if kind in ["quantile", "median"]:
        return khat_r

    h_theta_values = values.ravel() if kind == "mean" else values.ravel() ** 2
    unique_h = np.unique(h_theta_values[np.isfinite(h_theta_values)])

    if len(unique_h) <= 1:
        return khat_r
    if len(unique_h) == 2:
        return khat_r
    if np.any(~np.isfinite(h_theta_values)):
        return khat_r

    hr_theta_da = xr.DataArray(h_theta_values * r_theta_da.values, dims=["sample"])

    try:
        khat_hr_da = hr_theta_da.azstats.pareto_khat(
            sample_dims="sample", tail="both", log_weights=False
        )
        khat_hr = khat_hr_da.item()
    except ValueError:
        khat_hr = np.nan

    if np.isnan(khat_hr) and np.isnan(khat_r):
        return np.nan
    return np.nanmax([khat_hr, khat_r])
