"""Compute weighted expectations and predictive metrics using PSIS-LOO-CV."""

import numpy as np
import xarray as xr
from arviz_base import convert_to_datatree, extract, rcParams
from scipy.stats import dirichlet
from xarray import apply_ufunc

from arviz_stats.base.stats_utils import _circdiff, _circular_var
from arviz_stats.loo.helper_loo import _warn_pareto_k
from arviz_stats.metrics import _metrics, _summary_r2
from arviz_stats.utils import ELPDData, get_log_likelihood_dataset


def loo_expectations(
    data,
    var_name=None,
    log_weights=None,
    kind="mean",
    probs=None,
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
    log_weights : DataArray or ELPDData, optional
        Smoothed log weights. Can be either:

        - A DataArray with the same shape as the log likelihood data
        - An ELPDData object from a previous :func:`arviz_stats.loo` call.

        Defaults to None. If not provided, it will be computed using the PSIS-LOO method.
    kind: str, optional
        The kind of expectation to compute. Available options are:

        - 'mean'. Default.
        - 'median'.
        - 'var':.
        - 'sd'.
        - 'quantile'.
        - 'circular_mean': circular mean.
    probs: float or list of float, optional
        The quantile(s) to compute when kind is 'quantile'.
    circular: bool, default False
        Whether to compute circular statistics. Defaults to False.
        It's assumed that the circular data is in radians and ranges from -π to π.


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
    if kind not in ["mean", "median", "var", "sd", "quantile", "circular_mean"]:
        raise ValueError("kind must be either 'mean', 'median', 'var', 'sd' or 'quantile'")

    if kind == "quantile" and probs is None:
        raise ValueError("probs must be provided when kind is 'quantile'")

    dims = ("chain", "draw")
    if var_name is None:
        var_name = list(data.observed_data.data_vars.keys())[0]

    data = convert_to_datatree(data)
    log_likelihood = get_log_likelihood_dataset(data, var_names=var_name)
    n_samples = log_likelihood[var_name].sizes["chain"] * log_likelihood[var_name].sizes["draw"]

    if log_weights is None:
        log_weights, _ = log_likelihood.azstats.psislw()
        log_weights = log_weights[var_name]

    if isinstance(log_weights, ELPDData):
        if log_weights.log_weights is None:
            raise ValueError("ELPDData object does not contain log_weights")
        log_weights = log_weights.log_weights
        if var_name in log_weights:
            log_weights = log_weights[var_name]

    weights = np.exp(log_weights)

    posterior_predictive = extract(
        data, group="posterior_predictive", var_names=var_name, combined=False
    )

    weighted_predictions = posterior_predictive.weighted(weights)

    if kind == "mean":
        loo_expec = weighted_predictions.mean(dim=dims)

    elif kind == "median":
        loo_expec = weighted_predictions.quantile(0.5, dim=dims)

    elif kind == "var":
        # We use a Bessel's like correction term
        # instead of n/(n-1) we use ESS/(ESS-1)
        # where ESS/(ESS-1) = 1/(1-sum(weights**2))
        loo_expec = weighted_predictions.var(dim=dims) / (1 - np.sum(weights**2))

    elif kind == "sd":
        loo_expec = (weighted_predictions.var(dim=dims) / (1 - np.sum(weights**2))) ** 0.5

    elif kind == "quantile":
        loo_expec = weighted_predictions.quantile(probs, dim=dims)

    elif kind == "circular_mean":
        weights = weights / weights.sum(dim=dims)
        sum_sin = (weights * np.sin(posterior_predictive)).sum(dim=dims)
        sum_cos = (weights * np.cos(posterior_predictive)).sum(dim=dims)
        loo_expec = np.arctan2(sum_sin, sum_cos)

    log_ratios = -log_likelihood[var_name]

    # Compute function-specific khat
    khat = apply_ufunc(
        _get_function_khat,
        posterior_predictive,
        log_ratios,
        input_core_dims=[dims, dims],
        output_core_dims=[[]],
        exclude_dims=set(dims),
        kwargs={"kind": kind},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    _warn_pareto_k(khat.values, n_samples)

    return loo_expec, khat  # pylint: disable=E0606


def loo_metrics(data, kind="rmse", var_name=None, log_weights=None, round_to="2g"):
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
    log_weights: DataArray or ELPDData, optional
        Smoothed log weights. Can be either:

        - A DataArray with the same shape as the log likelihood data
        - An ELPDData object from a previous :func:`arviz_stats.loo` call.

        Defaults to None. If not provided, it will be computed using the PSIS-LOO method.
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
    predicted, _ = loo_expectations(data, kind="mean", var_name=var_name, log_weights=log_weights)

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
    round_to="2g",
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
    round_to: int or str, optional
        If integer, number of decimal places to round the result. If string of the
        form '2g' number of significant digits to round the result. Defaults to '2g'.
        Use None to return raw numbers.

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

    y = data.observed_data[var_name].values

    if circular:
        kind = "circular_mean"
    else:
        kind = "mean"

    # Here we should compute the loo-adjusted posterior mean, not the predictive mean
    ypred_loo = loo_expectations(data, var_name=var_name, kind=kind)[0].values

    if circular:
        eloo = _circdiff(ypred_loo, y)
    else:
        eloo = ypred_loo - y

    n = len(y)
    rd = dirichlet.rvs(np.ones(n), size=n_simulations, random_state=42)

    if circular:
        vary = _circular_var(y, rd)
        vareloo = _circular_var(eloo, rd)
    else:
        vary = (np.sum(rd * y**2, axis=1) - np.sum(rd * y, axis=1) ** 2) * (n / (n - 1))
        vareloo = (np.sum(rd * eloo**2, axis=1) - np.sum(rd * eloo, axis=1) ** 2) * (n / (n - 1))

    loo_r_squared = 1 - vareloo / vary
    loo_r_squared = np.clip(loo_r_squared, -1, 1)

    if summary:
        return _summary_r2(loo_r_squared, point_estimate, ci_kind, ci_prob, round_to)
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
