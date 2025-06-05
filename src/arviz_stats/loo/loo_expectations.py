"""Compute weighted expectations and predictive metrics using PSIS-LOO-CV."""

import numpy as np
import xarray as xr
from arviz_base import convert_to_datatree, extract
from xarray import apply_ufunc

from arviz_stats.loo.helper_loo import _warn_pareto_k
from arviz_stats.metrics import _metrics
from arviz_stats.utils import get_log_likelihood_dataset


def loo_expectations(
    data,
    var_name=None,
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
    if kind not in ["mean", "median", "var", "sd", "quantile"]:
        raise ValueError("kind must be either 'mean', 'median', 'var', 'sd' or 'quantile'")

    if kind == "quantile" and probs is None:
        raise ValueError("probs must be provided when kind is 'quantile'")

    dims = ("chain", "draw")
    if var_name is None:
        var_name = list(data.observed_data.data_vars.keys())[0]

    data = convert_to_datatree(data)
    log_likelihood = get_log_likelihood_dataset(data, var_names=var_name)
    n_samples = log_likelihood[var_name].sizes["chain"] * log_likelihood[var_name].sizes["draw"]

    log_weights, _ = log_likelihood.azstats.psislw()
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
    else:  # kind == "quantile"
        loo_expec = weighted_predictions.quantile(probs, dim=dims)

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

    return loo_expec, khat


def loo_metrics(data, kind="rmse", var_name=None, round_to="2g"):
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
    predicted, _ = loo_expectations(data, kind="mean", var_name=var_name)

    return _metrics(observed, predicted, kind, round_to)


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
    khat_r_da = r_theta_da.azstats.pareto_khat(dims="sample", tail="right", log_weights=False)
    khat_r = khat_r_da.item()

    # For quantile/median, only need khat_r
    if kind in ["quantile", "median"]:
        return khat_r

    h_theta_values = values.ravel() if kind == "mean" else values.ravel() ** 2
    h_theta_finite = h_theta_values[np.isfinite(h_theta_values)]

    if h_theta_finite.size == 0 or len(np.unique(h_theta_finite)) <= 2:
        return khat_r

    # Compute khat for h(theta) * r(theta)
    hr_theta_da = xr.DataArray(h_theta_values * r_theta_da.values, dims=["sample"])
    khat_hr_da = hr_theta_da.azstats.pareto_khat(dims="sample", tail="both", log_weights=False)
    khat_hr = khat_hr_da.item()

    if np.isnan(khat_hr) or np.isnan(khat_r):
        return khat_r

    return max(khat_hr, khat_r)
