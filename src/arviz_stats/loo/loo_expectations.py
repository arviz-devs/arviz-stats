"""Compute weighted expectations and predictive metrics using PSIS-LOO-CV."""

import numpy as np
from arviz_base import convert_to_datatree, extract

from arviz_stats.metrics import _metrics
from arviz_stats.utils import get_log_likelihood_dataset


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
    predicted = loo_expectations(data, kind="mean", var_name=var_name)

    return _metrics(observed, predicted, kind, round_to)
