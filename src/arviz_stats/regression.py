"""Regression metrics for Bayesian models."""

import numpy as np
import pandas as pd
from arviz_base import rcParams
from xarray import DataArray

from arviz_stats.base import array_stats


def R2(y_true, y_pred, summary=True, point_estimate=None, ci_kind=None, ci_prob=None, round_to=2):  # pylint: disable=invalid-name
    """R² for Bayesian regression models.

    The R², or coefficient of determination, is defined as the proportion of variance
    in the data that is explained by the model. It is computed as the variance of the
    predicted values divided by the variance of the predicted values plus the variance
    of the residuals. For details of the Bayesian R² see [1]_.

    Parameters
    ----------
    y_true: array-like of shape = (n_outputs,)
        Ground truth (correct) target values.
    y_pred: array-like of shape = (n_posterior_samples, n_outputs)
        Estimated target values.
    summary: bool
        Whether to return a summary (default) or an array of R² samples.
        The summary is a Pandas' series with a point estimate and a credible interval
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
    round_to : int
        Number of decimals used to round results. Defaults to 2. Use "none" to return raw numbers.

    Returns
    -------
    Pandas Series or array

    Examples
    --------
    Calculate R² samples for Bayesian regression models :

    .. ipython::

        In [1]: from arviz_stats import R2
           ...: from arviz_base import load_arviz_data
           ...: data = load_arviz_data('regression1d')
           ...: y_true = data.observed_data["y"].values
           ...: y_pred = data.posterior_predictive["y"].stack(sample=("chain", "draw")).T
           ...: R2(y_true, y_pred)

    References
    ----------

    .. [1] Gelman et al. *R-squared for Bayesian regression models*.
        The American Statistician. 73(3) (2019). https://doi.org/10.1080/00031305.2018.1549100
        preprint http://www.stat.columbia.edu/~gelman/research/published/bayes_R2_v3.pdf.
    """
    if point_estimate is None:
        point_estimate = rcParams["stats.point_estimate"]
    if ci_kind is None:
        ci_kind = rcParams["stats.ci_kind"]
    if ci_prob is None:
        ci_prob = rcParams["stats.ci_prob"]

    if y_pred.ndim == 1:
        var_y_est = np.var(y_pred)
        var_e = np.var(y_true - y_pred)
    else:
        var_y_est = np.var(y_pred, axis=1)
        var_e = np.var(y_true - y_pred, axis=1)
    r_squared = var_y_est / (var_y_est + var_e)

    if isinstance(r_squared, DataArray):
        r_squared = r_squared.values

    if summary:
        estimate = getattr(np, point_estimate)(r_squared)
        c_i = getattr(array_stats, ci_kind)(r_squared, ci_prob)

        summary_s = pd.Series(
            [estimate, c_i[0], c_i[1]],
            index=[point_estimate, f"{ci_kind}_", f"{ci_kind}^"],
            name="R2",
        )

        if (round_to is not None) and (round_to not in ("None", "none")):
            summary_s = summary_s.round(round_to)

        return summary_s

    return r_squared
