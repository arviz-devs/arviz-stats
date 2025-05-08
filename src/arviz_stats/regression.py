"""Regression metrics for Bayesian models."""

from collections import namedtuple

import numpy as np
from arviz_base import extract, rcParams

from arviz_stats.base import array_stats


def r2_score(data, summary=True, point_estimate=None, ci_kind=None, ci_prob=None, round_to=2):
    """R² for Bayesian regression models.

    The R², or coefficient of determination, is defined as the proportion of variance
    in the data that is explained by the model. It is computed as the variance of the
    predicted values divided by the variance of the predicted values plus the variance
    of the residuals. For details of the Bayesian R² see [1]_.

    Parameters
    ----------
    data : DataTree or InferenceData
        Input data. It should contain the posterior and the log_likelihood groups.
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

        In [1]: from arviz_stats import r2_score
           ...: from arviz_base import load_arviz_data
           ...: data = load_arviz_data('regression1d')
           ...: r2_score(data)

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

    y_true = extract(data, group="observed_data", combined=False).values
    y_pred = extract(data, group="posterior_predictive").values.T

    r_squared = array_stats.r2_score(y_true, y_pred)

    if summary:
        estimate = getattr(np, point_estimate)(r_squared).item()
        c_i = getattr(array_stats, ci_kind)(r_squared, ci_prob)

        r2_summary = namedtuple("R2", [point_estimate, f"{ci_kind}_lb", f"{ci_kind}_ub"])
        if (round_to is not None) and (round_to not in ("None", "none")):
            estimate = round(estimate, round_to)
            c_i = (round(c_i[0].item(), round_to), round(c_i[1].item(), round_to))

        return r2_summary(estimate, c_i[0], c_i[1])

    return r_squared
