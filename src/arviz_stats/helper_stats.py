"""Auxiliary statistical functions."""

import numpy as np
from arviz_base import dict_to_dataset, extract
from scipy import interpolate
from scipy.optimize import isotonic_regression


def isotonic_fit(dt, var_names, n_bootstrap, ci_prob):
    """
    Perform isotonic regression over a DataTree.

    Parameters
    ----------
    dt: DataTree
        DataTree with "posterior_predictive" and "observed_data" groups
    var_names : list of str, optional
        The variables to perform the isotonic regression on.
    n_bootstrap : int
        The number of bootstrap samples to use.
    ci_prob : float, optional
        The probability for the credible interval.
    """
    if "/observed_data" not in dt.groups:
        raise ValueError("DataTree must have an 'observed_data' group.")
    if "/posterior_predictive" not in dt.groups:
        raise ValueError("DataTree must have a 'posterior_predictive' group.")

    pp = extract(dt, group="posterior_predictive", keep_dataset=True)

    dictio = {}
    vars_ = []

    if var_names is None:
        var_names = dt.posterior_predictive.data_vars

    for var in dt.posterior_predictive.data_vars:
        if var in var_names:
            pred = pp[var].mean("sample")
            cep, counts, forecasted, ci_lb, ci_ub = _isotonic_fit(
                pred, dt.observed_data[var], n_bootstrap, ci_prob
            )
            dictio[var] = np.stack([cep, counts, forecasted, ci_lb, ci_ub])
            vars_.append(var)

    return (
        dict_to_dataset(dictio)
        .rename({"draw": "x_values", "chain": "plot_axis"})
        .assign_coords({"plot_axis": ["y", "counts", "x", "y_bottom", "y_top"]})
    )


def _isotonic_fit(pred, obs, n_bootstrap, ci_prob=None):
    """
    Perform isotonic regression on the observed data and return the fitted values.

    Parameters
    ----------
    pred : array-like
        The predictor variable.
    obs : array-like
        The observed variable.
    n_bootstrap : int
        The number of bootstrap samples to use.
    ci_prob : float, optional
        The probability for the credible interval.
    """
    pred, obs = _sort_pred_with_obs(pred, obs)
    y_out = isotonic_regression(obs, increasing=True).x
    rng = np.random.default_rng(42)

    # calculate the fitting function
    forecasted, counts = np.unique(pred, return_counts=True)
    ir_func = interpolate.interp1d(pred, y_out, bounds_error=False)
    cep = ir_func(forecasted)

    # bootstrap the isotonic regression
    result = np.zeros((n_bootstrap, len(counts)))
    for i in range(n_bootstrap):
        idx = rng.choice(len(pred), len(pred), replace=True)
        pred_boot, obs_boot = _sort_pred_with_obs(pred[idx], obs[idx])
        y_out_boot = isotonic_regression(obs_boot, increasing=True).x

        result[i] = interpolate.interp1d(pred_boot, y_out_boot, bounds_error=False)(forecasted)

    lb = (1 - ci_prob) / 2 * 100
    ub = (1 + ci_prob) / 2 * 100
    ci = np.nanpercentile(result, [lb, ub], axis=0)

    return cep, counts, forecasted, ci[0], ci[1]


def _sort_pred_with_obs(pred, obs):
    """Ensure ties in pred have same regression value in PAV algorithm."""
    sorter = np.lexsort((-obs, pred))
    pred = pred[sorter]
    obs = obs[sorter]

    return pred, obs
