"""Auxiliary statistical functions."""

import numpy as np
from arviz_base import dict_to_dataset, extract
from scipy import interpolate
from scipy.optimize import isotonic_regression


def isotonic_fit(dt, data_pairs, n_bootstrap, ci_prob):
    """
    Perform isotonic regression over a DataTree.

    Parameters
    ----------
    dt: DataTree
        DataTree with "posterior_predictive" and "observed_data" groups
    data_pairs : dict
        Dictionary of keys prior/posterior predictive data and values observed data variable names.
    n_bootstrap : int
        The number of bootstrap samples to use.
    ci_prob : float, optional
        The probability for the credible interval.
    """
    pp = extract(dt, group="posterior_predictive", keep_dataset=True)

    dictio = {}
    vars_ = []

    if None in data_pairs.keys():
        data_pairs = dict(zip(dt.posterior_predictive.data_vars, dt.observed_data.data_vars))

    for var_predictive, var_obs in data_pairs.items():
        pred = pp[var_predictive].mean("sample")
        cep, forecasted, ci_lb, ci_ub = _isotonic_fit(
            pred, dt.observed_data[var_obs], n_bootstrap, ci_prob
        )
        dictio[var_predictive] = np.stack([forecasted, cep, ci_lb, ci_ub])
        vars_.append(var_predictive)

    return (
        dict_to_dataset(dictio)
        .rename({"draw": "x_values", "chain": "plot_axis"})
        .assign_coords({"plot_axis": ["x", "y", "y_bottom", "y_top"]})
    )


def _isotonic_fit(pred, obs, n_bootstrap, ci_prob):
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
    ir_func = interpolate.interp1d(pred, y_out, bounds_error=False)
    cep = ir_func(pred)

    # bootstrap the isotonic regression
    result = np.zeros((n_bootstrap, len(pred)))
    for i in range(n_bootstrap):
        idx = rng.choice(len(pred), len(pred), replace=True)
        pred_boot, obs_boot = _sort_pred_with_obs(pred[idx], obs[idx])
        y_out_boot = isotonic_regression(obs_boot, increasing=True).x

        result[i] = interpolate.interp1d(pred_boot, y_out_boot, bounds_error=False)(pred)

    lb = (1 - ci_prob) / 2 * 100
    ub = (1 + ci_prob) / 2 * 100
    ci = np.nanpercentile(result, [lb, ub], axis=0)

    return cep, pred, ci[0], ci[1]


def _sort_pred_with_obs(pred, obs):
    """Ensure ties in pred have same regression value in PAV algorithm."""
    sorter = np.lexsort((-obs, pred))
    pred = pred[sorter]
    obs = obs[sorter]

    return pred, obs


def point_interval_unique(dt, var_names, group, ci_prob):
    """
    Compute the mean frequency and confidence intervals for unique values.

    Parameters
    ----------
    dt: DataTree
        DataTree with "posterior_predictive" and "observed_data" groups
    group : str
        The group from which to get the unique values.
    var_names : list of str, optional
        The variables to compute the unique values.
    ci_prob : float, optional
        The probability for the credible interval.
    """
    pp = extract(dt, group=group, keep_dataset=True)

    dictio = {}

    if var_names is None:
        var_names = dt[group].data_vars

    for var in dt[group].data_vars:
        if var in var_names:
            unique_values = np.unique(pp[var])
            group_counts = []
            for y in unique_values:
                mask = pp[var] == y
                unique_counts = mask.mean(dim="sample").values
                group_counts.append(unique_counts)

            lb = (1 - ci_prob) / 2 * 100
            ub = (1 + ci_prob) / 2 * 100
            ci = np.nanpercentile(group_counts, [lb, ub], axis=1) * np.sum(group_counts)
            means = np.sum(group_counts, axis=1)

            dictio[var] = np.stack([unique_values, means, ci[0], ci[1]])

    return (
        dict_to_dataset(dictio)
        .rename({"draw": "x_values", "chain": "plot_axis"})
        .assign_coords({"plot_axis": ["x", "y", "y_bottom", "y_top"]})
    )


def point_unique(dt, var_names):
    """
    Compute the mean frequency for unique values.

    Parameters
    ----------
    dt: DataTree
        DataTree with "posterior_predictive" and "observed_data" groups
    var_names : list of str, optional
        The variables to perform the isotonic regression on.
    """
    pp = dt["observed_data"]

    dictio = {}

    if var_names is None:
        var_names = pp.data_vars

    for var in pp.data_vars:
        if var in var_names:
            unique_values = np.unique(pp[var])
            group_counts = []
            for y in unique_values:
                mask = pp[var] == y
                unique_counts = mask.sum().values
                group_counts.append(unique_counts)

            dictio[var] = np.stack([unique_values, np.array(group_counts)])

    return (
        dict_to_dataset(dictio)
        .rename({"draw": "x_values", "chain": "plot_axis"})
        .assign_coords({"plot_axis": ["x", "y"]})
    )
