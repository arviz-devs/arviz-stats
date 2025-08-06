"""Auxiliary statistical functions."""

import warnings

import numpy as np
from arviz_base import dict_to_dataset, extract
from scipy import interpolate
from scipy.optimize import isotonic_regression


def isotonic_fit(dt, var_names, group, n_bootstrap, ci_prob, data_type="binary"):
    """
    Perform isotonic regression over a DataTree.

    This is helpful to create calibration plots for binary, categorical, or ordinal regression.
    Isotonic regression can also be used for continuous data, but it is less common.

    Parameters
    ----------
    dt: DataTree
        DataTree with "posterior_predictive" and "observed_data" groups
    var_names : str or list of str, optional
        The variables to perform the isotonic regression on.
    group : str
        The group from which to get the unique values.
    n_bootstrap : int
        The number of bootstrap samples to use.
    ci_prob : float, optional
        The probability for the credible interval.
    data_type : str
        Defaults to "binary", other options are "categorical" and "ordinal".
    """
    pp = extract(dt, group=group, keep_dataset=True)
    dictio = {}

    if var_names is None:
        var_names = dt.observed_data.data_vars

    if isinstance(var_names, str):
        var_names = [var_names]

    valid_data_types = ["binary", "categorical", "ordinal"]
    if data_type not in valid_data_types:
        raise ValueError(f"Invalid data_type '{data_type}'. Must be one of {valid_data_types}.")

    for v_name in var_names:
        preds = pp[v_name]
        observed = dt.observed_data[v_name]

        if data_type == "binary":
            pred_mean = preds.mean("sample")
            cep, forecasted, ci_lb, ci_ub = _isotonic_fit(pred_mean, observed, n_bootstrap, ci_prob)
            dictio[v_name] = np.stack([forecasted, cep, ci_lb, ci_ub])

        elif data_type in ("categorical", "ordinal"):
            obs_vals = observed.values
            if np.issubdtype(obs_vals.dtype, np.floating):
                warnings.warn(
                    f"Kind '{data_type}' specified but data for '{v_name}' appears to be "
                    "continuous. Isotonic regression can be used continuous data, but is not "
                    "the most common use case. If that's the case set `kind=binary`.",
                    UserWarning,
                )
            elif np.unique(obs_vals).size <= 2:
                warnings.warn(
                    f"Kind '{data_type}' specified but data for '{v_name}' appears to be binary. "
                    "setting `kind=binary` may be more appropriate.",
                    UserWarning,
                )
            n_categories = int(np.max([preds.max(), observed.max()])) + 1
            prob_per_cat = [(preds == k).mean("sample") for k in range(n_categories)]
            prob_per_cat = np.stack(prob_per_cat, axis=-1)

            if data_type == "categorical":
                # for categorical data, we need to handle each category separately
                # and compare each category against the rest (one vs others)
                obs_one_hot = np.eye(n_categories)[obs_vals]

                for k in range(n_categories):
                    cep, forecasted, ci_lb, ci_ub = _isotonic_fit(
                        prob_per_cat[..., k], obs_one_hot[..., k], n_bootstrap, ci_prob
                    )
                    dictio[f"{v_name} {k} vs others"] = np.stack([forecasted, cep, ci_lb, ci_ub])

            else:
                # for ordinal data, we need to respect the order so we compute the
                # cumulative probabilities
                cum_probs = np.cumsum(prob_per_cat, axis=-1)
                cum_indicators = (np.expand_dims(obs_vals, -1) <= np.arange(n_categories)).astype(
                    int
                )

                for k in range(n_categories - 1):
                    cep, forecasted, ci_lb, ci_ub = _isotonic_fit(
                        cum_probs[..., k], cum_indicators[..., k], n_bootstrap, ci_prob
                    )
                    dictio[f"{v_name} cumul≤{k}"] = np.stack([forecasted, cep, ci_lb, ci_ub])

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
