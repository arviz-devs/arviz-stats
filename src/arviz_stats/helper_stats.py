"""Auxiliary statistical functions."""

import warnings

import numpy as np
from arviz_base import dict_to_dataset, extract
from scipy.optimize import isotonic_regression


def isotonic_fit(dt, var_names, group, ci_prob, data_type="binary", residuals=False, x_var=None):
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
    ci_prob : float, optional
        The probability for the credible interval.
    data_type : str
        Defaults to "binary", other options are "categorical" and "ordinal".
    residuals : bool, optional
        If True, compute residuals instead of CEP.
        Defaults to False.
    x_var : array-like, optional
        Variable to use for x-axis when computing residuals
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
            cep, forecasted, ci_lb, ci_ub = _isotonic_fit(
                pred_mean, observed, ci_prob, residuals, preds, x_var
            )
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
            preds_per_cat = [(preds == k).values for k in range(n_categories)]

            if data_type == "categorical":
                # for categorical data, we need to handle each category separately
                # and compare each category against the rest (one vs others)
                obs_one_hot = np.eye(n_categories)[obs_vals]

                for k in range(n_categories):
                    cep, forecasted, ci_lb, ci_ub = _isotonic_fit(
                        prob_per_cat[..., k],
                        obs_one_hot[..., k],
                        ci_prob,
                        residuals,
                        preds_per_cat[k],
                        x_var,
                    )
                    dictio[f"{v_name} {k} vs others"] = np.stack([forecasted, cep, ci_lb, ci_ub])

            else:
                # for ordinal data, we need to respect the order so we compute the
                # cumulative probabilities
                cum_probs = np.cumsum(prob_per_cat, axis=-1)
                cum_indicators = (np.expand_dims(obs_vals, -1) <= np.arange(n_categories)).astype(
                    int
                )
                preds_cumul = [
                    np.cumsum([(preds == j).values for j in range(n_categories)], axis=0)[k]
                    for k in range(n_categories)
                ]

                for k in range(n_categories - 1):
                    cep, forecasted, ci_lb, ci_ub = _isotonic_fit(
                        cum_probs[..., k],
                        cum_indicators[..., k],
                        ci_prob,
                        residuals,
                        preds_cumul[k],
                        x_var,
                    )
                    dictio[f"{v_name} cumul≤{k}"] = np.stack([forecasted, cep, ci_lb, ci_ub])

    return (
        dict_to_dataset(dictio)
        .rename({"draw": "x_values", "chain": "plot_axis"})
        .assign_coords({"plot_axis": ["x", "y", "y_bottom", "y_top"]})
    )


def _isotonic_fit(pred, obs, ci_prob, residuals, preds, x_var):
    """
    Perform isotonic regression on the observed data and return the fitted values.

    Parameters
    ----------
    pred : array-like
        The predictor variable (mean predictions).
    obs : array-like
        The observed variable.
    ci_prob : float
        The probability for the credible interval.
    residuals : bool
        If True, compute residuals (CEP - pred) instead of CEP.
    preds : array-like
        Full posterior predictive samples. Will be transposed if needed to shape (n_samples, n_obs).
    x_var : array-like
        Variable to use for x-axis.
    """
    pred = np.asarray(pred)
    obs = np.asarray(obs)

    sorter = np.argsort(pred)
    pred_sorted = pred[sorter]
    obs_sorted = obs[sorter]

    if x_var is not None:
        x_var = np.asarray(x_var)
        x_sorted = x_var[sorter]
    else:
        x_sorted = pred_sorted

    cep = isotonic_regression(obs_sorted, increasing=True).x

    preds_array = np.asarray(preds)
    if preds_array.shape[0] == len(pred_sorted):
        preds_array = preds_array.T
    n_draws = preds_array.shape[0]
    result = np.zeros((n_draws, len(pred_sorted)))

    for i in range(n_draws):
        preds_sorted = preds_array[i][sorter]
        result[i] = isotonic_regression(preds_sorted, increasing=True).x

    lb = (1 - ci_prob) / 2 * 100
    ub = (1 + ci_prob) / 2 * 100
    ci = np.percentile(result, [lb, ub], axis=0)

    if residuals:
        cep = cep - pred_sorted
        ci[0] = ci[0] - pred_sorted
        ci[1] = ci[1] - pred_sorted

    if x_var is not None:
        x_order = np.argsort(x_sorted)
        x_sorted = x_sorted[x_order]
        cep = cep[x_order]
        ci[0] = ci[0][x_order]
        ci[1] = ci[1][x_order]

    return cep, x_sorted, ci[0], ci[1]


def point_interval_unique(dt, var_names, group, ci_prob, point_estimate):
    """Compute summary statistics and credible intervals for unique value frequencies.

    For each unique value in the data, computes how often it appears across predictive
    samples, then calculates the mean/median/mode frequency and credible intervals.

    Parameters
    ----------
    dt : DataTree
        DataTree with group ``group`` and "observed_data" groups
    var_names : list of str, optional
        The variables to compute the unique values. If None, uses all variables.
    group : str
        The group from which to get the unique values.
    ci_prob : float
        The probability for the credible interval.
    point_estimate : {"mean", "median", "mode"}
        The statistic to compute. Options: "mean", "median", or "mode".

    Returns
    -------
    Dataset
        Dataset with coordinates for x_values and plot_axis (x, y, y_bottom, y_top)
    """
    pp = extract(dt, group=group, keep_dataset=True)

    if var_names is None:
        var_names = list(dt[group].data_vars)

    piu_data = {}
    for var in var_names:
        if var not in dt[group].data_vars:
            continue

        unique_values = np.unique(pp[var])
        n_samples = pp[var].sizes["sample"]

        counts_array = np.zeros((n_samples, len(unique_values)), dtype=int)
        for i, y in enumerate(unique_values):
            mask = pp[var] == y
            counts_array[:, i] = mask.sum(axis=0).values

        lb = (1 - ci_prob) / 2 * 100
        ci = np.percentile(counts_array, [lb, 100 - lb], axis=0)

        if point_estimate == "mean":
            summary_stat = np.mean(counts_array, axis=0)
        elif point_estimate == "median":
            summary_stat = np.median(counts_array, axis=0)
        elif point_estimate == "mode":
            summary_stat = np.array([np.bincount(col).argmax() for col in counts_array.T])
        else:
            raise ValueError(f"stat must be 'mean', 'median', or 'mode', got '{point_estimate}'")

        piu_data[var] = np.stack([unique_values, summary_stat, ci[0], ci[1]])

    return (
        dict_to_dataset(piu_data)
        .rename({"draw": "x_values", "chain": "plot_axis"})
        .assign_coords({"plot_axis": ["x", "y", "y_bottom", "y_top"]})
    )


def point_unique(dt, var_names):
    """
    Count the occurrences of unique values in observed data.

    Parameters
    ----------
    dt : DataTree
        DataTree with "observed_data" group.
    var_names : list of str, optional
        Variables to compute unique value counts for. If None, uses all.

    Returns
    -------
    Dataset
        Dataset with coordinates for x_values and plot_axis (x, y),
        where x = unique values, y = counts.
    """
    pp = dt["observed_data"]

    if var_names is None:
        var_names = list(pp.data_vars)

    out = {}
    for var in var_names:
        if var not in pp.data_vars:
            continue

        values = pp[var].values
        unique_values, counts = np.unique(values, return_counts=True)

        out[var] = np.stack([unique_values, counts])

    return (
        dict_to_dataset(out)
        .rename({"draw": "x_values", "chain": "plot_axis"})
        .assign_coords(plot_axis=["x", "y"])
    )
