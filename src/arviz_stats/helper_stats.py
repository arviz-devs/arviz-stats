"""
Auxiliary statistical functions
"""

import numpy as np
from scipy.optimize import isotonic_regression
from scipy import interpolate

from arviz_base import extract, rcParams


def isotonic_fit(pred, obs, n_bootstrap, ci_prob=None):
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

    # calculate the fitting function
    unique_pred_sorted, pred_counts = np.unique(pred, return_counts=True)
    ir_func = interpolate.interp1d(pred, y_out, bounds_error=False)
    regression_values = ir_func(unique_pred_sorted)

    # bootstrap the isotonic regression
    result = np.zeros((n_bootstrap, len(pred_counts)))
    for i in range(n_bootstrap):
        idx = np.random.choice(len(pred), len(pred), replace=True)
        pred_boot, obs_boot = _sort_pred_with_obs(pred[idx], obs[idx])
        y_out_boot = isotonic_regression(obs_boot, increasing=True).x

        result[i] = interpolate.interp1d(pred_boot, y_out_boot, bounds_error=False)(unique_pred_sorted)


    # we should compute hdi or eti using already implemented functions
    lb = (1 - ci_prob) / 2 * 100
    ub = (1 + ci_prob) / 2 * 100
    regression_values_boot = np.percentile(result, [lb, ub], axis=0)

    return regression_values, unique_pred_sorted, regression_values_boot



def _sort_pred_with_obs(pred, obs):
    """ Ensure that ties in pred have same regression value in PAV algorithm"""
    sorter = np.lexsort((-obs, pred))
    pred = pred[sorter]
    obs = obs[sorter]

    return pred, obs