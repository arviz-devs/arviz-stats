"""Circular statistics utility functions."""

import numpy as np


def circular_diff(array0, array1):
    """Compute wrapped angular difference in [-π, π).

    Parameters
    ----------
    array0 : np.ndarray
        First array of angles in radians.
    array1 : np.ndarray
        Second array of angles in radians.

    Returns
    -------
    np.ndarray
        Wrapped angular difference in [-π, π).
    """
    return ((array0 - array1 + np.pi) % (2 * np.pi)) - np.pi


def circular_mean(angles, weights=None, dims=None):
    """Compute weighted circular mean for angles in [-π, π].

    Parameters
    ----------
    angles : np.ndarray
        Array of angles in radians, shape (n_samples, n_angles).
    weights : np.ndarray, optional
        Weights for each angle, shape (n_samples, n_angles). If None, equal weights are used.
        Weights should be non-negative and sum to 1 along axis 1.
    dims : str or list of str, optional
        Dimension(s) to sum over. If None, uses axis=1 for numpy arrays.
    """
    if weights is None:
        if dims is not None:
            weights = angles * 0 + 1
            weights = weights / weights.sum(dim=dims)
        else:
            weights = np.ones_like(angles) / angles.shape[1]

    if dims is not None:
        sum_sin = (weights * np.sin(angles)).sum(dim=dims)
        sum_cos = (weights * np.cos(angles)).sum(dim=dims)
    else:
        sum_sin = (weights * np.sin(angles)).sum(axis=1)
        sum_cos = (weights * np.cos(angles)).sum(axis=1)

    return np.arctan2(sum_sin, sum_cos)


def mean_resultant_length(angles, weights=None, dims=None):
    """Compute weighted mean resultant length for angles in [-π, π].

    Parameters
    ----------
    angles : np.ndarray
        Array of angles in radians, shape (n_samples, n_angles).
    weights : np.ndarray, optional
        Weights for each angle, shape (n_samples, n_angles). If None, equal weights are used.
        Weights should be non-negative and sum to 1 along axis 1.
    dims : str or list of str, optional
        Dimension(s) to sum over. If None, uses axis=1 for numpy arrays.
    """
    if weights is None:
        if dims is not None:
            weights = angles * 0 + 1
            weights = weights / weights.sum(dim=dims)
        else:
            weights = np.ones_like(angles) / angles.shape[1]

    if dims is not None:
        mean_cos = (weights * np.cos(angles)).sum(dim=dims)
        mean_sin = (weights * np.sin(angles)).sum(dim=dims)
    else:
        mean_cos = (weights * np.cos(angles)).sum(axis=1)
        mean_sin = (weights * np.sin(angles)).sum(axis=1)

    return np.sqrt(mean_cos**2 + mean_sin**2)


def circular_var(angles, weights=None, dims=None):
    """Compute weighted circular variance for angles in [-π, π].

    Parameters
    ----------
    angles : np.ndarray
        Array of angles in radians, shape (n_samples, n_angles).
    weights : np.ndarray, optional
        Weights for each angle, shape (n_samples, n_angles). If None, equal weights are used.
        Weights should be non-negative and sum to 1 along axis 1.
    dims : str or list of str, optional
        Dimension(s) to sum over. If None, uses axis=1 for numpy arrays.

    """
    return 1 - mean_resultant_length(angles, weights, dims)


def circular_sd(angles, weights=None, dims=None):
    """Compute weighted circular standard deviation for angles in [-π, π].

    The circular standard deviation is not the square root of the circular variance.
    It ranges from 0 (all angles identical) to infinity (uniform distribution), though
    finite precision limits the maximum to approximately 8.5 radians.

    Parameters
    ----------
    angles : np.ndarray
        Array of angles in radians, shape (n_samples, n_angles).
    weights : np.ndarray, optional
        Weights for each angle, shape (n_samples, n_angles). If None, equal weights are used.
        Weights should be non-negative and sum to 1 along axis 1.
    dims : str or list of str, optional
        Dimension(s) to sum over. If None, uses axis=1 for numpy arrays.
    """
    with np.errstate(divide="ignore"):
        return np.sqrt(-2 * np.log(mean_resultant_length(angles, weights, dims)))
