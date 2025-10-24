"""Survival analysis utilities."""

import warnings

import numpy as np
from arviz_base import dict_to_dataset, extract


def kaplan_meier(
    dt,
    var_names,
    group="observed_data",
):
    """Compute Kaplan-Meier survival curves for observed data.

    Parameters
    ----------
    dt: DataTree
        DataTree with "posterior_predictive" and "observed_data" groups
    var_names : str or list of str
        The variables to compute the unique values.
    group : str
        The group from which to get the unique values.
    """
    if isinstance(var_names, str):
        var_names = [var_names]
    else:
        var_names = list(var_names)

    pp = extract(dt, group=group, var_names=var_names, keep_dataset=True, combined=False)
    try:
        constant_data = dt["constant_data"].dataset
    except KeyError:
        warnings.warn(
            "No 'constant_data' group found in DataTree. Assuming all events are observed."
        )
        constant_data = []

    dictio = {}

    for var_name in var_names:
        var = pp[var_name].values.flatten()
        if constant_data:
            if var_name in constant_data.data_vars:
                status = constant_data[var_name].values.flatten()
            else:
                warnings.warn(
                    f"No status variable found for {var_name} in 'constant_data'. "
                    "Assuming all events are observed."
                )
                status = np.ones_like(var)
        else:
            status = np.ones_like(var)
        sorted_indices = np.argsort(var)
        sorted_times = var[sorted_indices]
        sorted_status = status[sorted_indices]

        survival_probs = []
        survival_times = []
        current_prob = 1.0

        unique_times = np.unique(sorted_times)

        for t in unique_times:
            n_events = np.sum((sorted_times == t) & (sorted_status == 1))
            n_at_risk_t = np.sum(sorted_times >= t)

            if n_at_risk_t > 0 and n_events > 0:
                current_prob *= 1 - n_events / n_at_risk_t

            survival_times.append(t)
            survival_probs.append(current_prob)

        dictio[var_name] = np.stack([survival_times, survival_probs])

    return (
        dict_to_dataset(dictio)
        .rename({"draw": "km_points", "chain": "plot_axis"})
        .assign_coords({"plot_axis": ["x", "y"]})
    )


def generate_survival_curves(
    dt,
    var_names,
    group="posterior_predictive",
    num_samples=None,
    truncation_factor=1.2,
):
    """Compute Kaplan-Meier curves for predictive samples.

    Parameters
    ----------
    dt : DataTree
        DataTree with posterior_predictive and observed_data groups
    var_names : str or list of str
        The variables to compute the survival curves for.
    group : str, default "posterior_predictive"
        Group containing the predictive samples.
    num_samples : int, optional
        Number of samples to draw from the predictive distribution.
    truncation_factor : float, default 1.2
        Factor by which truncates the survival curves beyond the maximum observed time.
        Set to None to show all posterior predictive draws.
    """
    if isinstance(var_names, str):
        var_names = [var_names]
    else:
        var_names = list(var_names)
    # Extract predictive data
    pp = extract(dt, group=group, var_names=var_names, num_samples=num_samples, keep_dataset=True)
    obs_data = dt["observed_data"].dataset

    dictio = {}

    for var_name in pp.data_vars:
        # Handle extrapolation - get max observed time if needed
        max_observed_time = np.inf
        if truncation_factor is not None:
            if var_name in obs_data.data_vars:
                max_observed_time = obs_data[var_name].max().values

        # Find the maximum number of points across all samples first
        max_points = 0
        sample_data_list = []

        for i in range(num_samples):
            times = pp[var_name].isel(sample=i)

            # Filter times based on extrapolation factor
            if truncation_factor is not None:
                times = times[times <= max_observed_time * truncation_factor]

            # Skip if no times left after filtering
            if len(times) == 0:
                continue

            unique_times = np.unique(np.sort(times))
            max_points = max(max_points, len(unique_times))
            sample_data_list.append((i, times))

        # Now create aligned arrays
        n_valid_samples = len(sample_data_list)
        times_array = np.full((n_valid_samples, max_points), np.nan)
        probs_array = np.full((n_valid_samples, max_points), np.nan)

        for idx, (_, times) in enumerate(sample_data_list):
            # Create empirical survival function for this sample
            sorted_times = np.sort(times)
            unique_times = np.unique(sorted_times)

            for j, t in enumerate(unique_times):
                times_array[idx, j] = t
                probs_array[idx, j] = 1 - (j + 1) / len(unique_times)

        dictio[var_name] = np.stack([times_array, probs_array], axis=0)

    return (
        dict_to_dataset(dictio)
        .rename({"draw": "sample", "chain": "plot_axis"})
        .assign_coords({"plot_axis": ["x", "y"]})
    )
