"""Power-scaling sensitivity diagnostics."""

from typing import cast

import numpy as np
import pandas as pd
import xarray as xr
from arviz_base import extract
from arviz_base.labels import BaseLabeller
from arviz_base.sel_utils import xarray_var_iter

from arviz_stats.utils import get_log_likelihood_dataset, get_log_prior
from arviz_stats.validate import validate_dims

labeller = BaseLabeller()

__all__ = ["psense", "psense_summary"]


def psense(
    dt,
    group="log_prior",
    sample_dims=None,
    component_var_names=None,
    component_coords=None,
    var_names=None,
    coords=None,
    filter_vars=None,
    delta=0.01,
):
    """
    Compute power-scaling sensitivity values.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an :class:`arviz.InferenceData` object.
        Refer to documentation of :func:`arviz.convert_to_dataset` for details.
        For ndarray: shape = (chain, draw).
        For n-dimensional ndarray transform first to dataset with ``az.convert_to_dataset``.
    group : {"log_prior", "log_likelihood"}, default "log_prior"
        When `component` is "likelihood", the log likelihood values are retrieved
        from the ``log_likelihood`` group as pointwise log likelihood and added
        together. With "prior", the log prior values are retrieved from the
        ``log_prior`` group.
    component_var_names : str, optional
        Name of the prior or log likelihood variables to use
    component_coords : dict, optional
        Coordinates defining a subset over the component element for which to
        compute the prior sensitivity diagnostic.
    var_names : list of str, optional
        Names of posterior variables to include in the power scaling sensitivity diagnostic
    coords : dict, optional
        Coordinates defining a subset over the posterior. Only these variables will
        be used when computing the prior sensitivity.
    filter_vars: {None, "like", "regex"}, default None
        If ``None`` (default), interpret var_names as the real variables names.
        If "like", interpret var_names as substrings of the real variables names.
        If "regex", interpret var_names as regular expressions on the real variables names.
    delta : float
        Value for finite difference derivative calculation.


    Returns
    -------
    xarray.DataTree
        Returns dataTree of power-scaling sensitivity diagnostic values.
        Higher sensitivity values indicate greater sensitivity.
        Prior sensitivity above 0.05 indicates informative prior.
        Likelihood sensitivity below 0.05 indicates weak or non-informative likelihood.

    Notes
    -----
    The diagnostic is computed by power-scaling the specified component (prior or likelihood)
    and determining the degree to which the posterior changes as described in [1]_.
    It uses Pareto-smoothed importance sampling to avoid refitting the model.

    References
    ----------
    .. [1] Kallioinen et al, *Detecting and diagnosing prior and likelihood sensitivity with
       power-scaling*, 2022, https://arxiv.org/abs/2107.14054
    """
    dataset = extract(
        dt, var_names=var_names, filter_vars=filter_vars, group="posterior", combined=False
    )
    sample_dims = validate_dims(sample_dims)
    if coords is not None:
        dataset = dataset.sel(coords)

    if group == "log_likelihood":
        component_draws = get_log_likelihood_dataset(dt, var_names=component_var_names)
    elif group == "log_prior":
        component_draws = get_log_prior(dt, var_names=component_var_names)
    else:
        raise ValueError("Value for `group` argument not recognized")

    if component_coords is not None:
        component_draws = component_draws.sel(component_coords)
    # we stack the different variables (if any) and dimensions in each variable (if any)
    # into a flat dimension "latent-obs_var", over which we sum afterwards.
    # Consequently, after this component_draws draws is a dataarray with only sample_dims as dims
    component_draws = component_draws.to_stacked_array(
        "latent-obs_var", sample_dims=sample_dims
    ).sum("latent-obs_var")

    # calculate lower and upper alpha values
    lower_alpha = 1 / (1 + delta)
    upper_alpha = 1 + delta

    # calculate importance sampling weights for lower and upper alpha power-scaling
    lower_w = np.exp(component_draws.azstats.power_scale_lw(alpha=lower_alpha, dims=sample_dims))
    lower_w = lower_w / lower_w.sum(sample_dims)

    upper_w = np.exp(component_draws.azstats.power_scale_lw(alpha=upper_alpha, dims=sample_dims))
    upper_w = upper_w / upper_w.sum(sample_dims)

    return dataset.azstats.power_scale_sense(
        lower_w=lower_w,
        upper_w=upper_w,
        delta=delta,
        dims=sample_dims,
    )


def psense_summary(data, threshold=0.05, round_to=3):
    """
    Compute the prior/likelihood sensitivity based on power-scaling perturbations.

    Parameters
    ----------
    data : DataTree
    threshold : float, optional
        Threshold value to determine the sensitivity diagnosis. Default is 0.05.
    round_to : int, optional
        Number of decimal places to round the sensitivity values. Default is 3.

    Returns
    -------
    psense_df : DataFrame
        DataFrame containing the prior and likelihood sensitivity values for each variable
        in the data. And a diagnosis column with the following values:
        - "prior-data conflict" if both prior and likelihood sensitivity are above threshold
        - "strong prior / weak likelihood" if the prior sensitivity is above threshold
        and the likelihood sensitivity is below the threshold
        - "-" otherwise
    """
    pssdp = psense(data, group="log_prior")
    pssdl = psense(data, group="log_likelihood")

    joined = xr.concat([pssdp, pssdl], dim="component").assign_coords(
        component=["prior", "likelihood"]
    )

    n_vars = np.sum([joined[var].size // 2 for var in joined.data_vars])

    psense_df = pd.DataFrame(
        (np.full((cast(int, n_vars), 2), np.nan)), columns=["prior", "likelihood"]
    )

    indices = []
    for i, (var_name, sel, isel, values) in enumerate(
        xarray_var_iter(joined, skip_dims={"component"})
    ):
        psense_df.iloc[i] = values
        indices.append(labeller.make_label_flat(var_name, sel, isel))
    psense_df.index = indices

    def _diagnose(row):
        if row["prior"] >= threshold and row["likelihood"] >= threshold:
            return "prior-data conflict"
        if row["prior"] > threshold > row["likelihood"]:
            return "strong prior / weak likelihood"

        return "-"

    psense_df["diagnosis"] = psense_df.apply(_diagnose, axis=1)

    return psense_df.round(round_to)
