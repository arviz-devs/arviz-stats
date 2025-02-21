"""Power-scaling sensitivity diagnostics."""

import logging
from typing import cast

import numpy as np
import pandas as pd
import xarray as xr
from arviz_base import convert_to_datatree, extract
from arviz_base.labels import BaseLabeller
from arviz_base.sel_utils import xarray_var_iter

from arviz_stats.utils import get_log_likelihood_dataset, get_log_prior
from arviz_stats.validate import validate_dims

_log = logging.getLogger(__name__)

labeller = BaseLabeller()

__all__ = ["psense", "psense_summary"]


def psense(
    data,
    var_names=None,
    filter_vars=None,
    group="prior",
    coords=None,
    sample_dims=None,
    alphas=(0.99, 1.01),
    group_var_names=None,
    group_coords=None,
):
    """
    Compute power-scaling sensitivity values.

    Parameters
    ----------
    data : DataTree or InferenceData
        Input data. It should contain the posterior and the log_likelihood and/or log_prior groups.
    var_names : list of str, optional
        Names of posterior variables to include in the power scaling sensitivity diagnostic
    filter_vars: {None, "like", "regex"}, default None
        Used for `var_names` only.
        If ``None`` (default), interpret var_names as the real variables names.
        If "like", interpret var_names as substrings of the real variables names.
        If "regex", interpret var_names as regular expressions on the real variables names.
    group : {"prior", "likelihood"}, default "prior"
        If "likelihood", the pointsize log likelihood values are retrieved
        from the ``log_likelihood`` group and added together.
        If "prior", the log prior values are retrieved from the ``log_prior`` group.
    coords : dict, optional
        Coordinates defining a subset over the posterior. Only these variables will
        be used when computing the prior sensitivity.
    sample_dims : str or sequence of hashable, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to ``rcParams["data.sample_dims"]``
    alphas : tuple
        Lower and upper alpha values for gradient calculation. Defaults to (0.99, 1.01).
    group_var_names : str, optional
        Name of the prior or log likelihood variables to use
    group_coords : dict, optional
        Coordinates defining a subset over the group element for which to
        compute the prior sensitivity diagnostic.

    Returns
    -------
    xarray.DataTree
        Returns dataTree of power-scaling sensitivity diagnostic values.
        Higher sensitivity values indicate greater sensitivity.
        Prior sensitivity above 0.05 indicates informative prior.
        Likelihood sensitivity below 0.05 indicates weak or non-informative likelihood.

    Notes
    -----
    The diagnostic is computed by power-scaling either the prior or likelihood
    and determining the degree to which the posterior changes as described in [1]_.
    It uses Pareto-smoothed importance sampling to avoid refitting the model.

    References
    ----------
    .. [1] Kallioinen et al, *Detecting and diagnosing prior and likelihood sensitivity with
       power-scaling*, Stat Comput 34, 57 (2024), https://doi.org/10.1007/s11222-023-10366-5
    """
    data = convert_to_datatree(data)

    dataset = extract(
        data,
        var_names=var_names,
        filter_vars=filter_vars,
        group="posterior",
        combined=False,
        keep_dataset=True,
    )
    if coords is not None:
        dataset = dataset.sel(coords)

    lower_w, upper_w = _get_power_scale_weights(
        data,
        alphas=alphas,
        group=group,
        sample_dims=sample_dims,
        group_var_names=group_var_names,
        group_coords=group_coords,
    )

    return dataset.azstats.power_scale_sense(
        lower_w=lower_w,
        upper_w=upper_w,
        lower_alpha=alphas[0],
        upper_alpha=alphas[1],
        dims=sample_dims,
    )


def psense_summary(
    data,
    var_names=None,
    filter_vars=None,
    coords=None,
    sample_dims=None,
    threshold=0.05,
    alphas=(0.99, 1.01),
    prior_var_names=None,
    likelihood_var_names=None,
    prior_coords=None,
    likelihood_coords=None,
    round_to=3,
):
    """
    Compute the prior/likelihood sensitivity based on power-scaling perturbations.

    Parameters
    ----------
    data : DataTree or InferenceData
        Input data. It should contain the posterior and the log_likelihood and/or log_prior groups.
    var_names : list of str, optional
        Names of posterior variables to include in the power scaling sensitivity diagnostic
    filter_vars: {None, "like", "regex"}, default None
        Used for `var_names` only.
        If ``None`` (default), interpret var_names as the real variables names.
        If "like", interpret var_names as substrings of the real variables names.
        If "regex", interpret var_names as regular expressions on the real variables names.
    coords : dict, optional
        Coordinates defining a subset over the posterior. Only these variables will
        be used when computing the prior sensitivity.
    sample_dims : str or sequence of hashable, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to ``rcParams["data.sample_dims"]``
    threshold : float, optional
        Threshold value to determine the sensitivity diagnosis. Default is 0.05.
    alphas : tuple
        Lower and upper alpha values for gradient calculation. Defaults to (0.99, 1.01).
    prior_var_names : str, optional
        Name of the log-prior variables to include in the power scaling sensitivity diagnostic
    likelihood_var_names : str, optional
        Name of the log-likelihood variables to include in the power scaling sensitivity diagnostic
    prior_coords : dict, optional
        Coordinates defining a subset over the group element for which to
        compute the log-prior sensitivity diagnostic
    likelihood_coords : dict, optional
        Coordinates defining a subset over the group element for which to
        compute the log-likelihood sensitivity diagnostic
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

    Examples
    --------
    .. ipython::

        In [1]: from arviz_base import load_arviz_data
           ...: from arviz_stats import psense_summary
           ...: rugby = load_arviz_data("rugby")
           ...: psense_summary(rugby, var_names="atts")

    Notes
    -----
    The diagnostic is computed by power-scaling either the prior or likelihood
    and determining the degree to which the posterior changes as described in [1]_.
    It uses Pareto-smoothed importance sampling to avoid refitting the model.

    References
    ----------
    .. [1] Kallioinen et al, *Detecting and diagnosing prior and likelihood sensitivity with
       power-scaling*, Stat Comput 34, 57 (2024), https://doi.org/10.1007/s11222-023-10366-5
    """
    pssdp = psense(
        data,
        var_names=var_names,
        filter_vars=filter_vars,
        group="prior",
        sample_dims=sample_dims,
        coords=coords,
        alphas=alphas,
        group_var_names=prior_var_names,
        group_coords=prior_coords,
    )
    pssdl = psense(
        data,
        var_names=var_names,
        filter_vars=filter_vars,
        group="likelihood",
        coords=coords,
        sample_dims=sample_dims,
        alphas=alphas,
        group_var_names=likelihood_var_names,
        group_coords=likelihood_coords,
    )

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
            return "potential prior-data conflict"
        if row["prior"] > threshold > row["likelihood"]:
            return "potential strong prior / weak likelihood"

        return "✓"

    psense_df["diagnosis"] = psense_df.apply(_diagnose, axis=1)

    if "potential" in "".join(psense_df["diagnosis"]):
        _log.warning(
            "We detected potential issues. For more information on how to interpret the results, "
            "please check\n"
            "https://arviz-devs.github.io/Exploratory-Analysis-of-Bayesian-Models/Chapters/"
            "Sensitivity_checks.html#interpreting-sensitivity-diagnostics-summary\n"
            "or read original paper https://doi.org/10.1007/s11222-023-10366-5"
        )

    return psense_df.round(round_to)


def power_scale_dataset(data, group, alphas, sample_dims, group_var_names, group_coords):
    """Resample posterior based on power-scaled weights.

    Parameters
    ----------
    data : DataTree or InferenceData
        Input data. It should contain the posterior and the log_likelihood and/or log_prior groups.
    group : str
        Group to resample. Either "prior" or "likelihood"
    alphas : tuple of float
        Lower and upper alpha values for power scaling.
    sample_dims : str or sequence of hashable
        Dimensions to reduce unless mapped to an aesthetic.
    group_var_names : str
        Name of the log-prior or log-likelihood variables to use.
    group_coords : dict
        Coordinates defining a subset over the group element for which to
        compute the sensitivity diagnostic.

    Returns
    -------
    DataSet with resampled data.
    """
    dt = convert_to_datatree(data)

    lower_w, upper_w = _get_power_scale_weights(
        dt,
        alphas,
        group=group,
        sample_dims=sample_dims,
        group_var_names=group_var_names,
        group_coords=group_coords,
    )

    lower_w = lower_w.values.flatten()
    upper_w = upper_w.values.flatten()
    s_size = len(lower_w)

    idxs_to_drop = sample_dims if len(sample_dims) == 1 else ["sample"] + sample_dims
    idxs_to_drop = set(idxs_to_drop).union(
        [
            idx
            for idx in dt["posterior"].xindexes
            if any(dim in dt["posterior"][idx].dims for dim in sample_dims)
        ]
    )
    resampled = [
        extract(
            dt,
            group="posterior",
            sample_dims=sample_dims,
            num_samples=s_size,
            weights=weights,
            random_seed=42,
            resampling_method="stratified",
        ).drop_indexes(idxs_to_drop)
        for weights in (lower_w, upper_w)
    ]
    resampled.insert(
        1, extract(dt, group="posterior", sample_dims=sample_dims).drop_indexes(idxs_to_drop)
    )

    return xr.concat(resampled, dim="alpha").assign_coords(alpha=[alphas[0], 1, alphas[1]])


def _get_power_scale_weights(
    dt, alphas=None, group=None, sample_dims=None, group_var_names=None, group_coords=None
):
    """Compute power scale weights."""
    sample_dims = validate_dims(sample_dims)

    if group == "likelihood":
        group_draws = get_log_likelihood_dataset(dt, var_names=group_var_names)
    elif group == "prior":
        group_draws = get_log_prior(dt, var_names=group_var_names)
    else:
        raise ValueError("Value for `group` argument not recognized")

    if group_coords is not None:
        group_draws = group_draws.sel(group_coords)
    # we stack the different variables (if any) and dimensions in each variable (if any)
    # into a flat dimension "latent-obs_var", over which we sum afterwards.
    # Consequently, after this group_draws draws is a dataarray with only sample_dims as dims
    group_draws = group_draws.to_stacked_array("latent-obs_var", sample_dims=sample_dims).sum(
        "latent-obs_var"
    )

    # calculate importance sampling weights for lower and upper alpha power-scaling
    lower_w = np.exp(group_draws.azstats.power_scale_lw(alpha=alphas[0], dims=sample_dims))
    lower_w = lower_w / lower_w.sum(sample_dims)

    upper_w = np.exp(group_draws.azstats.power_scale_lw(alpha=alphas[1], dims=sample_dims))
    upper_w = upper_w / upper_w.sum(sample_dims)

    return lower_w, upper_w
