"""Power-scaling sensitivity diagnostics."""

from typing import cast

import numpy as np
import pandas as pd
import xarray as xr
from arviz_base import extract
from arviz_base.labels import BaseLabeller
from arviz_base.sel_utils import xarray_var_iter 
from scipy.stats import gaussian_kde
import logging
# from arviz_stats.base.density import _DensityBase
 
from arviz_stats.utils import get_log_likelihood_dataset, get_log_prior
from arviz_stats.validate import validate_dims

labeller = BaseLabeller()

__all__ = ["psense", "psense_summary", "bayes_factor"]


def psense(
    dt,
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
    dt : obj
        Any object that can be converted to an :class:arviz.InferenceData object.
        Refer to documentation of :func:arviz.convert_to_dataset for details.
        For ndarray: shape = (chain, draw).
        For n-dimensional ndarray transform first to dataset with `az.convert_to_dataset.
    var_names : list of str, optional
        Names of posterior variables to include in the power scaling sensitivity diagnostic
    filter_vars: {None, "like", "regex"}, default None
        Used for var_names only.
        If `None (default), interpret var_names as the real variables names.
        If "like", interpret var_names as substrings of the real variables names.
        If "regex", interpret var_names as regular expressions on the real variables names.
    group : {"prior", "likelihood"}, default "prior"
        If "likelihood", the pointsize log likelihood values are retrieved
        from the `log_likelihood group and added together.
        If "prior", the log prior values are retrieved from the `log_prior group.
    coords : dict, optional
        Coordinates defining a subset over the posterior. Only these variables will
        be used when computing the prior sensitivity.
    sample_dims : str or sequence of hashable, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to `rcParams["data.sample_dims"]
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
    dataset = extract(
        dt,
        var_names=var_names,
        filter_vars=filter_vars,
        group="posterior",
        combined=False,
        keep_dataset=True,
    )
    if coords is not None:
        dataset = dataset.sel(coords)

    lower_w, upper_w = _get_power_scale_weights(
        dt,
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
    group_var_names=None,
    group_coords=None,
    round_to=3,
):
    """
    Compute the prior/likelihood sensitivity based on power-scaling perturbations.

    Parameters
    ----------
    data : DataTree
    var_names : list of str, optional
        Names of posterior variables to include in the power scaling sensitivity diagnostic
    filter_vars: {None, "like", "regex"}, default None
        Used for var_names only.
        If `None (default), interpret var_names as the real variables names.
        If "like", interpret var_names as substrings of the real variables names.
        If "regex", interpret var_names as regular expressions on the real variables names.
    coords : dict, optional
        Coordinates defining a subset over the posterior. Only these variables will
        be used when computing the prior sensitivity.
    sample_dims : str or sequence of hashable, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to `rcParams["data.sample_dims"]
    threshold : float, optional
        Threshold value to determine the sensitivity diagnosis. Default is 0.05.
    alphas : tuple
        Lower and upper alpha values for gradient calculation. Defaults to (0.99, 1.01).
    group_var_names : str, optional
        Name of the prior or log likelihood variables to use
    group_coords : dict, optional
        Coordinates defining a subset over the group element for which to
        compute the prior sensitivity diagnostic
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
    """
    pssdp = psense(
        data,
        var_names=var_names,
        filter_vars=filter_vars,
        group="prior",
        sample_dims=sample_dims,
        coords=coords,
        alphas=alphas,
        group_var_names=group_var_names,
        group_coords=group_coords,
    )
    pssdl = psense(
        data,
        var_names=var_names,
        filter_vars=filter_vars,
        group="likelihood",
        coords=coords,
        sample_dims=sample_dims,
        alphas=alphas,
        group_var_names=group_var_names,
        group_coords=group_coords,
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
            return "prior-data conflict"
        if row["prior"] > threshold > row["likelihood"]:
            return "strong prior / weak likelihood"

        return "✓"

    psense_df["diagnosis"] = psense_df.apply(_diagnose, axis=1)

    return psense_df.round(round_to)


def power_scale_dataset(dt, group, alphas, sample_dims):
    """Resample the dataset with the power scale weights.

    Parameters
    ----------
    dt : DataSet
    group : str
        Group to resample. Either "prior" or "likelihood"
    alphas : tuple of float
        Lower and upper alpha values for power scaling.
    sample_dims : str or sequence of hashable
        Dimensions to reduce unless mapped to an aesthetic.

    Returns
    -------
    DataSet with resampled data.
    """
    lower_w, upper_w = _get_power_scale_weights(dt, alphas, group=group, sample_dims=sample_dims)
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
        raise ValueError("Value for group argument not recognized")

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



# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def bayes_factor(idata, var_name, ref_val=0, prior=None, return_ref_vals=False):
    """
    Approximates the Bayes Factor for comparing hypotheses of two nested models.
    
    The Bayes factor compares a model (H1) against a model where the parameter
    is constrained to a point-null (H0). This uses the Savage-Dickey density ratio method.
    
    Parameters
    ----------
    idata : InferenceData
        Object containing posterior and prior data.
    var_name : str
        Name of the variable to test.
    ref_val : int, default 0
        Reference (point-null) value for Bayes factor estimation.
    prior : numpy.array, optional
        Custom prior for sensitivity analysis. Defaults to prior extracted from idata.
    return_ref_vals : bool, default False
        If True, also return the values of prior and posterior densities at the reference value.
    
    Returns
    -------
    dict
        A dictionary with Bayes Factor values: BF10 (H1/H0 ratio) and BF01 (H0/H1 ratio).
    """
    # Extract the posterior values for the specified variable
    posterior = extract(idata, var_names=var_name).values

    # Warn if the reference value is outside the range of the posterior
    if ref_val > posterior.max() or ref_val < posterior.min():
        logger.warning(
            "The reference value is outside of the posterior range. "
            "This results in infinite support for H1, which may overstate evidence."
        )

    # Warn if the posterior has more than one dimension
    if posterior.ndim > 1:
        logger.warning(f"Posterior distribution has {posterior.ndim} dimensions.")

    # Use the default prior if none is provided
    if prior is None:
        prior = extract(idata, var_names=var_name, group="prior").values

    # Handle continuous data with Gaussian KDE
    if posterior.dtype.kind == "f":
        kde_posterior = gaussian_kde(posterior)
        kde_prior = gaussian_kde(prior)

        # Generate grids for interpolation
        posterior_grid = np.linspace(min(posterior), max(posterior), 1000)
        prior_grid = np.linspace(min(prior), max(prior), 1000)

        # Evaluate PDF on the grid
        posterior_pdf = kde_posterior(posterior_grid)
        prior_pdf = kde_prior(prior_grid)

        # Get the density at the reference value
        posterior_at_ref_val = np.interp(ref_val, posterior_grid, posterior_pdf)
        prior_at_ref_val = np.interp(ref_val, prior_grid, prior_pdf)

    # Handle discrete data
    elif posterior.dtype.kind == "i":
        posterior_at_ref_val = (posterior == ref_val).mean()
        prior_at_ref_val = (prior == ref_val).mean()

    else:
        raise ValueError("Unsupported data type for posterior/prior.")

    # Compute Bayes Factor
    bf_10 = prior_at_ref_val / posterior_at_ref_val
    bf = {"BF10": bf_10, "BF01": 1 / bf_10}

    # Optionally return prior and posterior values at the reference value
    if return_ref_vals:
        return (bf, {"prior": prior_at_ref_val, "posterior": posterior_at_ref_val})
    else:
        return bf