"""Bayes Factor using Savage-Dickey density ratio."""

import warnings

import numpy as np
import xarray as xr
from arviz_base import convert_to_datatree, from_dict


def bayes_factor(data, var_names, ref_vals=0, return_ref_vals=False, prior=None, circular=False):
    """
    Compute Bayes factor using Savage–Dickey ratio.

    Parameters
    ----------
    data : DataTree, or InferenceData
        The data object containing the posterior and optionally the prior distributions.
    var_names : str or list of str
        Names of the variables for which the Bayes factor should be computed.
    ref_vals : float or list of float, default 0
        Reference value for each variable. Must match var_names in length if list.
    return_ref_vals : bool, default False
        If True, return the reference density values for the posterior and prior.
    prior : dict, optional
        Dictionary with prior distributions for each variable of interest. If not provided,
        the prior will be taken from the `prior` group in the data object.
    circular : bool, default False
        Whether the variables are circular (e.g. angles). This affects KDE computation,
        which is used to estimate the density at the reference value.

    Returns
    -------
    xr.Dataset
        Dataset with one variable per requested variable. Each DataArray has a
        ``bf_type`` dimension with coordinates ``["BF10", "BF01"]``, plus any
        non-sample coordinates of the original variable (e.g. ``school``).

    References
    ----------
    .. [1] Heck DW. *A caveat on the Savage-Dickey density ratio:
       The case of computing Bayes factors for regression parameters.*
       Br J Math Stat Psychol, 72. (2019) https://doi.org/10.1111/bmsp.12150

    Examples
    --------
    Compute Bayes factor for a home and intercept variable in a rugby dataset
    using a reference value of 0.15 for home and 3 for intercept.

    .. ipython::

        In [1]: from arviz_base import load_arviz_data
           ...: from arviz_stats import bayes_factor
           ...: dt = load_arviz_data("rugby")
           ...: bayes_factor(dt, var_names=["home", "intercept"], ref_vals=[0.15, 3])
    """
    data = convert_to_datatree(data)

    if isinstance(var_names, str):
        var_names = [var_names]

    if isinstance(ref_vals, int | float):
        ref_vals = [ref_vals] * len(var_names)

    if len(var_names) != len(ref_vals):
        raise ValueError("Length of var_names and ref_vals must match.")

    results = {}
    ref_density_vals = {}

    for var, ref_val in zip(var_names, ref_vals):
        if not isinstance(ref_val, int | float):
            raise ValueError(f"Reference value for variable '{var}' must be numerical")

    if prior is not None:
        if isinstance(prior, dict):
            prior_ds = from_dict({"prior": prior}).prior.dataset
        else:
            prior_ds = prior if isinstance(prior, xr.Dataset) else prior.to_dataset()
    else:
        prior_ds = data.prior.dataset

    posterior_kde = data.posterior.dataset[var_names].azstats.kde(grid_len=512, circular=circular)
    prior_kde = prior_ds[var_names].azstats.kde(grid_len=512, circular=circular)

    for var, ref_val in zip(var_names, ref_vals):
        if ref_val > data.posterior[var].max() or ref_val < data.posterior[var].min():
            warnings.warn(
                f"Reference value {ref_val} for '{var}' is outside the posterior range. "
                "This may overstate evidence in favor of H1."
            )

        if ref_val > prior_ds[var].max() or ref_val < prior_ds[var].min():
            warnings.warn(
                f"Reference value {ref_val} for '{var}' is outside the prior range. "
                "Bayes factor computation is not reliable."
            )

        posterior_val = _eval_kde_at_ref(posterior_kde[var], ref_val)
        prior_val = _eval_kde_at_ref(prior_kde[var], ref_val)

        if (prior_val <= 0).any() or (posterior_val <= 0).any():
            raise ValueError(
                f"Invalid KDE values at ref_val={ref_val}: "
                f"prior={prior_val.values}, posterior={posterior_val.values}"
            )

        bf_10 = prior_val / posterior_val
        bf_01 = 1 / bf_10
        results[var] = xr.concat(
            [bf_10, bf_01],
            dim=xr.DataArray(["BF10", "BF01"], dims="bf_type"),
        ).rename(var)

        if return_ref_vals:
            ref_density_vals[var] = xr.concat(
                [posterior_val, prior_val],
                dim=xr.DataArray(["posterior", "prior"], dims="density_type"),
            ).rename(var)

    result_ds = xr.Dataset(results)

    if return_ref_vals:
        return result_ds, xr.Dataset(ref_density_vals)

    return result_ds


def _eval_kde_at_ref(kde_da, ref_val):
    """Evaluate KDE at ref_val for every coordinate combination (e.g. per school)."""
    x = kde_da.sel(plot_axis="x")
    y = kde_da.sel(plot_axis="y")
    return xr.apply_ufunc(
        lambda xi, yi: np.interp(ref_val, xi, yi),
        x,
        y,
        input_core_dims=[["kde_dim"], ["kde_dim"]],
        vectorize=True,
    )
