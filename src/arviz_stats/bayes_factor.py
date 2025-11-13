"""Bayes Factor using Savage-Dickey density ratio."""

import warnings

from arviz_base import convert_to_datatree
from numpy import finfo
from scipy.interpolate import interp1d


def bayes_factor(data, var_names, ref_vals=0, return_ref_vals=False, prior=None):
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
        Dictionary with prior distributions for each variable. If not provided,
        the prior will be taken from the `prior` group in the data object.

    Returns
    -------
    dict
        Dictionary with Bayes Factor values: BF10 and BF01 per variable.

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
        posterior = data.posterior[var]
        prior = data.prior[var]

        if not isinstance(ref_val, int | float):
            raise ValueError(f"Reference value for variable '{var}' must be numerical")

        if ref_val > posterior.max() or ref_val < posterior.min():
            warnings.warn(
                f"Reference value {ref_val} for '{var}' is outside the posterior range. "
                "This may overstate evidence in favor of H1."
            )

        if ref_val > prior.max() or ref_val < prior.min():
            warnings.warn(
                f"Reference value {ref_val} for '{var}' is outside the prior range. "
                "Bayes factor computation is not reliable."
            )

        posterior_kde = posterior.azstats.kde(grid_len=512, circular=False)
        prior_kde = prior.azstats.kde(grid_len=512, circular=False)

        posterior_val = interp1d(
            posterior_kde.values[0],
            posterior_kde.values[1],
            bounds_error=False,
            fill_value=finfo("float").eps,
        )(ref_val).item()
        prior_val = interp1d(
            prior_kde.values[0],
            prior_kde.values[1],
            bounds_error=False,
            fill_value=finfo("float").eps,
        )(ref_val).item()

        if prior_val <= 0 or posterior_val <= 0:
            raise ValueError(
                f"Invalid KDE values at ref_val={ref_val}: "
                f"prior={prior_val}, posterior={posterior_val}"
            )

        bf_10 = prior_val / posterior_val
        bf_01 = 1 / bf_10
        results[var] = {"BF10": bf_10, "BF01": bf_01}

        if return_ref_vals:
            ref_density_vals[var] = {"prior": prior_val, "posterior": posterior_val}

    if return_ref_vals:
        return results, ref_density_vals

    return results
