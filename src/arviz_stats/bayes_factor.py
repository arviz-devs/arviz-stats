"""Bayes Factor using Savage-Dickey density ratio."""

import warnings


def bayes_factor(idata, var_names, ref_vals=0, return_ref_vals=False, prior=None):
    """
    Compute Bayes factor(s) using Savage–Dickey ratio for one or more variables.

    Parameters
    ----------
    idata : InferenceData
        Object containing posterior and prior data.
    var_names : str or list of str
        Name(s) of the variable(s) to test.
    ref_vals : float or list of float, default 0
        Reference value(s) for each variable. Must match var_names in length if list.
    return_ref_vals : bool, default False
        If True, also return the values of prior and posterior densities at the reference value(s).

    Returns
    -------
    dict
        Dictionary with Bayes Factor values: BF10 and BF01 per variable.

    References
    ----------
    .. [1] Heck, D., 2019. A caveat on the Savage-Dickey density ratio:
       The case of computing Bayes factors for regression parameters.

    Examples
    --------
    Moderate evidence indicating that the parameter "a" is different from zero,
    and the parameter "b" is different from 1.

    .. ipython::

        In [1]: import numpy as np
           ...: from arviz_base import from_dict
           ...: import arviz_stats as azs
           ...: dt = from_dict({
           ...:     "posterior": {
           ...:         "a": np.random.normal(1, 0.5, (2, 1000)),
           ...:         "b": np.random.normal(2, 0.5, (2, 1000))
           ...:     },
           ...:     "prior": {
           ...:         "a": np.random.normal(0, 1, (2, 1000)),
           ...:         "b": np.random.normal(1, 1, (2, 1000))
           ...: }})
           ...: ref_vals = [0, 1]
           ...: result = azs.bayes_factor(dt, var_names=["a", "b"], ref_vals=ref_vals)
           ...: result

    """
    if isinstance(var_names, str):
        var_names = [var_names]

    if isinstance(ref_vals, int | float):
        ref_vals = [ref_vals] * len(var_names)

    if len(var_names) != len(ref_vals):
        raise ValueError("Length of var_names and ref_vals must match.")

    results = {}
    ref_density_vals = {}

    for var, ref_val in zip(var_names, ref_vals):
        posterior = idata.posterior[var]
        prior = idata.prior[var]

        if not isinstance(ref_val, int | float):
            raise ValueError(f"Reference value for variable '{var}' must be numerical")

        if ref_val > posterior.max() or ref_val < posterior.min():
            warnings.warn(
                f"Reference value {ref_val} for '{var}' is outside the posterior range. "
                "This may overstate evidence in favor of H1."
            )

        posterior_kde = posterior.azstats.kde(grid_len=512, circular=False)
        prior_kde = prior.azstats.kde(grid_len=512, circular=False)

        posterior_at_ref = posterior_kde.interp(kde_dim=ref_val)
        prior_at_ref = prior_kde.interp(kde_dim=ref_val)

        prior_val = float(prior_at_ref.sel(plot_axis="y"))
        posterior_val = float(posterior_at_ref.sel(plot_axis="y"))

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
