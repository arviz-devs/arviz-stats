"""Bayes Factor using Savage-Dickey density ratio."""

import warnings

import numpy as np
from arviz_base import extract

from arviz_stats.base.density import _DensityBase


def bayes_factor(idata, var_name, ref_val=0, return_ref_vals=False, prior=None):
    """
    Approximated Bayes Factor for comparing hypothesis of two nested models.

    Parameters
    ----------
    idata : InferenceData
        Object containing posterior and prior data.
    var_name : str
        Name of the variable to test.
    ref_val : int or float, default 0
        Reference (point-null) value for Bayes factor estimation.
    return_ref_vals : bool, default False
        If True, also return the values of prior and posterior densities at the reference value.

    Returns
    -------
    dict
        A dictionary with Bayes Factor values: BF10 (H1/H0 ratio) and BF01 (H0/H1 ratio).

    References
    ----------
    .. [1] Heck, D., 2019. A caveat on the Savage-Dickey density ratio:
       The case of computing Bayes factors for regression parameters.

    Examples
    --------
    Moderate evidence indicating that the parameter "a" is different from zero.

    .. ipython::

        In [1]: import numpy as np
           ...: from arviz_base import from_dict
           ...: import arviz_stats as azs
           ...: idata = from_dict({"posterior":{"a":np.random.normal(1, 0.5, (2, 1000))},
           ...:                    {"prior":{"a":np.random.normal(0, 1, (2, 1000))}}})
           ...: azs.plot_bf(idata, var_name="a", ref_val=0)
    """
    posterior = extract(idata, var_names=var_name).values
    prior = extract(idata, var_names=var_name, group="prior").values

    if not isinstance(ref_val, int | float):
        raise ValueError("The reference value (ref_val) must be a numerical value (int or float).")

    if ref_val > posterior.max() or ref_val < posterior.min():
        warnings.warn(
            "The reference value is outside the posterior range. "
            "This results in infinite support for H1, which may overstate evidence."
        )

    prior_at_ref_val = 0
    posterior_at_ref_val = 0

    if posterior.dtype.kind == "f":
        # pylint: disable=W0212
        density_instance = _DensityBase()
        posterior_grid, posterior_pdf, _ = density_instance._kde(
            x=posterior, grid_len=512, circular=False
        )
        prior_grid, prior_pdf, _ = density_instance._kde(x=prior, grid_len=512, circular=False)

        posterior_at_ref_val = np.interp(ref_val, posterior_grid, posterior_pdf)
        prior_at_ref_val = np.interp(ref_val, prior_grid, prior_pdf)

    elif posterior.dtype.kind == "i":
        posterior_at_ref_val = (posterior == ref_val).mean()
        prior_at_ref_val = (prior == ref_val).mean()

    bf_10 = prior_at_ref_val / posterior_at_ref_val
    bf = {"BF10": bf_10, "BF01": 1 / bf_10}

    if return_ref_vals:
        return (bf, {"prior": prior_at_ref_val, "posterior": posterior_at_ref_val})
    return bf
