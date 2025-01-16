from .base.density import _DensityBase 
import numpy as np 
import warnings

def bayes_factor(idata, var_name, ref_val=0, return_ref_vals=False, prior=None):
    """
    Approximated Bayes Factor for comparing hypotheses of two nested models, 
    using KDE for density estimation.
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
    """
    posterior = extract(idata, var_names=var_name).values
    prior = extract(idata, var_names=var_name, group="prior").values

    if not isinstance(ref_val, (int, float)):
        raise ValueError("The reference value (ref_val) must be a numerical value (int or float).")

    if ref_val > posterior.max() or ref_val < posterior.min():
        warnings.warn(
            "The reference value is outside the posterior range. "
            "This results in infinite support for H1, which may overstate evidence."
        )
    if posterior.dtype.kind == "f":
        density_instance = _DensityBase()
        posterior_grid, posterior_pdf, _ = density_instance._kde(x=posterior, grid_len=512, circular=False)
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
    else:
        return bf
