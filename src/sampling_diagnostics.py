"""Functions for sampling diagnostics."""

from arviz_base import convert_to_dataset
from numpy import atleast_2d, ndarray

from arviz_stats.base.array import BaseArray

base_array = BaseArray()


def ess(data, var_names=None, method="bulk", relative=False, prob=None):
    r"""Estimate the effective sample size (ess).

    Parameters
    ----------
    data : obj
        Array-like or any object that can be converted to a :class:`xarray.DataSet` object.
        Refer to documentation of :func:`arviz.convert_to_dataset` for details.
        For ndarray: shape = (chain, draw).
    var_names : str or list of str
        Names of the variables for which the ess should be computed.
    method : str, optional
        Defaults to "bulk". Valid methods are:

        - "bulk"
        - "tail"     # prob, optional
        - "quantile" # prob
        - "mean" (old ess)
        - "sd"
        - "median"
        - "mad" (mean absolute deviance)
        - "z_scale"
        - "folded"
        - "identity"
        - "local"  # prob
    relative : bool
        Return relative ess
        ``ress = ess / n``
    prob : float, or tuple of two floats, optional
        probability value for "tail", "quantile" or "local" ess functions.

    Returns
    -------
    xarray.Dataset or numpy array with the effective sample size values

    See Also
    --------
    arviz.rhat : Compute estimate of rank normalized split R-hat for a set of traces.
    arviz.mcse : Calculate Markov Chain Standard Error statistic.
    plot_ess : Plot quantile, local or evolution of effective sample sizes (ESS).
    arviz.summary : Create a data frame with summary statistics.

    Examples
    --------
    Calculate the effective_sample_size using the default arguments:

    .. ipython::

        In [1]: from arviz_base import load_arviz_data
           ...: import arviz_stats as azs
           ...: data = load_arviz_data('non_centered_eight')
           ...: azs.ess(data)

    Calculate ess for a subset of the variables

    .. ipython::

        In [1]: azs.ess(data, relative=True, var_names=["mu", "theta_t"])

    Calculate ess using the "tail" method, leaving the `prob` at its default value.

    .. ipython::

        In [1]: azs.ess(data, method="tail")
    """
    if isinstance(data, (list, tuple, ndarray)):
        data = atleast_2d(data)
        return base_array.ess(data, method=method, relative=relative, prob=prob)

    data = convert_to_dataset(data)

    if var_names is not None:
        return data.azstats.filter_vars(var_names=var_names).ess(method=method, relative=relative, prob=prob)

    return data.azstats.ess(method=method, relative=relative, prob=prob)