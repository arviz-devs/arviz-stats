"""Functions for sampling diagnostics."""

from arviz_base import convert_to_dataset
from numpy import atleast_2d, ndarray


def ess(data, var_names=None, method="bulk", relative=False, prob=None):
    r"""Estimate the effective sample size (ess).

    Parameters
    ----------
    data : obj
        Any object that can be converted to a :class:`xarray.DataSet` object.
        Refer to documentation of :func:`arviz.convert_to_dataset` for details.
        For ndarray: shape = (chain, draw).
    var_names : str or list of str
        Names of variables to include in the return value Dataset.
    method : str, optional, default "bulk"
        Select ess method. Valid methods are:

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
    xarray.Dataset
        Return the effective sample size
    """
    # maybe have something like this inside convert_to_dataset
    if isinstance(data, (list, tuple, ndarray)):
        data = atleast_2d(data)

    data = convert_to_dataset(data)

    if var_names is not None:
        data = data.filter_vars(var_names=var_names)

    return data.azstats.ess(method=method, relative=relative, prob=prob)
