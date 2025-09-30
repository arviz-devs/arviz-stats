"""Functions for manipulating samples."""

import numpy as np
import xarray as xr
from arviz_base import convert_to_dataset

from arviz_stats.utils import get_array_function
from arviz_stats.validate import validate_dims


def thin(
    data,
    sample_dims="draw",
    group="posterior",
    var_names=None,
    filter_vars=None,
    coords=None,
    factor="auto",
    chain_axis=0,
    draw_axis=1,
):
    r"""Perform thinning.

    Thinning refers to retaining only every nth sample from a Markov Chain Monte Carlo (MCMC)
    simulation. This is usually done to reduce autocorrelation in the stored samples or simply
    to reduce the size of the stored samples.

    Parameters
    ----------
    data : array-like, DataArray, Dataset, DataTree, DataArrayGroupBy, DatasetGroupBy, or idata-like
        Input data. It will have different pre-processing applied to it depending on its type:

        - array-like: call array layer within ``arviz-stats``.
        - xarray object: apply dimension aware function to all relevant subsets
        - others: passed to :func:`arviz_base.convert_to_dataset`

    sample_dims : iterable of hashable, optional
        Dimensions to be considered sample dimensions and are to be reduced.
        Default ``rcParams["data.sample_dims"]``.
    group : hashable, default "posterior"
        Group on which to compute the ESS.
    var_names : str or list of str, optional
        Names of the variables for which the ess should be computed.
    filter_vars : {None, "like", "regex"}, default None
    coords : dict, optional
        Dictionary of dimension/index names to coordinate values defining a subset
        of the data for which to perform the computation.
    factor : str or int, default "auto"
        The thinning factor. If "auto", the thinning factor is computed based on bulk and tail
        effective sample size as suggested by Säilynoja et al. (2022).
        If an integer, the thinning factor is set to that value.
    chain_axis, draw_axis : int, optional
        Integer indicators of the axis that correspond to the chain and the draw dimension.
        `chain_axis` can be ``None``.

    Returns
    -------
    ndarray, DataArray, Dataset, DataTree
        Thinned samples


    Examples
    --------
    Thin the posterior samples using the default arguments:

    .. ipython::

        In [1]: from arviz_base import load_arviz_data
           ...: import arviz_stats as azs
           ...: data = load_arviz_data('non_centered_eight')
           ...: azs.thin(data)

    Thin a subset of the variables with a thinning factor of 10:

    .. ipython::

        In [1]: azs.thin(data, factor=10, var_names=["mu"])
    """
    if isinstance(data, list | tuple | np.ndarray):
        data = np.array(data)
        return get_array_function("thin")(
            data,
            factor=factor,
            chain_axis=chain_axis,
            draw_axis=draw_axis,
        )

    if isinstance(data, xr.core.groupby.DataArrayGroupBy | xr.core.groupby.DatasetGroupBy):
        # Make sure the grouped dimension is added as one of the dimensions to be reduced
        sample_dims = list(set(validate_dims(sample_dims)).union(data.group1d.dims))
        return data.map(
            thin,
            sample_dims=sample_dims,
            var_names=var_names,
            coords=coords,
            factor=factor,
        )

    if isinstance(data, xr.DataArray):
        if coords is not None:
            data = data.sel(coords)
        return data.azstats.thin(sample_dims=sample_dims, factor=factor)

    if isinstance(data, xr.DataTree):
        data = data.azstats.filter_vars(
            group=group, var_names=var_names, filter_vars=filter_vars
        ).datatree
        if coords is not None:
            data = data.sel(coords)
        return data.azstats.thin(sample_dims=sample_dims, group=group, factor=factor)

    data = convert_to_dataset(data, group=group)

    data = data.azstats.filter_vars(var_names=var_names, filter_vars=filter_vars).dataset
    if coords is not None:
        data = data.sel(coords)

    return data.azstats.thin(sample_dims=sample_dims, factor=factor)
