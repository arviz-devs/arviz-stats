"""Functions for sampling diagnostics."""

import numpy as np
import xarray as xr
from arviz_base import convert_to_dataset

from arviz_stats.utils import get_array_function
from arviz_stats.validate import validate_dims


def ess(
    data,
    sample_dims=None,
    group="posterior",
    var_names=None,
    filter_vars=None,
    coords=None,
    method="bulk",
    relative=False,
    prob=None,
    chain_axis=0,
    draw_axis=1,
):
    r"""Estimate the effective sample size (ess).

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
    method : str, default "bulk"
        Valid methods are:

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
        Return relative ess ``ress = ess / n``
    prob : float, or tuple of two floats, optional
        Probability value for "tail", "quantile" or "local" ess functions.
    chain_axis, draw_axis : int, optional
        Integer indicators of the axis that correspond to the chain and the draw dimension.
        `chain_axis` can be ``None``.

    Returns
    -------
    ndarray, DataArray, Dataset, DataTree
        Requested ESS summary of the provided input

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
    if isinstance(data, list | tuple | np.ndarray):
        data = np.array(data)
        return get_array_function("ess")(
            data,
            method=method,
            relative=relative,
            prob=prob,
            chain_axis=chain_axis,
            draw_axis=draw_axis,
        )

    if isinstance(data, xr.core.groupby.DataArrayGroupBy | xr.core.groupby.DatasetGroupBy):
        # Make sure the grouped dimension is added as one of the dimensions to be reduced
        sample_dims = list(set(validate_dims(sample_dims)).union(data.group1d.dims))
        return data.map(
            ess,
            sample_dims=sample_dims,
            var_names=var_names,
            coords=coords,
            method=method,
            relative=relative,
            prob=prob,
        )

    if isinstance(data, xr.DataArray):
        if coords is not None:
            data = data.sel(coords)
        return data.azstats.ess(dims=sample_dims, method=method, relative=relative, prob=prob)

    if isinstance(data, xr.DataTree):
        data = data.azstats.filter_vars(
            group=group, var_names=var_names, filter_vars=filter_vars
        ).datatree
        if coords is not None:
            data = data.sel(coords)
        return data.azstats.ess(
            dims=sample_dims, group=group, method=method, relative=relative, prob=prob
        )

    data = convert_to_dataset(data, group=group)

    data = data.azstats.filter_vars(var_names=var_names, filter_vars=filter_vars).dataset
    if coords is not None:
        data = data.sel(coords)

    return data.azstats.ess(dims=sample_dims, method=method, relative=relative, prob=prob)


def rhat(
    data,
    sample_dims=None,
    group="posterior",
    var_names=None,
    filter_vars=None,
    coords=None,
    method="rank",
    chain_axis=0,
    draw_axis=1,
):
    r"""Compute estimate of rank normalized split R-hat for a set of traces.

    The rank normalized R-hat diagnostic tests for lack of convergence by comparing the variance
    between multiple chains to the variance within each chain. If convergence has been achieved,
    the between-chain and within-chain variances should be identical. To be most effective in
    detecting evidence for nonconvergence, each chain should have been initialized to starting
    values that are dispersed relative to the target distribution.

    Parameters
    ----------
    data : array-like, DataArray, Dataset, DataTree, DataArrayGroupBy, DatasetGroupBy, or idata-like
        Input data. It will have different pre-processing applied to it depending on its type:

        - array-like: call array layer within ``arviz-stats``.
        - xarray object: apply dimension aware function to all relevant subsets
        - others: passed to :func:`arviz_base.convert_to_dataset`

        At least 2 posterior chains are needed to compute this diagnostic of one or more
        stochastic parameters.

    sample_dims : iterable of hashable, optional
        Dimensions to be considered sample dimensions and are to be reduced.
        Default ``rcParams["data.sample_dims"]``.
    group : hashable, default "posterior"
        Group on which to compute the ESS.
    var_names : str or list of str, optional
        Names of the variables for which the Rhat should be computed.
    filter_vars : {None, "like", "regex"}, default None
    coords : dict, optional
        Dictionary of dimension/index names to coordinate values defining a subset
        of the data for which to perform the computation.
    method : str, default "rank"
        Valid methods are:
        - "rank"        # recommended by Vehtari et al. (2021)
        - "split"
        - "folded"
        - "z_scale"
        - "identity"
    chain_axis, draw_axis : int, optional
        Integer indicators of the axis that correspond to the chain and the draw dimension.
        `chain_axis` can be ``None``.

    Returns
    -------
    ndarray, DataArray, Dataset, DataTree
        Requested Rhat summary of the provided input


    See Also
    --------
    arviz.ess : Calculate estimate of the effective sample size (ess).
    arviz.mcse : Calculate Markov Chain Standard Error statistic.
    plot_forest : Forest plot to compare HDI intervals from a number of distributions.

    Notes
    -----
    The diagnostic is computed by:

      .. math:: \hat{R} = \sqrt{\frac{\hat{V}}{W}}

    where :math:`W` is the within-chain variance and :math:`\hat{V}` is the posterior variance
    estimate for the pooled rank-traces. This is the potential scale reduction factor, which
    converges to unity when each of the traces is a sample from the target posterior. Values
    greater than one indicate that one or more chains have not yet converged.

    Rank values are calculated over all the chains with ``scipy.stats.rankdata``.
    Each chain is split in two and normalized with the z-transform following
    Vehtari et al. (2021).

    References
    ----------
    * Vehtari et al. (2021).  Rank-normalization, folding, and
      localization: An improved Rhat for assessing convergence of
      MCMC. Bayesian analysis, 16(2):667-718.
    * Gelman et al. BDA3 (2013)
    * Brooks and Gelman (1998)
    * Gelman and Rubin (1992)

    Examples
    --------
    Calculate the R-hat using the default arguments:

    .. ipython::

        In [1]: from arviz_base import load_arviz_data
           ...: import arviz_stats as azs
           ...: data = load_arviz_data('non_centered_eight')
           ...: azs.rhat(data)

    Calculate the R-hat of some variables using the folded method:

    .. ipython::

        In [1]: azs.rhat(data, var_names=["mu", "theta_t"], method="folded")

    """
    if isinstance(data, list | tuple | np.ndarray):
        data = np.array(data)
        return get_array_function("rhat")(
            data,
            method=method,
            chain_axis=chain_axis,
            draw_axis=draw_axis,
        )

    if isinstance(data, xr.core.groupby.DataArrayGroupBy | xr.core.groupby.DatasetGroupBy):
        # Make sure the grouped dimension is added as one of the dimensions to be reduced
        sample_dims = list(set(validate_dims(sample_dims)).union(data.group1d.dims))
        return data.map(
            rhat,
            sample_dims=sample_dims,
            var_names=var_names,
            coords=coords,
            method=method,
        )

    if isinstance(data, xr.DataArray):
        if coords is not None:
            data = data.sel(coords)
        return data.azstats.rhat(dims=sample_dims, method=method)

    if isinstance(data, xr.DataTree):
        data = data.azstats.filter_vars(
            group=group, var_names=var_names, filter_vars=filter_vars
        ).datatree
        if coords is not None:
            data = data.sel(coords)
        return data.azstats.rhat(dims=sample_dims, group=group, method=method)

    data = convert_to_dataset(data, group=group)

    data = data.azstats.filter_vars(var_names=var_names, filter_vars=filter_vars).dataset
    if coords is not None:
        data = data.sel(coords)

    return data.azstats.rhat(dims=sample_dims, method=method)


def mcse(
    data,
    sample_dims=None,
    group="posterior",
    var_names=None,
    filter_vars=None,
    coords=None,
    method="mean",
    prob=None,
    chain_axis=0,
    draw_axis=1,
):
    """Calculate Markov Chain Standard Error statistic.

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
        Names of the variables for which the mcse should be computed.
    filter_vars : {None, "like", "regex"}, default None
    coords : dict, optional
        Dictionary of dimension/index names to coordinate values defining a subset
        of the data for which to perform the computation.
    method : str, default "mean"
        Valid methods are:
        - "mean"
        - "sd"
        - "median"
        - "quantile"
    prob : float, or tuple of two floats, optional
        Probability value "quantile".
    chain_axis, draw_axis : int, optional
        Integer indicators of the axis that correspond to the chain and the draw dimension.
        `chain_axis` can be ``None``.

    Returns
    -------
    ndarray, DataArray, Dataset, DataTree
        Requested mcse summary of the provided input

    See Also
    --------
    arviz.ess : Compute autocovariance estimates for every lag for the input array.
    arviz.summary : Create a data frame with summary statistics.
    plot_mcse : Plot quantile or local Monte Carlo Standard Error.

    Examples
    --------
    Calculate the Markov Chain Standard Error using the default arguments:

    .. ipython::

        In [1]: from arviz_base import load_arviz_data
           ...: import arviz_stats as azs
           ...: data = load_arviz_data('non_centered_eight')
           ...: azs.mcse(data)

    Calculate the Markov Chain Standard Error using the quantile method:

    .. ipython::

        In [1]: azs.mcse(data, method="quantile", prob=0.7)

    """
    if isinstance(data, list | tuple | np.ndarray):
        data = np.array(data)
        return get_array_function("mcse")(
            data,
            method=method,
            prob=prob,
            chain_axis=chain_axis,
            draw_axis=draw_axis,
        )

    if isinstance(data, xr.core.groupby.DataArrayGroupBy | xr.core.groupby.DatasetGroupBy):
        # Make sure the grouped dimension is added as one of the dimensions to be reduced
        sample_dims = list(set(validate_dims(sample_dims)).union(data.group1d.dims))
        return data.map(
            mcse,
            sample_dims=sample_dims,
            var_names=var_names,
            coords=coords,
            method=method,
            prob=prob,
        )

    if isinstance(data, xr.DataArray):
        if coords is not None:
            data = data.sel(coords)
        return data.azstats.mcse(dims=sample_dims, method=method, prob=prob)

    if isinstance(data, xr.DataTree):
        data = data.azstats.filter_vars(
            group=group, var_names=var_names, filter_vars=filter_vars
        ).datatree
        if coords is not None:
            data = data.sel(coords)
        return data.azstats.mcse(dims=sample_dims, group=group, method=method, prob=prob)

    data = convert_to_dataset(data, group=group)

    data = data.azstats.filter_vars(var_names=var_names, filter_vars=filter_vars).dataset
    if coords is not None:
        data = data.sel(coords)

    return data.azstats.mcse(dims=sample_dims, method=method, prob=prob)
