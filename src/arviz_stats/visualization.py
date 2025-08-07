"""Top level functions related to visualization of distributions."""

from arviz_stats.utils import _apply_multi_input_function
from arviz_stats.validate import validate_ci_prob


def hdi(
    data,
    prob=None,
    dim=None,
    group="posterior",
    var_names=None,
    filter_vars=None,
    coords=None,
    method="nearest",
    circular=False,
    max_modes=10,
    skipna=False,
    **kwargs,
):
    r"""Compute the highest density interval (HDI) given a probability.

    The HDI is the shortest interval that contains the specified probability mass.

    Parameters
    ----------
    data : array-like, DataArray, Dataset, DataTree, DataArrayGroupBy, DatasetGroupBy, or idata-like
        Input data. It will have different pre-processing applied to it depending on its type:

        - array-like: call array layer within ``arviz-stats``.
        - xarray object: apply dimension aware function to all relevant subsets
        - others: passed to :func:`arviz_base.convert_to_dataset` then treated as
          :class:`xarray.Dataset`. This option is discouraged due to needing this conversion
          which is completely automated and will be needed again in future executions or
          similar functions.

          It is recommeded to first perform the conversion manually and then call
          ``arviz_stats.hdi``. This allows controlling the conversion step and inspecting
          its results.
    prob : float, optional
        Probability for the credible interval. Defaults to ``rcParams["stats.ci_prob"]``
    dim : sequence of hashable, optional
        Dimensions to be reduced when computing the HDI. Default ``rcParams["data.sample_dims"]``.
    group : hashable, default "posterior"
        Group on which to compute the HDI.
    var_names : str or list of str, optional
        Names of the variables for which the HDI should be computed.
    filter_vars : {None, "like", "regex"}, default None
    coords : dict, optional
        Dictionary of dimension/index names to coordinate values defining a subset
        of the data for which to perform the computation.
    method : str, default "nearest"
        Valid options are "nearest", "multimodal" or "multimodal_sample"
    circular : bool, default False
        Whether to compute the HDI taking into account that `data` represent circular variables
        (in the range [-np.pi, np.pi]) or not. Defaults to False (i.e non-circular variables).
    max_modes : int, default 10
        Maximum number of modes to consider when computing the HDI using the multimodal method.
    skipna : bool, default False
        If true ignores nan values when computing the HDI.
    **kwargs : any, optional
        Forwarded to the array or dataarray interface for HDI.

    Returns
    -------
    ndarray, DataArray, Dataset, DataTree
        Requested HDI of the provided input. It will have a ``ci_bound`` dimension
        with coordinate values "lower" and "upper" indicating the two extremes of
        the credible interval. In addition when using a multimodal `method` a
        ``mode`` dimension is also added.

    See Also
    --------
    arviz_stats.eti : Calculate the equal tail interval (ETI).
    arviz_stats.summary : Calculate summary statistics and diagnostics.

    Examples
    --------
    Calculate the HDI of a Normal random variable:

    .. ipython::

        In [1]: import arviz_stats as azs
           ...: import numpy as np
           ...: data = np.random.default_rng().normal(size=2000)
           ...: azs.hdi(data, 0.68)

    Calculate the HDI for specific variables:

    .. ipython::

        In [1]: import arviz_base as azb
           ...: dt = azb.load_arviz_data("centered_eight")
           ...: azs.hdi(dt, var_names=["mu", "theta"])

    Calculate the HDI also over the school dimension (for variables where present):

    .. ipython::

        In [1]: azs.hdi(data, dim=["chain","draw", "school"])
    """
    prob = validate_ci_prob(prob)
    return _apply_multi_input_function(
        "hdi",
        data,
        dim,
        "dim",
        group=group,
        var_names=var_names,
        filter_vars=filter_vars,
        coords=coords,
        prob=prob,
        method=method,
        circular=circular,
        max_modes=max_modes,
        skipna=skipna,
        **kwargs,
    )


def eti(
    data,
    prob=None,
    dim=None,
    group="posterior",
    var_names=None,
    filter_vars=None,
    coords=None,
    method="linear",
    skipna=False,
    **kwargs,
):
    r"""Compute the  equal tail interval (ETI) given a probability.

    The ETI is constructed by dividing the remaining probability (e.g., 6% for a 94% interval)
    equally between the two tails of a distribution. Other names for ETI are central interval and
    quantile-based interval.

    Parameters
    ----------
    data : array-like, DataArray, Dataset, DataTree, DataArrayGroupBy, DatasetGroupBy, or idata-like
        Input data. It will have different pre-processing applied to it depending on its type:
        - array-like: call array layer within ``arviz-stats``.
        - xarray object: apply dimension aware function to all relevant subsets
        - others: passed to :func:`arviz_base.convert_to_dataset` then treated as
          :class:`xarray.Dataset`. This option is discouraged due to needing this conversion
          which is completely automated and will be needed again in future executions or
          similar functions.

          It is recommeded to first perform the conversion manually and then call
          ``arviz_stats.hdi``. This allows controlling the conversion step and inspecting
          its results.
    prob : float, optional
        Probability for the credible interval. Defaults to ``rcParams["stats.ci_prob"]``
    dim : sequence of hashable, optional
        Dimensions to be reduced when computing the HDI. Default ``rcParams["data.sample_dims"]``.
    group : hashable, default "posterior"
        Group on which to compute the ETI.
    var_names : str or list of str, optional
        Names of the variables for which the ETI should be computed.
    filter_vars : {None, "like", "regex"}, default None
    coords : dict, optional
        Dictionary of dimension/index names to coordinate values defining a subset
        of the data for which to perform the computation.
    method : str, default "linear"
        For other options see :func:`numpy.quantile`.
    skipna : bool, default False
        If true ignores nan values when computing the ETI.
    **kwargs : any, optional
        Forwarded to the array or dataarray interface for ETI.

    Returns
    -------
    ndarray, DataArray, Dataset, DataTree
        Requested ETI of the provided input

    See Also
    --------
    arviz_stats.hdi : Calculate the highest density interval (HDI).
    arviz_stats.summary : Calculate summary statistics and diagnostics.

    Examples
    --------
    Calculate the ETI of a Normal random variable:

    .. ipython::

        In [1]: import arviz_stats as azs
           ...: import numpy as np
           ...: data = np.random.default_rng().normal(size=2000)
           ...: azs.eti(data, hdi_prob=.68)

    Calculate the ETI for specific variables:

    .. ipython::

        In [1]: import arviz_base as azb
           ...: dt = azb.load_arviz_data("centered_eight")
           ...: azs.eti(dt, var_names=["mu", "theta"])

    Calculate the ETI also over the school dimension (for variables where present):

    .. ipython::

        In [1]: azs.hdi(data, dim=["chain","draw", "school"])
    """
    prob = validate_ci_prob(prob)
    return _apply_multi_input_function(
        "eti",
        data,
        dim,
        "dim",
        group=group,
        var_names=var_names,
        filter_vars=filter_vars,
        coords=coords,
        prob=prob,
        method=method,
        skipna=skipna,
        **kwargs,
    )


def ecdf(
    data,
    dim=None,
    group="posterior",
    var_names=None,
    filter_vars=None,
    coords=None,
    pit=False,
    **kwargs,
):
    """Compute ECDF.

    TODO: add full detailed docstring that can be referenced from accessors and da/array interfaces.
    """
    return _apply_multi_input_function(
        "ecdf",
        data,
        dim,
        "dim",
        group=group,
        var_names=var_names,
        filter_vars=filter_vars,
        coords=coords,
        pit=pit,
        **kwargs,
    )


def histogram(
    data,
    dim=None,
    group="posterior",
    var_names=None,
    filter_vars=None,
    coords=None,
    bins=None,
    range=None,  # pylint: disable=redefined-builtin
    weights=None,
    density=True,
):
    """Compute histogram.

    TODO: add full detailed docstring that can be referenced from accessors and da/array interfaces.
    """
    return _apply_multi_input_function(
        "histogram",
        data,
        dim,
        "dim",
        group=group,
        var_names=var_names,
        filter_vars=filter_vars,
        coords=coords,
        bins=bins,
        range=range,
        weights=weights,
        density=density,
    )


def kde(
    data,
    dim=None,
    group="posterior",
    var_names=None,
    filter_vars=None,
    coords=None,
    circular=False,
    **kwargs,
):
    """Compute KDE.

    TODO: add full detailed docstring that can be referenced from accessors and da/array interfaces.
    """
    return _apply_multi_input_function(
        "kde",
        data,
        dim,
        "dim",
        group=group,
        var_names=var_names,
        filter_vars=filter_vars,
        coords=coords,
        circular=circular,
        **kwargs,
    )
