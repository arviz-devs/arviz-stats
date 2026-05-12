"""Top level functions related to visualization of distributions."""

import numpy as np
import xarray as xr
from arviz_base import convert_to_dataset

from arviz_stats.utils import _apply_multi_input_function, get_function
from arviz_stats.validate import validate_ci_prob, validate_dims


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

          It is recommended to first perform the conversion manually and then call
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

        In [1]: azs.hdi(dt, dim=["chain","draw", "school"])
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

          It is recommended to first perform the conversion manually and then call
          ``arviz_stats.eti``. This allows controlling the conversion step and inspecting
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
        Requested ETI of the provided input. It will have a ``ci_bound`` dimension
        with coordinate values "lower" and "upper" indicating the two extremes of
        the credible interval.

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
           ...: azs.eti(data, 0.68)

    Calculate the ETI for specific variables:

    .. ipython::

        In [1]: import arviz_base as azb
           ...: dt = azb.load_arviz_data("centered_eight")
           ...: azs.eti(dt, var_names=["mu", "theta"])

    Calculate the ETI also over the school dimension (for variables where present):

    .. ipython::

        In [1]: azs.eti(dt, dim=["chain","draw", "school"])
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
    r"""Compute the marginal empirical cumulative density functions (ECDF).

    See the EABM chapter on `Visualization of Random Variables with ArviZ <https://arviz-devs.github.io/EABM/Chapters/Distributions.html#distributions-in-arviz>`_
    for more details.

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

          It is recommended to first perform the conversion manually and then call
          ``arviz_stats.ecdf``. This allows controlling the conversion step and inspecting
          its results.
    dim : sequence of hashable, optional
        Dimensions to be reduced when computing the ECDF. Default ``rcParams["data.sample_dims"]``.
    group : hashable, default "posterior"
        Group on which to compute the ECDF
    var_names : str or list of str, optional
        Names of the variables for which the ECDF should be computed.
    filter_vars : {None, "like", "regex"}, default None
    coords : dict, optional
        Dictionary of dimension/index names to coordinate values defining a subset
        of the data for which to perform the computation.
    pit : bool, default False
    **kwargs : any, optional
        Forwarded to the array or dataarray interface for ECDF.

    Returns
    -------
    ndarray, DataArray, Dataset, DataTree
        Requested ECDF of the provided input.
        It will have a ``quantile`` dimension and a ``plot_axis`` dimension with coordinate
        values "x" and "y".

    See Also
    --------
    arviz_stats.histogram, arviz_stats.kde, arviz_stats.qds :
        Alternative visual summaries for marginal distributions
    arviz_plots.plot_dist

    Examples
    --------
    Calculate the ECDF of a Normal random variable:

    .. ipython::

        In [1]: import arviz_stats as azs
           ...: import numpy as np
           ...: data = np.random.default_rng().normal(size=2000)
           ...: # not available yet in array interface azs.ecdf(data)

    Calculate the ECDF for specific variables:

    .. ipython::

        In [1]: import arviz_base as azb
           ...: dt = azb.load_arviz_data("centered_eight")
           ...: azs.ecdf(dt.posterior.dataset, var_names=["mu", "theta"])

    Calculate the ECDF also over the school dimension (for variables where present):

    .. ipython::

        In [1]: azs.ecdf(dt.posterior.dataset, dim=["chain", "draw", "school"])
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
    r"""Compute the batched histogram.

    See the EABM chapter on `Visualization of Random Variables with ArviZ <https://arviz-devs.github.io/EABM/Chapters/Distributions.html#distributions-in-arviz>`_
    for more details.

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

          It is recommended to first perform the conversion manually and then call
          ``arviz_stats.histogram``. This allows controlling the conversion step and inspecting
          its results.
    dim : sequence of hashable, optional
        Dimensions to be reduced when computing the histogram.
        Default ``rcParams["data.sample_dims"]``.
    group : hashable, default "posterior"
        Group on which to compute the histogram
    var_names : str or list of str, optional
        Names of the variables for which the histogram should be computed.
    filter_vars : {None, "like", "regex"}, default None
    coords : dict, optional
        Dictionary of dimension/index names to coordinate values defining a subset
        of the data for which to perform the computation.
    bind : array-like, optional
    range : array-like, optional
    weights : array-like, optional
    density : bool, default True
    **kwargs : any, optional
        Forwarded to the array or dataarray interface for histogram.

    Returns
    -------
    ndarray, DataArray, Dataset, DataTree
        Requested histogram of the provided input.
        It will have a ``hist_dim_{var_name}`` dimension and a ``plot_axis`` dimension
        with coordinates "histogram", "left_edges" and "right_edges"

    See Also
    --------
    arviz_stats.ecdf, arviz_stats.kde, arviz_stats.qds :
        Alternative visual summaries for marginal distributions
    arviz_plots.plot_dist

    Examples
    --------
    Calculate the histogram of a Normal random variable:

    .. ipython::

        In [1]: import arviz_stats as azs
           ...: import numpy as np
           ...: data = np.random.default_rng().normal(size=2000)
           ...: azs.histogram(data)

    Calculate the histogram for specific variables:

    .. ipython::

        In [1]: import arviz_base as azb
           ...: dt = azb.load_arviz_data("centered_eight")
           ...: azs.histogram(dt, var_names=["mu", "theta"])

    Calculate the histogram also over the school dimension (for variables where present):

    .. ipython::

        In [1]: azs.histogram(dt, dim=["chain", "draw", "school"])
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
    r"""Compute the marginal kernel density estimates (KDE).

    See the EABM chapter on `Visualization of Random Variables with ArviZ <https://arviz-devs.github.io/EABM/Chapters/Distributions.html#distributions-in-arviz>`_
    for more details.

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

          It is recommended to first perform the conversion manually and then call
          ``arviz_stats.kde``. This allows controlling the conversion step and inspecting
          its results.
    dim : sequence of hashable, optional
        Dimensions to be reduced when computing the KDE.
        Default ``rcParams["data.sample_dims"]``.
    group : hashable, default "posterior"
        Group on which to compute the KDE
    var_names : str or list of str, optional
        Names of the variables for which the KDE should be computed.
    filter_vars : {None, "like", "regex"}, default None
    coords : dict, optional
        Dictionary of dimension/index names to coordinate values defining a subset
        of the data for which to perform the computation.
    circular : bool, default False
    **kwargs : any, optional
            Forwarded to the array or dataarray interface for KDE.
            See :func:`arviz_stats.base.array_stats.kde` for the full list of supported arguments.

    Returns
    -------
    ndarray, DataArray, Dataset, DataTree
        Requested KDE of the provided input.
        The xarray objects will have a ``kde_dim`` dimension and a ``plot_axis`` dimension
        with coordinates "x", and "y".

    See Also
    --------
    arviz_stats.ecdf, arviz_stats.histogram, arviz_stats.qds:
        Alternative visual summaries for marginal distributions
    arviz_plots.plot_dist

    Examples
    --------
    Calculate the KDE of a Normal random variable:

    .. ipython::

        In [1]: import arviz_stats as azs
           ...: import numpy as np
           ...: data = np.random.default_rng().normal(size=2000)
           ...: azs.kde(data)

    Calculate the KDE for specific variables:

    .. ipython::

        In [1]: import arviz_base as azb
           ...: dt = azb.load_arviz_data("centered_eight")
           ...: azs.kde(dt, var_names=["mu", "theta"])

    Calculate the KDE also over the school dimension (for variables where present):

    .. ipython::

        In [1]: azs.kde(dt, dim=["chain", "draw", "school"])
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


def qds(
    data,
    dim=None,
    group="posterior",
    var_names=None,
    filter_vars=None,
    coords=None,
    nquantiles=100,
    binwidth=None,
    dotsize=1,
    stackratio=1,
    top_only=False,
    **kwargs,
):
    r"""Compute the marginal quantile dots.

    For details see [1]_ and check the EABM chapter on `Visualization of Random Variables with
    ArviZ <https://arviz-devs.github.io/EABM/Chapters/Distributions.html#distributions-in-arviz>`_.

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

          It is recommended to first perform the conversion manually and then call
          ``arviz_stats.kde``. This allows controlling the conversion step and inspecting
          its results.
    dim : sequence of hashable, optional
        Dimensions to be reduced when computing the quantile dots
        Default ``rcParams["data.sample_dims"]``.
    group : hashable, default "posterior"
        Group on which to compute the quantile dots
    var_names : str or list of str, optional
        Names of the variables for which the quantile dots should be computed.
    filter_vars : {None, "like", "regex"}, default None
    coords : dict, optional
        Dictionary of dimension/index names to coordinate values defining a subset
        of the data for which to perform the computation.
    nquantiles : int, default 100
        Number of quantiles (i.e., dots) to compute.
    binwidth : float, optional
        Width of the bin for the dots.
    dotsize : float, default 1
        The size of the dots relative to the bin width. The default makes dots be just about as
        wide as the bin width.
    stackratio : float, default 1
        The distance between the center of the dots in the same stack relative to the bin height.
        The default makes dots in the same stack just touch each other.
    top_only : bool, default False
        If true, only the top dots of each stack are returned.
    **kwargs : any, optional
        Forwarded to the array or dataarray interface for quantile dots.

    Returns
    -------
    ndarray, DataArray, Dataset, DataTree
        Requested QDs of the provided input.
        The xarray objects will have a ``qds_dim`` dimension and a ``plot_axis`` dimension
        with coordinates "x", and "y".

    See Also
    --------
    arviz_stats.ecdf, arviz_stats.histogram, arviz_stats.kde:
        Alternative visual summaries for marginal distributions
    arviz_plots.plot_dist

    References
    ----------
    .. [1] Kay M, Kola T, Hullman JR, and Munson SA. *When (ish) is My Bus?:
       User-centered Visualizations of Uncertainty in Everyday, Mobile Predictive
       Systems.* In Proceedings of the 2016 CHI Conference Association for Computing
       Machinery. 2016. https://doi.org/10.1145/2858036.2858558

    Examples
    --------
    Calculate the QDs of a Normal random variable:

    .. ipython::

        In [1]: import arviz_stats as azs
           ...: import numpy as np
           ...: data = np.random.default_rng().normal(size=2000)
           ...: azs.qds(data)

    Calculate the QDs for specific variables:

    .. ipython::

        In [1]: import arviz_base as azb
           ...: dt = azb.load_arviz_data("centered_eight")
           ...: azs.qds(dt, var_names=["mu", "theta"])

    Calculate the QDs also over the school dimension (for variables where present):

    .. ipython::

        In [1]: azs.qds(dt, dim=["chain", "draw", "school"])
    """
    return _apply_multi_input_function(
        "qds",
        data,
        dim,
        "dim",
        group=group,
        var_names=var_names,
        filter_vars=filter_vars,
        coords=coords,
        nquantiles=nquantiles,
        binwidth=binwidth,
        dotsize=dotsize,
        stackratio=stackratio,
        top_only=top_only,
        **kwargs,
    )


def kde2d(
    da_x,
    da_y,
    dim=None,
    group="posterior",
    gridsize=(128, 128),
    circular=False,
    hdi_probs=None,
):
    r"""Compute a 2D kernel density estimate (KDE) for two variables.

    Parameters
    ----------
    da_x : array-like or DataArray or DataTree or idata-like
        Samples for the first variable (x axis).

        - array-like: call the array layer within ``arviz-stats`` directly.
        - :class:`xarray.DataArray`: apply the dimension-aware function.
        - :class:`xarray.DataTree` / InferenceData-like: the group indicated by
          *group* is used; the object must contain a single variable, or you
          should extract the DataArray beforehand.
    da_y : array-like or DataArray or DataTree or idata-like
        Samples for the second variable (y axis).  Must be the same type as
        da_x and contain the same number of samples along the sample dimensions.
    dim : str or sequence of hashable, optional
        Dimensions to reduce over (the sample dimensions).
        Defaults to ``rcParams["data.sample_dims"]``.
    group : hashable, default "posterior"
        Group to use when da_x / da_y are DataTree or InferenceData objects.
    gridsize : tuple of int, default (128, 128)
        Number of grid points along each axis.  Powers of 2 are recommended for
        FFT efficiency.
    circular : bool, default False
        Whether to use circular (wrap) boundary conditions.
    hdi_probs : list of float, optional
        If given, compute the HDI contour levels for the specified probabilities and
        return them together with the KDE grid and coordinates.

    Returns
    -------
    tuple or xarray.Dataset
        When da_x and da_y are plain arrays the raw outputs of the array
        layer are returned: a tuple ``(grid, x_coords, y_coords)`` or
        ``(grid, x_coords, y_coords, contours)`` if hdi_probs is given.

        For xarray inputs an :class:`xarray.Dataset` is returned with variables:

        * ``density`` - 2D KDE on a regular grid, dims ``(*batch_dims, kde2d_x, kde2d_y)``.
        * ``x_coords`` - Grid coordinates for the x axis, dims ``(*batch_dims, kde2d_x)``.
        * ``y_coords`` - Grid coordinates for the y axis, dims ``(*batch_dims, kde2d_y)``.
        * ``contours`` - (only when hdi_probs is given) Contour levels,
          dims ``(*batch_dims, hdi_prob)`` with the requested probabilities as
          coordinate values.

    See Also
    --------
    arviz_stats.kde : 1D marginal KDE.

    Examples
    --------
    Compute the 2D KDE of two correlated normal variables:

    .. ipython::

        In [1]: import arviz_stats as azs
           ...: import numpy as np
           ...: rng = np.random.default_rng(0)
           ...: x = rng.normal(size=2000)
           ...: y = 0.8 * x + rng.normal(scale=0.6, size=2000)
           ...: azs.kde2d(x, y)

    Include HDI contour levels:

    .. ipython::

        In [1]: azs.kde2d(x, y, hdi_probs=[0.5, 0.9])

    Compute on posterior DataArrays:

    .. ipython::

        In [1]: import arviz_base as azb
           ...: dt = azb.load_arviz_data("centered_eight")
           ...: azs.kde2d(
           ...:     dt.posterior["mu"],
           ...:     dt.posterior["tau"],
           ...: )
    """
    if isinstance(da_x, np.ndarray | list | tuple) and isinstance(da_y, np.ndarray | list | tuple):
        from arviz_stats.utils import get_array_function

        x = np.asarray(da_x)
        y = np.asarray(da_y)
        return get_array_function("kde2d")(
            x, y, gridsize=gridsize, circular=circular, hdi_probs=hdi_probs
        )

    dims = validate_dims(dim)
    if not isinstance(da_x, xr.DataArray):
        ds_x = convert_to_dataset(da_x, group=group)
        var_names_x = list(ds_x.data_vars)
        if len(var_names_x) != 1:
            raise ValueError(
                "When da_x is not a DataArray it must contain exactly one variable. "
                f"Found: {var_names_x}"
            )
        da_x = ds_x[var_names_x[0]]

    if not isinstance(da_y, xr.DataArray):
        ds_y = convert_to_dataset(da_y, group=group)
        var_names_y = list(ds_y.data_vars)
        if len(var_names_y) != 1:
            raise ValueError(
                "When da_y is not a DataArray it must contain exactly one variable. "
                f"Found: {var_names_y}"
            )
        da_y = ds_y[var_names_y[0]]

    return get_function("kde2d")(
        da_x,
        da_y,
        dim=dims,
        gridsize=gridsize,
        circular=circular,
        hdi_probs=hdi_probs,
    )
