"""Functions for manipulating samples."""

import numpy as np
import xarray as xr
from arviz_base import convert_to_dataset, convert_to_datatree, extract, from_dict, rcParams

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
        effective sample size as suggested by Säilynoja et al. (2022) [1]_.
        If an integer, the thinning factor is set to that value.
    chain_axis, draw_axis : int, optional
        Integer indicators of the axis that correspond to the chain and the draw dimension.
        `chain_axis` can be ``None``.

    Returns
    -------
    ndarray, DataArray, Dataset, DataTree
        Thinned samples

    References
    ----------

    .. [1] Säilynoja, T., Bürkner, PC. & Vehtari, A. "Graphical test for discrete
           uniformity and its applications in goodness-of-fit evaluation and
           multiple sample comparison." Statistics and Computing 32(2), 32 (2022).
           https://doi.org/10.1007/s11222-022-10090-6

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


def weight_predictions(
    dts, weights=None, group="posterior_predictive", sample_dims=None, random_seed=None
):
    """Generate weighted posterior predictive samples.

    Parameters
    ----------
    dts : list[DataTree] or list[InferenceData]
        Elements in the list should contain the groups `posterior_predictive`
        and `observed_data`. Observations should be the same for all models.
    weights : array-like, optional
        Individual weights for each model. Weights should be positive. If they do not sum up to 1,
        they will be normalized. Default, same weight for each model.
        Weights can be computed using many different methods including those in
        :func:`arviz.compare`.
    group : str, optional
        Group from which to extract predictions to weight. Default to `posterior_predictive`.
    sample_dims : iterable of hashable, optional
        Dimensions to be considered sample dimensions and are to be reduced.
        Default ``rcParams["data.sample_dims"]``.
    random_seed : int, optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    dt: DataTree
        DataTree object with the groups `posterior_predictive` and `observed_data`.

    See Also
    --------
    compare :  Compare models based on PSIS-LOO-CV `loo`
    """
    dts = [convert_to_datatree(dt) for dt in dts]
    if len(dts) < 2:
        raise ValueError("You should provide a list with at least two elements")

    if not all(group in dt for dt in dts):
        raise ValueError(f"All the objects must contain the `{group}` group")

    # For InferenceData objects this only checks the values but for DataTree objects
    # this also checks the coords and dims
    # if not all(dts[0].observed_data.equals(dt.observed_data) for dt in dts[1:]):
    #     raise ValueError("The observed data should be the same for all DataTree objects")

    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]

    if weights is None:
        weights = np.ones(len(dts)) / len(dts)
    elif len(dts) != len(weights):
        raise ValueError(
            "The number of weights should be the same as the number of elements in the dts list"
        )

    weights = np.array(weights, dtype=float)
    weights /= weights.sum()
    len_data = np.array([np.prod([dt[group].sizes[dim] for dim in sample_dims]) for dt in dts])

    if not all(len_data):
        raise ValueError("At least one of your DataTree objects has 0 samples")

    new_samples = (np.min(len_data) * weights).astype(int)

    new_idatas = [
        extract(dt, group=group, num_samples=samples, random_seed=random_seed).reset_coords()
        for samples, dt in zip(new_samples, dts)
    ]

    weighted_samples = from_dict(
        {
            group: xr.concat(new_idatas, dim="sample"),
            "observed_data": dts[0].observed_data,
        },
        sample_dims=list(new_idatas[0].dims.keys()),
    )

    return weighted_samples
