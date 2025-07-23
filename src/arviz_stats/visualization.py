"""Top level functions related to visualization of distributions."""
import warnings

import numpy as np
import xarray as xr

from arviz_base import convert_to_dataset
from arviz_stats.utils import get_array_function
from arviz_stats.validate import validate_dims, validate_ci_prob

def _warn_non_unique_coords(xr_obj, dims_to_reduce):
    non_unique_coords = [
        dim for dim in xr_obj.dims
        if len(np.unique(xr_obj.coords[dim])) != xr_obj.sizes[dim]
        and dim not in dims_to_reduce
    ]
    if non_unique_coords:
        warnings.warn(
            f"Dimensions {non_unique_coords} don't have unique coordinates which might result "
            "in unexpected results. Either use 'groupby' beforehand or define unique coordiates "
            "to remove this warning"
        )

def _apply_multi_input_function(
    name,
    data,
    dims,
    dims_arg,
    *,
    group,
    var_names,
    filter_vars,
    coords,
    **kwargs,
):
    all_kwargs = {dims_arg: dims} | kwargs
    if isinstance(data, list | tuple | np.ndarray):
        data = np.array(data)
        return get_array_function(name)(data, **kwargs)

    dims = validate_dims(dims)
    if isinstance(data, xr.core.groupby.DataArrayGroupBy | xr.core.groupby.DatasetGroupBy):
        # Make sure the grouped dimension is added as one of the dimensions to be reduced
        dims = list(set(dims).union(data.group1d.dims))

        if isinstance(data, xr.core.groupby.DataArrayGroupBy):
            def reduce_func(xr_obj):
                if coords is not None:
                    xr_obj = xr_obj.sel(coords)
                return getattr(xr_obj.azstats, name)(**{dims_arg: dims}, **kwargs)
        else:
            def reduce_func(xr_obj):
                xr_obj = xr_obj.azstats.filter_vars(var_names=var_names, filter_vars=filter_vars).dataset
                if coords is not None:
                    xr_obj = xr_obj.sel(coords)
                return getattr(xr_obj.azstats, name)(**{dims_arg: dims}, **kwargs)

        return data.map(reduce_func)


    if isinstance(data, xr.DataArray):
        if coords is not None:
            data = data.sel(coords)
        _warn_non_unique_coords(data, dims)
        return getattr(data.azstats, name)(**all_kwargs)

    if isinstance(data, xr.DataTree):
        data = data.azstats.filter_vars(
            group=group, var_names=var_names, filter_vars=filter_vars
        ).datatree
        if coords is not None:
            data = data.sel(coords)
        _warn_non_unique_coords(data, dims)
        return getattr(data.azstats, name)(**all_kwargs)

    data = convert_to_dataset(data, group=group)

    data = data.azstats.filter_vars(var_names=var_names, filter_vars=filter_vars).dataset
    if coords is not None:
        data = data.sel(coords)

    _warn_non_unique_coords(data, dims)
    return getattr(data.azstats, name)(**all_kwargs)

def hdi(data, prob=None, dim=None, group="posterior", var_names=None, filter_vars=None, coords=None, method="nearest", circular=False, max_modes=10, skipna=False, **kwargs):
    """HDI

    TODO: add full detailed docstring that can be referenced from accessors and da/array interfaces."""
    prob = validate_ci_prob(prob)
    return _apply_multi_input_function("hdi", data, dim, "dim", group=group, var_names=var_names, filter_vars=filter_vars, coords=coords, prob=prob, method=method, circular=circular, max_modes=max_modes, skipna=skipna, **kwargs)

def eti(data, prob=None, dim=None, group="posterior", var_names=None, filter_vars=None, coords=None, method="linear", skipna=False, **kwargs):
    """ETI

    TODO: add full detailed docstring that can be referenced from accessors and da/array interfaces."""
    prob = validate_ci_prob(prob)
    return _apply_multi_input_function("eti", data, dim, "dim", group=group, var_names=var_names, filter_vars=filter_vars, coords=coords, prob=prob, method=method, skipna=skipna, **kwargs)

def ecdf(data, dim=None, group="posterior", var_names=None, filter_vars=None, coords=None, pit=False, **kwargs):
    """ECDF

    TODO: add full detailed docstring that can be referenced from accessors and da/array interfaces."""
    return _apply_multi_input_function("ecdf", data, dim, "dim", group=group, var_names=var_names, filter_vars=filter_vars, coords=coords, pit=pit, **kwargs)

def histogram(data, dim=None, group="posterior", var_names=None, filter_vars=None, coords=None, bins=None, range=None, weights=None, density=True):
    """Histogram

    TODO: add full detailed docstring that can be referenced from accessors and da/array interfaces."""
    return _apply_multi_input_function("histogram", data, dim, "dim", group=group, var_names=var_names, filter_vars=filter_vars, coords=coords, bins=bins, range=range, weights=weights, density=density)

def kde(data, dim=None, group="posterior", var_names=None, filter_vars=None, coords=None, circular=False, **kwargs):
    """KDE

    TODO: add full detailed docstring that can be referenced from accessors and da/array interfaces."""
    return _apply_multi_input_function("kde", data, dim, "dim", group=group, var_names=var_names, filter_vars=filter_vars, coords=coords, circular=circular, **kwargs)
