import warnings

import xarray as xr
from arviz_base.utils import _var_names
from datatree import register_datatree_accessor

from .utils import get_function

__all__ = ["AzStatsDsAccessor", "AzStatsDaAccessor", "AzStatsDtAccessor"]


class _BaseAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj


@xr.register_dataarray_accessor("azstats")
class AzStatsDaAccessor(_BaseAccessor):
    def eti(self, prob=None, dims=None, method="linear", skipna=False):
        """Compute the equal tail interval on the DataArray."""
        return get_function("eti")(self._obj, prob=prob, dims=dims, method=method, skipna=skipna)


@xr.register_dataset_accessor("azstats")
class AzStatsDsAccessor(_BaseAccessor):
    def filter_vars(self, var_names=None, filter_vars=None):
        """Filter variables in the dataset."""
        var_names = _var_names(var_names=var_names, data=self._obj, filter_vars=filter_vars)
        if var_names is not None:
            self._obj = self._obj[var_names]
        return self

    def eti(self, prob=None, dims=None, method="linear", skipna=False):
        """Compute the equal tail interval of all the variables in the dataset."""
        return self._obj.map(
            get_function("eti"), prob=prob, dims=dims, method=method, skipna=skipna
        )

    def hdi(self, prob=None, dims=None, **kwargs):
        """Compute hdi on all variables in the dataset."""
        return self._obj.map(get_function("hdi"), prob=prob, dims=dims, **kwargs)

    def kde(self, dims=None, **kwargs):
        """Compute the KDE for all variables in the dataset."""
        return self._obj.map(get_function("kde"), dims=dims, **kwargs)


@register_datatree_accessor("azstats")
class AzStatsDtAccessor(_BaseAccessor):
    def _process_input(self, group, method):
        if self._obj.name == group:
            return self._obj
        if self._obj.children and group in self._obj.children:
            return self._obj[group]
        warnings.warn(
            f"Computing {method} on DataTree named {self._obj.name} which doesn't match "
            f"the group argument {group}"
        )
        return self._obj

    def eti(
        self,
        prob=None,
        dims=None,
        method="linear",
        skipna=False,
        group="posterior",
    ):
        """Compute the equal tail interval of all the variables in a group of the DataTree."""
        dt = self._process_input(group, "eti")
        return dt.map(get_function("eti"), prob=prob, dims=dims, method=method, skipna=skipna)
