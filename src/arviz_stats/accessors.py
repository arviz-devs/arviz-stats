"""ArviZ stats accessors."""
import warnings

import xarray as xr
from arviz_base.utils import _var_names
from datatree import DataTree, register_datatree_accessor
from xarray_einstats.numba import ecdf

from arviz_stats.utils import get_function

__all__ = ["AzStatsDsAccessor", "AzStatsDaAccessor", "AzStatsDtAccessor"]


class UnsetDefault:
    pass


def update_dims(dims, da):
    """Update dims to contain only those present in da."""
    if dims is None:
        return None
    if isinstance(dims, str):
        dims = [dims]
    return [dim for dim in dims if dim in da.dims]


unset = UnsetDefault()


class _BaseAccessor:
    """Base accessor class."""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj


@xr.register_dataarray_accessor("azstats")
class AzStatsDaAccessor(_BaseAccessor):
    """ArviZ stats accessor class for DataArrays."""

    def eti(self, prob=None, dims=None, **kwargs):
        """Compute the equal tail interval on the DataArray."""
        return get_function("eti")(self._obj, prob=prob, dims=dims, **kwargs)

    def hdi(self, prob=None, dims=None, **kwargs):
        """Compute the highest density interval on the DataArray."""
        return get_function("hdi")(self._obj, prob=prob, dims=dims, **kwargs)

    def kde(self, dims=None, **kwargs):
        """Compute the KDE on the DataArray."""
        return get_function("kde")(self._obj, dims=dims, **kwargs)


@xr.register_dataset_accessor("azstats")
class AzStatsDsAccessor(_BaseAccessor):
    """ArviZ stats accessor class for Datasets.

    Notes
    -----
    Whenever "dims" indicates a set of dimensions that are to be reduced, the behaviour
    should be to reduce all present dimensions and ignore the ones not present.
    Thus, they can't use :meth:`.Dataset.map` and instead we must manually loop over variables
    in the dataset, remove elements from dims if necessary and afterwards rebuild the output
    Dataset.
    """

    @property
    def ds(self):
        """Return the underlying Dataset."""
        return self._obj

    def filter_vars(self, var_names=None, filter_vars=None):
        """Filter variables in the dataset.

        Parameters
        ----------
        var_names : iterable, optional
        filter_vars : {None, "like", "regex"}, default None

        Returns
        -------
        accessor
            This method returns the accessor after filtering its underlying xarray object.
            To get the filtered dataset, use ``.ds``.
        """
        var_names = _var_names(var_names=var_names, data=self._obj, filter_vars=filter_vars)
        if var_names is not None:
            self._obj = self._obj[var_names]
        return self

    def _apply(self, fun, dims, **kwargs):
        """Apply a function to all variables subsetting dims to existing dimensions."""
        return xr.Dataset(
            {
                var_name: fun(da, dims=update_dims(dims, da), **kwargs)
                for var_name, da in self._obj.items()
            }
        )

    def eti(self, prob=None, dims=None, **kwargs):
        """Compute the equal tail interval of all the variables in the dataset."""
        kwargs["prob"] = prob
        return self._apply(get_function("eti"), dims=dims, **kwargs)

    def hdi(self, prob=None, dims=None, **kwargs):
        """Compute hdi on all variables in the dataset."""
        kwargs["prob"] = prob
        return self._apply(get_function("hdi"), dims=dims, **kwargs)

    def ess(self, dims=None, method="bulk", relative=False, prob=None):
        """Compute the ess of all the variables in the dataset."""
        return self._apply(
            get_function("ess"), dims=dims, method=method, relative=relative, prob=prob
        )

    def rhat(self, dims=None, method="rank"):
        """Compute the rhat of all the variables in the dataset."""
        return self._apply(get_function("rhat"), dims=dims, method=method)

    def mcse(self, dims=None, method="mean", prob=None):
        """Compute the mcse of all the variables in the dataset."""
        return self._apply(get_function("mcse"), dims=dims, method=method, prob=prob)

    def kde(self, dims=None, **kwargs):
        """Compute the KDE for all variables in the dataset."""
        return self._apply(get_function("kde"), dims=dims, **kwargs)

    def histogram(self, dims=None, **kwargs):
        """Compute the KDE for all variables in the dataset."""
        return self._apply(get_function("histogram"), dims=dims, **kwargs)

    def ecdf(self, dims=None, **kwargs):
        """Compute the ecdf for all variables in the dataset."""
        # TODO: implement ecdf here so it doesn't depend on numba
        return self._apply(ecdf, dims=dims, **kwargs).rename(ecdf_axis="plot_axis")


@register_datatree_accessor("azstats")
class AzStatsDtAccessor(_BaseAccessor):
    """ArviZ stats accessor class for DataTrees."""

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

    def _apply(self, fun_name, dims, group, **kwargs):
        return DataTree.from_dict(
            {
                var_name: get_function(fun_name)(da, dims=update_dims(dims, da), **kwargs)
                for var_name, da in self._process_input(group, fun_name).items()
            },
            name=group,
        )

    def filter_vars(self, group="posterior", var_names=None, filter_vars=None):
        """Access and filter variables of the provided group."""
        ds = self._process_input(group, "filter_vars").ds

        return ds.azstats.filter_vars(var_names, filter_vars)

    def eti(self, prob=None, dims=None, group="posterior", **kwargs):
        """Compute the equal tail interval of all the variables in a group of the DataTree."""
        return self._apply("eti", prob=prob, dims=dims, group=group, **kwargs)

    def hdi(self, prob=None, dims=None, group="posterior", **kwargs):
        """Compute the highest density interval of all the variables in a group of the DataTree."""
        return self._apply("hdi", prob=prob, dims=dims, group=group, **kwargs)

    def ess(self, dims=None, group="posterior", **kwargs):
        """Compute ess of all variables in a group of the DataTree."""
        return self._apply("ess", dims=dims, group=group, **kwargs)

    def rhat(self, dims=None, group="posterior", method="rank"):
        """Compute the rhat of all the variables in a group of the DataTree."""
        return self._apply("rhat", dims=dims, group=group, method=method)

    def mcse(self, dims=None, group="posterior", method="mean", prob=None):
        """Compute the mcse of all the variables in a group of the DataTree."""
        return self._apply("mcse", dims=dims, group=group, method=method, prob=prob)

    def kde(self, dims=None, group="posterior", **kwargs):
        """Compute the KDE for all variables in a group of the DataTree."""
        return self._apply("kde", dims=dims, group=group, **kwargs)

    def histogram(self, dims=None, group="posterior", **kwargs):
        """Compute the KDE for all variables in a group of the DataTree."""
        return self._apply("histogram", dims=dims, group=group, **kwargs)
