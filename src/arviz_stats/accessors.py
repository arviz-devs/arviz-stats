"""ArviZ stats accessors."""

import warnings
from collections.abc import Hashable

import numpy as np
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


def update_kwargs_with_dims(da, kwargs):
    """Update kwargs dict which may have a `dims` keyword."""
    kwargs = kwargs.copy()
    if "dims" in kwargs:
        kwargs.update({"dims": update_dims(kwargs["dims"], da)})
    return kwargs


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

    def thin(self, factor="auto", dims=None, **kwargs):
        """Perform thinning on the DataArray."""
        return get_function("thin")(self._obj, factor=factor, dims=dims, **kwargs)

    def pareto_min_ss(self, dims=None):
        """Compute the minimum effective sample size on the DataArray."""
        return get_function("pareto_min_ss")(self._obj, dims=dims)

    def power_scale_lw(self, alpha=1, dims=None):
        """Compute log weights for power-scaling of the DataTree."""
        return get_function("power_scale_lw")(self._obj, alpha=alpha, dims=dims)

    def power_scale_sense(self, lower_w=None, upper_w=None, delta=None, dims=None):
        """Compute power-scaling sensitivity."""
        return get_function("power_scale_sense")(
            self._obj, lower_w=lower_w, upper_w=upper_w, delta=delta, dims=dims
        )


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

    def _apply(self, fun, **kwargs):
        """Apply a function to all variables subsetting dims to existing dimensions."""
        if isinstance(fun, str):
            fun = get_function(fun)
        return xr.Dataset(
            {
                var_name: fun(da, **update_kwargs_with_dims(da, kwargs))
                for var_name, da in self._obj.items()
            }
        )

    def eti(self, prob=None, dims=None, **kwargs):
        """Compute the equal tail interval of all the variables in the dataset."""
        kwargs["prob"] = prob
        return self._apply("eti", dims=dims, **kwargs)

    def hdi(self, prob=None, dims=None, **kwargs):
        """Compute hdi on all variables in the dataset."""
        kwargs["prob"] = prob
        return self._apply("hdi", dims=dims, **kwargs)

    def ess(self, dims=None, method="bulk", relative=False, prob=None):
        """Compute the ess of all the variables in the dataset."""
        return self._apply("ess", dims=dims, method=method, relative=relative, prob=prob)

    def rhat(self, dims=None, method="rank"):
        """Compute the rhat of all the variables in the dataset."""
        return self._apply("rhat", dims=dims, method=method)

    def mcse(self, dims=None, method="mean", prob=None):
        """Compute the mcse of all the variables in the dataset."""
        return self._apply("mcse", dims=dims, method=method, prob=prob)

    def kde(self, dims=None, **kwargs):
        """Compute the KDE for all variables in the dataset."""
        return self._apply("kde", dims=dims, **kwargs)

    def get_bins(self, dims=None, **kwargs):
        """Compute the histogram bin edges for all variables in the dataset."""
        return self._apply(get_function("get_bins"), dims=dims, **kwargs)

    def histogram(self, dims=None, **kwargs):
        """Compute the histogram for all variables in the dataset."""
        return self._apply("histogram", dims=dims, **kwargs)

    def compute_ranks(self, dims=None, relative=False):
        """Compute ranks for all variables in the dataset."""
        return self._apply("compute_ranks", dims=dims, relative=relative)

    def ecdf(self, dims=None, **kwargs):
        """Compute the ecdf for all variables in the dataset."""
        # TODO: implement ecdf here so it doesn't depend on numba
        return self._apply(ecdf, dims=dims, **kwargs).rename(ecdf_axis="plot_axis")

    def thin_factor(self, **kwargs):
        """Get thinning factor for all the variables in the dataset."""
        reduce_func = kwargs.get("reduce_func", "mean")
        thin_factors = self._apply("thin_factor", **kwargs).to_array()
        if reduce_func == "mean":
            return int(np.floor(thin_factors.mean().item()))
        if reduce_func == "min":
            return int(np.floor(thin_factors.min().item()))
        raise ValueError(
            f"`reduce_func` {reduce_func} not recognized. Valid values are 'mean' or 'min'"
        )

    def thin(self, dims=None, factor="auto"):
        """Perform thinning for all the variables in the dataset."""
        if factor == "auto":
            factor = self.thin_factor()
            dims = "draw"
        return self._apply("thin", dims=dims, factor=factor)

    def pareto_min_ss(self, dims=None):
        """Compute the min sample size for all variables in the dataset."""
        return self._apply("pareto_min_ss", dims=dims)

    def power_scale_lw(self, dims=None, **kwargs):
        """Compute log weights for power-scaling of the DataTree."""
        return self._apply("power_scale_lw", dims=dims, **kwargs)

    def power_scale_sense(self, dims=None, **kwargs):
        """Compute power-scaling sensitivity."""
        return self._apply("power_scale_sense", dims=dims, **kwargs)


@register_datatree_accessor("azstats")
class AzStatsDtAccessor(_BaseAccessor):
    """ArviZ stats accessor class for DataTrees."""

    def _process_input(self, group, method, allow_non_matching=True):
        if self._obj.name == group:
            return self._obj
        if self._obj.children and group in self._obj.children:
            return self._obj[group]
        if allow_non_matching:
            warnings.warn(
                f"Computing {method} on DataTree named {self._obj.name} which doesn't match "
                f"the group argument {group}"
            )
            return self._obj
        raise ValueError(
            f"Group {group} not available in DataTree. Present groups are {self._obj.children} "
            f"and the DataTree itself is named {self._obs.name}"
        )

    def _apply(self, fun_name, group, **kwargs):
        hashable_group = False
        if isinstance(group, Hashable):
            group = [group]
            hashable_group = True
        out_dt = DataTree.from_dict(
            {
                group_i: xr.Dataset(
                    {
                        var_name: get_function(fun_name)(da, **update_kwargs_with_dims(da, kwargs))
                        for var_name, da in self._process_input(
                            # if group is a single str/hashable that doesn't match the group
                            # name, still allow it and apply the function to the top level of
                            # the provided datatree
                            group_i,
                            fun_name,
                            allow_non_matching=hashable_group,
                        ).items()
                    }
                )
                for group_i in group
            }
        )
        if hashable_group:
            # if group was a string/hashable, return a datatree with a single node
            # (from the provided group) as the root of the DataTree
            return out_dt[group[0]]
        # if group was a sequence, return a DataTree with multiple groups in the 1st level,
        # as many groups as requested
        return out_dt

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

    def thin_factor(self, group="posterior", **kwargs):
        """Get thinning factor for all the variables in a group of the datatree."""
        if not isinstance(group, str):
            raise ValueError("Thin factor can only be applied over a single group.")
        reduce_func = kwargs.get("reduce_func", "mean")
        thin_factors = self._apply("thin_factor", group=group, **kwargs)[group].ds.to_array()
        if reduce_func == "mean":
            return int(np.floor(thin_factors.mean().item()))
        if reduce_func == "min":
            return int(np.floor(thin_factors.min().item()))
        raise ValueError(
            f"`reduce_func` {reduce_func} not recognized. Valid values are 'mean' or 'min'"
        )

    def thin(self, dims=None, group="posterior", **kwargs):
        """Perform thinning for all variables in a group of the DataTree."""
        if kwargs.get("factor", "auto") == "auto":
            kwargs["factor"] = self.thin_factor()
            dims = "draw"
        return self._apply("thin", dims=dims, group=group, **kwargs)

    def pareto_min_ss(self, dims=None, group="posterior"):
        """Compute the min sample size for all variables in a group of the DataTree."""
        return self._apply("pareto_min_ss", dims=dims, group=group)

    def power_scale_lw(self, dims=None, group="log_likelihood", **kwargs):
        """Compute log weights for power-scaling of the DataTree."""
        return self._apply("power_scale_lw", dims=dims, group=group, **kwargs)

    def power_scale_sense(self, dims=None, group="posterior", **kwargs):
        """Compute power-scaling sensitivity."""
        return self._apply("power_scale_sense", dims=dims, group=group, **kwargs)
