"""ArviZ stats accessors."""

import warnings
from collections.abc import Hashable

import numpy as np
import xarray as xr
from arviz_base.utils import _var_names
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


def check_var_name_subset(obj, var_name):
    if isinstance(obj, xr.Dataset):
        return obj[var_name]
    if isinstance(obj, xr.DataTree):
        return obj.ds[var_name]
    return obj


def apply_function_to_dataset(func, ds, kwargs):
    return xr.Dataset(
        {
            var_name: func(
                da,
                **{
                    key: check_var_name_subset(value, var_name)
                    for key, value in update_kwargs_with_dims(da, kwargs).items()
                },
            )
            for var_name, da in ds.items()
        }
    )


unset = UnsetDefault()


class _BaseAccessor:
    """Base accessor class."""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def _apply(self, func, **kwargs):
        raise NotImplementedError("_apply private method needs to be implemented in subclasses")

    def eti(self, prob=None, dims=None, **kwargs):
        """Compute the equal tail interval.

        For full documentation and available arguments see :ref:`~arviz_stats.eti`
        """
        kwargs["prob"] = prob
        return self._apply("eti", dims=dims, **kwargs)

    def hdi(self, prob=None, dims=None, **kwargs):
        """Compute hdi on all variables in the dataset."""
        kwargs["prob"] = prob
        return self._apply("hdi", dims=dims, **kwargs)

    def ess(self, dims=None, method="bulk", relative=False, prob=None, **kwargs):
        """Compute the ess of all the variables in the dataset."""
        return self._apply("ess", dims=dims, method=method, relative=relative, prob=prob, **kwargs)

    def rhat(self, dims=None, method="rank", **kwargs):
        """Compute the rhat of all the variables in the dataset."""
        return self._apply("rhat", dims=dims, method=method, **kwargs)

    def mcse(self, dims=None, method="mean", prob=None, **kwargs):
        """Compute the mcse of all the variables in the dataset."""
        return self._apply("mcse", dims=dims, method=method, prob=prob, **kwargs)

    def kde(self, dims=None, **kwargs):
        """Compute the KDE for all variables in the dataset."""
        return self._apply("kde", dims=dims, **kwargs)

    def get_bins(self, dims=None, **kwargs):
        """Compute the histogram bin edges for all variables in the dataset."""
        return self._apply(get_function("get_bins"), dims=dims, **kwargs)

    def histogram(self, dims=None, **kwargs):
        """Compute the histogram for all variables in the dataset."""
        return self._apply("histogram", dims=dims, **kwargs)

    def compute_ranks(self, dims=None, relative=False, **kwargs):
        """Compute ranks for all variables in the dataset."""
        return self._apply("compute_ranks", dims=dims, relative=relative, **kwargs)

    def ecdf(self, dims=None, **kwargs):
        """Compute the ecdf for all variables in the dataset."""
        # TODO: implement ecdf here so it doesn't depend on numba
        return self._apply(ecdf, dims=dims, **kwargs).rename(ecdf_axis="plot_axis")

    def pareto_min_ss(self, dims=None, **kwargs):
        """Compute the min sample size for all variables in the dataset."""
        return self._apply("pareto_min_ss", dims=dims, **kwargs)

    def power_scale_lw(self, dims=None, **kwargs):
        """Compute log weights for power-scaling of the DataTree."""
        return self._apply("power_scale_lw", dims=dims, **kwargs)

    def power_scale_sense(self, dims=None, **kwargs):
        """Compute power-scaling sensitivity."""
        return self._apply("power_scale_sense", dims=dims, **kwargs)


@xr.register_dataarray_accessor("azstats")
class AzStatsDaAccessor(_BaseAccessor):
    """ArviZ stats accessor class for DataArrays."""

    def _apply(self, func, **kwargs):
        """Apply function to DataArray input."""
        if isinstance(func, str):
            func = get_function(func)
        return func(self._obj, **kwargs)


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

    @property
    def dataset(self):
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

    def _apply(self, func, **kwargs):
        """Apply a function to all variables subsetting dims to existing dimensions."""
        if isinstance(func, str):
            func = get_function(func)
        return apply_function_to_dataset(func, self._obj, kwargs=kwargs)

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


@xr.register_datatree_accessor("azstats")
class AzStatsDtAccessor(_BaseAccessor):
    """ArviZ stats accessor class for DataTrees."""

    @property
    def datatree(self):
        """Return the underlying Dataset."""
        return self._obj

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

    def _apply(self, func, **kwargs):
        group = kwargs.pop("group", "posterior")
        hashable_group = False
        if isinstance(group, Hashable):
            group = [group]
            hashable_group = True
        out_dt = xr.DataTree.from_dict(
            {
                group_i: apply_function_to_dataset(
                    get_function(func),
                    # if group is a single str/hashable that doesn't match the group name,
                    # still allow it and apply the function to the top level of the provided input
                    self._process_input(group_i, func, allow_non_matching=hashable_group),
                    kwargs=kwargs,
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
        if var_names is None:
            return self
        hashable_group = False
        if isinstance(group, Hashable):
            group = [group]
            hashable_group = True
        out_dt = xr.DataTree.from_dict(
            {
                group_i: self._process_input(  # pylint: disable=protected-access
                    group_i, "filter_vars", allow_non_matching=hashable_group
                )
                .dataset.azstats.filter_vars(var_names=var_names, filter_vars=filter_vars)
                ._obj
                for group_i in group
            }
        )
        if hashable_group:
            # if group was a string/hashable, return a datatree with a single node
            # (from the provided group) as the root of the DataTree
            return out_dt[group[0]].azstats
        # if group was a sequence, return a DataTree with multiple groups in the 1st level,
        # as many groups as requested
        return out_dt.azstats

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
