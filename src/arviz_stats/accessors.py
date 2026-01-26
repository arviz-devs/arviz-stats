"""ArviZ stats accessors."""

import warnings
from collections.abc import Hashable

import numpy as np
import xarray as xr
from arviz_base.utils import _var_names

# from xarray_einstats.numba import ecdf
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
    """Update kwargs dict which may have a `dim` keyword."""
    kwargs = kwargs.copy()
    if "dim" in kwargs:
        kwargs.update({"dim": update_dims(kwargs["dim"], da)})
    # TODO: remove. Used only for ecdf right now which needs to be moved here
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
    result_dicts = None

    for var_name, da in ds.items():
        updated_kwargs = update_kwargs_with_dims(da, kwargs)
        subset_kwargs = {
            key: check_var_name_subset(value, var_name) for key, value in updated_kwargs.items()
        }
        result = func(da, **subset_kwargs)

        if not isinstance(result, tuple):
            result = (result,)

        if result_dicts is None:
            result_dicts = [{} for _ in range(len(result))]

        for i, res in enumerate(result):
            result_dicts[i][var_name] = res

    datasets = tuple(xr.Dataset(res_dict) for res_dict in result_dicts)

    return datasets if len(datasets) > 1 else datasets[0]


unset = UnsetDefault()


class _BaseAccessor:
    """Base accessor class."""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def _apply(self, func, **kwargs):
        raise NotImplementedError("_apply private method needs to be implemented in subclasses")

    def eti(self, prob=None, dim=None, **kwargs):
        """Compute the equal tail interval."""
        kwargs["prob"] = prob
        return self._apply("eti", dim=dim, **kwargs)

    def hdi(self, prob=None, dim=None, **kwargs):
        """Compute hdi on all variables in the dataset."""
        kwargs["prob"] = prob
        return self._apply("hdi", dim=dim, **kwargs)

    def ess(self, sample_dims=None, method="bulk", relative=False, prob=None, **kwargs):
        """Compute the ess of all the variables in the dataset."""
        return self._apply(
            "ess", sample_dims=sample_dims, method=method, relative=relative, prob=prob, **kwargs
        )

    def rhat(self, sample_dims=None, method="rank", **kwargs):
        """Compute the rhat of all the variables in the dataset."""
        return self._apply("rhat", sample_dims=sample_dims, method=method, **kwargs)

    def rhat_nested(self, sample_dims=None, method="rank", superchain_ids=None, **kwargs):
        """Compute nested rhat of all the variables in the dataset."""
        return self._apply(
            "rhat_nested",
            sample_dims=sample_dims,
            method=method,
            superchain_ids=superchain_ids,
            **kwargs,
        )

    def mcse(self, sample_dims=None, method="mean", prob=None, **kwargs):
        """Compute the mcse of all the variables in the dataset."""
        return self._apply("mcse", sample_dims=sample_dims, method=method, prob=prob, **kwargs)

    def kde(self, dim=None, **kwargs):
        """Compute the KDE for all variables in the dataset."""
        return self._apply("kde", dim=dim, **kwargs)

    def qds(self, dim=None, **kwargs):
        """Compute quantile dots."""
        return self._apply("qds", dim=dim, **kwargs)

    def get_bins(self, dim=None, **kwargs):
        """Compute the histogram bin edges for all variables in the dataset."""
        return self._apply(get_function("get_bins"), dim=dim, **kwargs)

    def histogram(self, dim=None, **kwargs):
        """Compute the histogram for all variables in the dataset."""
        return self._apply("histogram", dim=dim, **kwargs)

    def compute_ranks(self, dim=None, relative=False, **kwargs):
        """Compute ranks for all variables in the dataset."""
        return self._apply("compute_ranks", dim=dim, relative=relative, **kwargs)

    def ecdf(self, dim=None, **kwargs):
        """Compute the ecdf for all variables in the dataset."""
        return self._apply("ecdf", dim=dim, **kwargs)

    def pareto_min_ss(self, sample_dims=None, **kwargs):
        """Compute the min sample size for all variables in the dataset."""
        return self._apply("pareto_min_ss", sample_dims=sample_dims, **kwargs)

    def psislw(self, dim=None, **kwargs):
        """Pareto smoothed importance sampling."""
        return self._apply("psislw", dim=dim, **kwargs)

    def bfmi(self, sample_dims=None, **kwargs):
        """Calculate the estimated Bayesian fraction of missing information (BFMI)."""
        return self._apply("bfmi", sample_dims=sample_dims, **kwargs)

    def pareto_khat(self, sample_dims=None, **kwargs):
        """Compute Pareto k-hat diagnostic."""
        return self._apply("pareto_khat", sample_dims=sample_dims, **kwargs)

    def loo(self, sample_dims=None, r_eff=1.0, log_weights=None, pareto_k=None, log_jacobian=None):
        """Compute PSIS-LOO-CV."""
        return self._apply(
            "loo",
            sample_dims=sample_dims,
            r_eff=r_eff,
            log_weights=log_weights,
            pareto_k=pareto_k,
            log_jacobian=log_jacobian,
        )

    def loo_mixture(self, sample_dims=None, log_jacobian=None):
        """Compute mixture importance sampling LOO (Mix-IS-LOO)."""
        return self._apply(
            "loo_mixture",
            sample_dims=sample_dims,
            log_jacobian=log_jacobian,
        )

    def loo_approximate_posterior(self, log_p, log_q, sample_dims=None, log_jacobian=None):
        """Compute PSIS-LOO-CV with approximate posterior correction."""
        return self._apply(
            "loo_approximate_posterior",
            log_p=log_p,
            log_q=log_q,
            sample_dims=sample_dims,
            log_jacobian=log_jacobian,
        )

    def loo_score(
        self,
        y_obs,
        log_ratios=None,
        kind="crps",
        r_eff=1.0,
        log_weights=None,
        pareto_k=None,
        sample_dims=None,
        **kwargs,
    ):
        """Compute CRPS or SCRPS with PSIS-LOO-CV weights."""
        return self._apply(
            "loo_score",
            y_obs=y_obs,
            log_ratios=log_ratios,
            kind=kind,
            r_eff=r_eff,
            log_weights=log_weights,
            pareto_k=pareto_k,
            sample_dims=sample_dims,
            **kwargs,
        )

    def loo_pit(
        self,
        y_obs,
        log_ratios=None,
        r_eff=1.0,
        log_weights=None,
        pareto_k=None,
        sample_dims=None,
        random_state=None,
        **kwargs,
    ):
        """Compute LOO-PIT values with PSIS-LOO-CV weights."""
        return self._apply(
            "loo_pit",
            y_obs=y_obs,
            log_ratios=log_ratios,
            r_eff=r_eff,
            log_weights=log_weights,
            pareto_k=pareto_k,
            sample_dims=sample_dims,
            random_state=random_state,
            **kwargs,
        )

    def loo_expectation(
        self,
        log_ratios=None,
        kind="mean",
        r_eff=1.0,
        log_weights=None,
        pareto_k=None,
        sample_dims=None,
        **kwargs,
    ):
        """Compute weighted expectation with PSIS-LOO-CV weights."""
        return self._apply(
            "loo_expectation",
            log_ratios=log_ratios,
            kind=kind,
            r_eff=r_eff,
            log_weights=log_weights,
            pareto_k=pareto_k,
            sample_dims=sample_dims,
            **kwargs,
        )

    def loo_quantile(
        self,
        log_ratios=None,
        probs=None,
        r_eff=1.0,
        log_weights=None,
        pareto_k=None,
        sample_dims=None,
        **kwargs,
    ):
        """Compute weighted quantile with PSIS-LOO-CV weights."""
        return self._apply(
            "loo_quantile",
            log_ratios=log_ratios,
            probs=probs,
            r_eff=r_eff,
            log_weights=log_weights,
            pareto_k=pareto_k,
            sample_dims=sample_dims,
            **kwargs,
        )

    def loo_summary(self, p_loo_i):
        """Aggregate pointwise LOO values."""
        return self._apply("loo_summary", p_loo_i=p_loo_i)

    def loo_r2(
        self,
        ypred_loo,
        n_simulations=4000,
        circular=False,
        random_state=42,
        **kwargs,
    ):
        """Compute LOO-adjusted :math:`R^2` using Dirichlet-weighted bootstrap."""
        return self._apply(
            "loo_r2",
            ypred_loo=ypred_loo,
            n_simulations=n_simulations,
            circular=circular,
            random_state=random_state,
            **kwargs,
        )

    def power_scale_lw(self, dim=None, **kwargs):
        """Compute log weights for power-scaling of the DataTree."""
        return self._apply("power_scale_lw", dim=dim, **kwargs)

    def power_scale_sense(self, sample_dims=None, **kwargs):
        """Compute power-scaling sensitivity."""
        return self._apply("power_scale_sense", sample_dims=sample_dims, **kwargs)

    def autocorr(self, dim=None, **kwargs):
        """Compute autocorrelation for all variables in the dataset."""
        return self._apply("autocorr", dim=dim, **kwargs)

    def mean(self, dim=None, **kwargs):
        """Compute mean for all variables in the dataset."""
        return self._apply("mean", dim=dim, **kwargs)

    def median(self, dim=None, **kwargs):
        """Compute median for all variables in the dataset."""
        return self._apply("median", dim=dim, **kwargs)

    def mode(self, dim=None, **kwargs):
        """Compute mode for all variables in the dataset."""
        return self._apply("mode", dim=dim, **kwargs)

    def srs_estimator(self, n_data_points, **kwargs):
        """Compute simple random sampling estimate for subsampled LOO."""
        return self._apply(
            "srs_estimator",
            n_data_points=n_data_points,
            **kwargs,
        )

    def diff_srs_estimator(
        self,
        lpd_approx_sample,
        lpd_approx_all,
        n_data_points,
        **kwargs,
    ):
        """Difference estimator for subsampled LOO."""
        return self._apply(
            "diff_srs_estimator",
            lpd_approx_sample_da=lpd_approx_sample,
            lpd_approx_all_da=lpd_approx_all,
            n_data_points=n_data_points,
            **kwargs,
        )


@xr.register_dataarray_accessor("azstats")
class AzStatsDaAccessor(_BaseAccessor):
    """ArviZ stats accessor class for DataArrays."""

    def _apply(self, func, **kwargs):
        """Apply function to DataArray input."""
        if isinstance(func, str):
            func = get_function(func)
        return func(self._obj, **kwargs)

    def thin(self, sample_dims=None, factor="auto"):
        """Apply thinning to DataArray input."""
        return self._apply("thin", sample_dims=sample_dims, factor=factor)


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

    def thin(self, sample_dims=None, factor="auto"):
        """Perform thinning for all the variables in the dataset."""
        if factor == "auto":
            factor = self.thin_factor()
            sample_dims = "draw"
        return self._apply("thin", sample_dims=sample_dims, factor=factor)


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
            f"and the DataTree itself is named {self._obj.name}"
        )

    def _apply(self, func, **kwargs):
        group = kwargs.pop("group", "posterior")
        hashable_group = False
        if isinstance(group, Hashable):
            group = [group]
            hashable_group = True
        if isinstance(func, str):
            func = get_function(func)
        out_dt = xr.DataTree.from_dict(
            {
                group_i: apply_function_to_dataset(
                    func,
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

    def ecdf(self, *args, **kwargs):
        """Overwrite ecdf method with error until fixed for datatree."""
        raise NotImplementedError(
            "DataTree ecdf not available yet, use 'dt[group].dataset' instead."
        )

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
        thin_factors = self._apply("thin_factor", group=group, **kwargs).ds.to_array()
        if reduce_func == "mean":
            return int(np.floor(thin_factors.mean().item()))
        if reduce_func == "min":
            return int(np.floor(thin_factors.min().item()))
        raise ValueError(
            f"`reduce_func` {reduce_func} not recognized. Valid values are 'mean' or 'min'"
        )

    def thin(self, sample_dims=None, group="posterior", **kwargs):
        """Perform thinning for all variables in a group of the DataTree."""
        if kwargs.get("factor", "auto") == "auto":
            kwargs["factor"] = self.thin_factor()
            sample_dims = "draw"
        return self._apply("thin", sample_dims=sample_dims, group=group, **kwargs)
