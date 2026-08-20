# File generated with docstub

import warnings
from collections.abc import Callable, Hashable, Iterable, Sequence
from typing import Any, Literal, NoReturn

import numpy as np
import xarray
import xarray as xr
from _typeshed import Incomplete
from arviz_base.utils import _var_names
from xarray import DataArray, Dataset

from arviz_stats.utils import get_function

__all__ = ["AzStatsDsAccessor", "AzStatsDaAccessor", "AzStatsDtAccessor"]

class UnsetDefault:
    pass

def update_dims(
    dims: str | Sequence[Hashable] | None, da: xr.DataArray
) -> list[Hashable] | None: ...
def update_kwargs_with_dims(da: xr.DataArray, kwargs: dict) -> dict: ...
def check_var_name_subset(
    obj: xr.Dataset | xr.DataTree | xr.DataArray, var_name: Hashable
) -> xr.DataArray: ...
def apply_function_to_dataset(
    func: Callable, ds: xr.Dataset, kwargs: dict
) -> xr.Dataset | tuple[xr.Dataset, ...]: ...

unset: Incomplete

class _BaseAccessor:
    def __init__(self, xarray_obj: xr.DataArray | xr.Dataset | xr.DataTree) -> None: ...
    def _apply(self, func: str | Callable, **kwargs: Incomplete) -> None: ...
    def eti(
        self, prob: Incomplete = ..., dim: Incomplete = ..., **kwargs: Incomplete
    ) -> Dataset: ...
    def hdi(
        self, prob: Incomplete = ..., dim: Incomplete = ..., **kwargs: Incomplete
    ) -> Dataset: ...
    def ess(
        self,
        sample_dims: Incomplete = ...,
        method: Incomplete = ...,
        relative: Incomplete = ...,
        prob: Incomplete = ...,
        **kwargs: Incomplete,
    ) -> Dataset: ...
    def rhat(
        self,
        sample_dims: Incomplete = ...,
        method: Incomplete = ...,
        **kwargs: Incomplete,
    ) -> Dataset: ...
    def rhat_nested(
        self,
        sample_dims: Incomplete = ...,
        method: Incomplete = ...,
        superchain_ids: Incomplete = ...,
        **kwargs: Incomplete,
    ) -> Dataset: ...
    def mcse(
        self,
        sample_dims: Incomplete = ...,
        method: Incomplete = ...,
        prob: Incomplete = ...,
        **kwargs: Incomplete,
    ) -> Dataset: ...
    def kde(self, dim: str | Sequence[str] | None = ..., **kwargs: Any) -> Dataset: ...
    def qds(self, dim: Incomplete = ..., **kwargs: Incomplete) -> Dataset: ...
    def get_bins(self, dim: Incomplete = ..., **kwargs: Incomplete) -> Dataset: ...
    def histogram(self, dim: Incomplete = ..., **kwargs: Incomplete) -> Dataset: ...
    def compute_ranks(
        self, dim: Incomplete = ..., relative: Incomplete = ..., **kwargs: Incomplete
    ) -> Dataset: ...
    def ecdf(self, dim: Incomplete = ..., **kwargs: Incomplete) -> Dataset: ...
    def uniformity_test(
        self, dim: Incomplete = ..., method: Incomplete = ..., **kwargs: Incomplete
    ) -> Dataset: ...
    def mchain_uniformity_test(
        self, dim: Incomplete = ..., **kwargs: Incomplete
    ) -> Dataset: ...
    def pareto_min_ss(
        self, sample_dims: Incomplete = ..., **kwargs: Incomplete
    ) -> Dataset: ...
    def psislw(
        self, dim: Incomplete = ..., **kwargs: Incomplete
    ) -> tuple[Dataset, Dataset]: ...
    def bfmi(self, sample_dims: Incomplete = ..., **kwargs: Incomplete) -> Dataset: ...
    def pareto_khat(
        self, sample_dims: Incomplete = ..., **kwargs: Incomplete
    ) -> Dataset: ...
    def loo(
        self,
        sample_dims: Incomplete = ...,
        r_eff: Incomplete = ...,
        log_weights: Incomplete = ...,
        pareto_k: Incomplete = ...,
        log_jacobian: Incomplete = ...,
    ) -> tuple[Dataset, Dataset, Dataset]: ...
    def loo_mixture(
        self, sample_dims: Incomplete = ..., log_jacobian: Incomplete = ...
    ) -> Dataset: ...
    def loo_approximate_posterior(
        self,
        log_p: Incomplete,
        log_q: Incomplete,
        sample_dims: Incomplete = ...,
        log_jacobian: Incomplete = ...,
    ) -> tuple[Dataset, Dataset, Dataset]: ...
    def loo_score(
        self,
        y_obs: Incomplete,
        log_ratios: Incomplete = ...,
        kind: Incomplete = ...,
        r_eff: Incomplete = ...,
        log_weights: Incomplete = ...,
        pareto_k: Incomplete = ...,
        sample_dims: Incomplete = ...,
        **kwargs: Incomplete,
    ) -> Dataset: ...
    def loo_pit(
        self,
        y_obs: Incomplete,
        log_ratios: Incomplete = ...,
        r_eff: Incomplete = ...,
        log_weights: Incomplete = ...,
        pareto_k: Incomplete = ...,
        sample_dims: Incomplete = ...,
        random_state: Incomplete = ...,
        pareto_pit: Incomplete = ...,
        **kwargs: Incomplete,
    ) -> Dataset: ...
    def loo_expectation(
        self,
        log_ratios: Incomplete = ...,
        kind: Incomplete = ...,
        r_eff: Incomplete = ...,
        log_weights: Incomplete = ...,
        pareto_k: Incomplete = ...,
        sample_dims: Incomplete = ...,
        **kwargs: Incomplete,
    ) -> Dataset: ...
    def loo_quantile(
        self,
        log_ratios: Incomplete = ...,
        probs: Incomplete = ...,
        r_eff: Incomplete = ...,
        log_weights: Incomplete = ...,
        pareto_k: Incomplete = ...,
        sample_dims: Incomplete = ...,
        **kwargs: Incomplete,
    ) -> Dataset: ...
    def loo_summary(self, p_loo_i: Incomplete) -> Dataset: ...
    def loo_r2(
        self,
        ypred_loo: Incomplete,
        n_simulations: Incomplete = ...,
        circular: Incomplete = ...,
        random_state: Incomplete = ...,
        **kwargs: Incomplete,
    ) -> Dataset: ...
    def power_scale_lw(
        self, dim: Incomplete = ..., **kwargs: Incomplete
    ) -> Dataset: ...
    def power_scale_sense(
        self, sample_dims: Incomplete = ..., **kwargs: Incomplete
    ) -> Dataset: ...
    def autocorr(self, dim: Incomplete = ..., **kwargs: Incomplete) -> Dataset: ...
    def mean(self, dim: Incomplete = ..., **kwargs: Incomplete) -> Dataset: ...
    def median(self, dim: Incomplete = ..., **kwargs: Incomplete) -> Dataset: ...
    def mode(self, dim: Incomplete = ..., **kwargs: Incomplete) -> Dataset: ...
    def std(self, dim: Incomplete = ..., **kwargs: Incomplete) -> Dataset: ...
    def var(self, dim: Incomplete = ..., **kwargs: Incomplete) -> Dataset: ...
    def mad(self, dim: Incomplete = ..., **kwargs: Incomplete) -> Dataset: ...
    def iqr(self, dim: Incomplete = ..., **kwargs: Incomplete) -> Dataset: ...
    def srs_estimator(
        self, n_data_points: Incomplete, **kwargs: Incomplete
    ) -> Dataset: ...
    def diff_srs_estimator(
        self,
        lpd_approx_sample: Incomplete,
        lpd_approx_all: Incomplete,
        n_data_points: Incomplete,
        **kwargs: Incomplete,
    ) -> Dataset: ...

class AzStatsDaAccessor(_BaseAccessor):
    def _apply(self, func: str | Callable, **kwargs: Incomplete) -> None: ...
    def thin(
        self, sample_dims: Incomplete = ..., factor: Incomplete = ...
    ) -> DataArray: ...

class AzStatsDsAccessor(_BaseAccessor):
    @property
    def ds(self) -> Dataset: ...
    @property
    def dataset(self) -> Dataset: ...
    def filter_vars(
        self,
        var_names: Iterable | None = ...,
        filter_vars: Literal[None, "like", "regex"] | None = ...,
    ) -> AzStatsDsAccessor: ...
    def _apply(self, func: str | Callable, **kwargs: Incomplete) -> None: ...
    def thin_factor(self, **kwargs: Incomplete) -> int: ...
    def thin(
        self, sample_dims: Incomplete = ..., factor: Incomplete = ...
    ) -> Dataset: ...

class AzStatsDtAccessor(_BaseAccessor):
    @property
    def datatree(self) -> xarray.DataTree: ...
    def _process_input(
        self, group: str, method: str, allow_non_matching: bool = ...
    ) -> None: ...
    def _apply(self, func: str | Callable, **kwargs: Incomplete) -> None: ...
    def ecdf(self, *args: Incomplete, **kwargs: Incomplete) -> NoReturn: ...
    def filter_vars(
        self,
        group: Incomplete = ...,
        var_names: Incomplete = ...,
        filter_vars: Incomplete = ...,
    ) -> AzStatsDtAccessor: ...
    def thin_factor(self, group: Incomplete = ..., **kwargs: Incomplete) -> int: ...
    def thin(
        self,
        sample_dims: Incomplete = ...,
        group: Incomplete = ...,
        **kwargs: Incomplete,
    ) -> xarray.DataTree: ...
