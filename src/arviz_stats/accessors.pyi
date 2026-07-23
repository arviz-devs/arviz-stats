# File generated with docstub

import warnings
from collections.abc import Hashable, Iterable, Sequence
from typing import Any, Literal

import numpy as np
import xarray as xr
from _typeshed import Incomplete
from _typeshed import Incomplete as accessor
from arviz_base.utils import _var_names
from xarray import Dataset

from arviz_stats.utils import get_function

__all__ = ["AzStatsDsAccessor", "AzStatsDaAccessor", "AzStatsDtAccessor"]

class UnsetDefault:
    pass

def update_dims(dims: Incomplete, da: Incomplete) -> None: ...
def update_kwargs_with_dims(da: Incomplete, kwargs: Incomplete) -> None: ...
def check_var_name_subset(obj: Incomplete, var_name: Incomplete) -> None: ...
def apply_function_to_dataset(func: Incomplete, ds: Incomplete, kwargs: Incomplete) -> None: ...

unset: Incomplete

class _BaseAccessor:
    def __init__(self, xarray_obj: Incomplete) -> None: ...
    def _apply(self, func: Incomplete, **kwargs: Incomplete) -> None: ...
    def eti(self, prob: Incomplete = ..., dim: Incomplete = ..., **kwargs: Incomplete) -> None: ...
    def hdi(self, prob: Incomplete = ..., dim: Incomplete = ..., **kwargs: Incomplete) -> None: ...
    def ess(
        self,
        sample_dims: Incomplete = ...,
        method: Incomplete = ...,
        relative: Incomplete = ...,
        prob: Incomplete = ...,
        **kwargs: Incomplete,
    ) -> None: ...
    def rhat(
        self,
        sample_dims: Incomplete = ...,
        method: Incomplete = ...,
        **kwargs: Incomplete,
    ) -> None: ...
    def rhat_nested(
        self,
        sample_dims: Incomplete = ...,
        method: Incomplete = ...,
        superchain_ids: Incomplete = ...,
        **kwargs: Incomplete,
    ) -> None: ...
    def mcse(
        self,
        sample_dims: Incomplete = ...,
        method: Incomplete = ...,
        prob: Incomplete = ...,
        **kwargs: Incomplete,
    ) -> None: ...
    def kde(self, dim: str | Sequence[str] | None = ..., **kwargs: Any) -> Dataset: ...
    def qds(self, dim: Incomplete = ..., **kwargs: Incomplete) -> None: ...
    def get_bins(self, dim: Incomplete = ..., **kwargs: Incomplete) -> None: ...
    def histogram(self, dim: Incomplete = ..., **kwargs: Incomplete) -> None: ...
    def compute_ranks(
        self, dim: Incomplete = ..., relative: Incomplete = ..., **kwargs: Incomplete
    ) -> None: ...
    def ecdf(self, dim: Incomplete = ..., **kwargs: Incomplete) -> None: ...
    def uniformity_test(
        self, dim: Incomplete = ..., method: Incomplete = ..., **kwargs: Incomplete
    ) -> None: ...
    def mchain_uniformity_test(self, dim: Incomplete = ..., **kwargs: Incomplete) -> None: ...
    def pareto_min_ss(self, sample_dims: Incomplete = ..., **kwargs: Incomplete) -> None: ...
    def psislw(self, dim: Incomplete = ..., **kwargs: Incomplete) -> None: ...
    def bfmi(self, sample_dims: Incomplete = ..., **kwargs: Incomplete) -> None: ...
    def pareto_khat(self, sample_dims: Incomplete = ..., **kwargs: Incomplete) -> None: ...
    def loo(
        self,
        sample_dims: Incomplete = ...,
        r_eff: Incomplete = ...,
        log_weights: Incomplete = ...,
        pareto_k: Incomplete = ...,
        log_jacobian: Incomplete = ...,
    ) -> None: ...
    def loo_mixture(
        self, sample_dims: Incomplete = ..., log_jacobian: Incomplete = ...
    ) -> None: ...
    def loo_approximate_posterior(
        self,
        log_p: Incomplete,
        log_q: Incomplete,
        sample_dims: Incomplete = ...,
        log_jacobian: Incomplete = ...,
    ) -> None: ...
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
    ) -> None: ...
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
    ) -> None: ...
    def loo_expectation(
        self,
        log_ratios: Incomplete = ...,
        kind: Incomplete = ...,
        r_eff: Incomplete = ...,
        log_weights: Incomplete = ...,
        pareto_k: Incomplete = ...,
        sample_dims: Incomplete = ...,
        **kwargs: Incomplete,
    ) -> None: ...
    def loo_quantile(
        self,
        log_ratios: Incomplete = ...,
        probs: Incomplete = ...,
        r_eff: Incomplete = ...,
        log_weights: Incomplete = ...,
        pareto_k: Incomplete = ...,
        sample_dims: Incomplete = ...,
        **kwargs: Incomplete,
    ) -> None: ...
    def loo_summary(self, p_loo_i: Incomplete) -> None: ...
    def loo_r2(
        self,
        ypred_loo: Incomplete,
        n_simulations: Incomplete = ...,
        circular: Incomplete = ...,
        random_state: Incomplete = ...,
        **kwargs: Incomplete,
    ) -> None: ...
    def power_scale_lw(self, dim: Incomplete = ..., **kwargs: Incomplete) -> None: ...
    def power_scale_sense(self, sample_dims: Incomplete = ..., **kwargs: Incomplete) -> None: ...
    def autocorr(self, dim: Incomplete = ..., **kwargs: Incomplete) -> None: ...
    def mean(self, dim: Incomplete = ..., **kwargs: Incomplete) -> None: ...
    def median(self, dim: Incomplete = ..., **kwargs: Incomplete) -> None: ...
    def mode(self, dim: Incomplete = ..., **kwargs: Incomplete) -> None: ...
    def std(self, dim: Incomplete = ..., **kwargs: Incomplete) -> None: ...
    def var(self, dim: Incomplete = ..., **kwargs: Incomplete) -> None: ...
    def mad(self, dim: Incomplete = ..., **kwargs: Incomplete) -> None: ...
    def iqr(self, dim: Incomplete = ..., **kwargs: Incomplete) -> None: ...
    def srs_estimator(self, n_data_points: Incomplete, **kwargs: Incomplete) -> None: ...
    def diff_srs_estimator(
        self,
        lpd_approx_sample: Incomplete,
        lpd_approx_all: Incomplete,
        n_data_points: Incomplete,
        **kwargs: Incomplete,
    ) -> None: ...

class AzStatsDaAccessor(_BaseAccessor):
    def _apply(self, func: Incomplete, **kwargs: Incomplete) -> None: ...
    def thin(self, sample_dims: Incomplete = ..., factor: Incomplete = ...) -> None: ...

class AzStatsDsAccessor(_BaseAccessor):
    @property
    def ds(self) -> None: ...
    @property
    def dataset(self) -> None: ...
    def filter_vars(
        self,
        var_names: Iterable | None = ...,
        filter_vars: Literal[None, "like", "regex"] | None = ...,
    ) -> accessor: ...
    def _apply(self, func: Incomplete, **kwargs: Incomplete) -> None: ...
    def thin_factor(self, **kwargs: Incomplete) -> None: ...
    def thin(self, sample_dims: Incomplete = ..., factor: Incomplete = ...) -> None: ...

class AzStatsDtAccessor(_BaseAccessor):
    @property
    def datatree(self) -> None: ...
    def _process_input(
        self,
        group: Incomplete,
        method: Incomplete,
        allow_non_matching: Incomplete = ...,
    ) -> None: ...
    def _apply(self, func: Incomplete, **kwargs: Incomplete) -> None: ...
    def ecdf(self, *args: Incomplete, **kwargs: Incomplete) -> None: ...
    def filter_vars(
        self,
        group: Incomplete = ...,
        var_names: Incomplete = ...,
        filter_vars: Incomplete = ...,
    ) -> None: ...
    def thin_factor(self, group: Incomplete = ..., **kwargs: Incomplete) -> None: ...
    def thin(
        self,
        sample_dims: Incomplete = ...,
        group: Incomplete = ...,
        **kwargs: Incomplete,
    ) -> None: ...
