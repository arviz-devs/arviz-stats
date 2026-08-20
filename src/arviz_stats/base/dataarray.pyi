# File generated with docstub

import numbers
import warnings
from collections.abc import Generator, Hashable, Sequence
from typing import Any, Literal

import numpy as np
from _typeshed import Incomplete
from arviz_base import rcParams
from numpy.typing import ArrayLike, NDArray
from xarray import DataArray, Dataset, apply_ufunc, broadcast, concat
from xarray_einstats.stats import _apply_nonreduce_func

from arviz_stats.base.array import BaseArray, array_stats
from arviz_stats.validate import (
    validate_ci_prob,
    validate_dims,
    validate_dims_chain_draw_axis,
    validate_prob,
)

class BaseDataArray:
    def __init__(self, array_class: BaseArray | None = ...) -> None: ...
    def eti(
        self,
        da: DataArray,
        prob: float | None = ...,
        dim: Hashable | Sequence[Hashable] | None = ...,
        method: str = ...,
        **kwargs: Incomplete,
    ) -> DataArray: ...
    def hdi(
        self,
        da: DataArray,
        prob: float | None = ...,
        dim: Hashable | Sequence[Hashable] | None = ...,
        method: str = ...,
        **kwargs: Incomplete,
    ) -> DataArray: ...
    def ess(
        self,
        da: DataArray,
        sample_dims: list[str] | None = ...,
        method: str = ...,
        relative: bool = ...,
        prob: float | None = ...,
    ) -> DataArray: ...
    def compute_ranks(
        self,
        da: DataArray,
        dim: Hashable | Sequence[Hashable] | None = ...,
        relative: bool = ...,
    ) -> DataArray: ...
    def rhat(
        self, da: DataArray, sample_dims: list[str] | None = ..., method: str = ...
    ) -> DataArray: ...
    def rhat_nested(
        self,
        da: DataArray,
        superchain_ids: ArrayLike,
        sample_dims: list[str] | None = ...,
        method: str = ...,
    ) -> DataArray: ...
    def mcse(
        self,
        da: DataArray,
        sample_dims: list[str] | None = ...,
        method: str = ...,
        prob: float | None = ...,
        circular: bool = ...,
    ) -> DataArray: ...
    def get_bins(
        self,
        da: DataArray,
        dim: Hashable | Sequence[Hashable] | None = ...,
        bins: str | numbers.Number | ArrayLike = ...,
    ) -> DataArray: ...
    def histogram(
        self,
        da: DataArray,
        dim: Hashable | Sequence[Hashable] | None = ...,
        bins: str | numbers.Number | ArrayLike | None = ...,
        range: tuple[float, float] | None = ...,
        weights: ArrayLike | None = ...,
        density: bool = ...,
    ) -> DataArray: ...
    def kde(
        self,
        da: DataArray,
        dim: str | Sequence[str] | None = ...,
        circular: bool = ...,
        grid_len: int = ...,
        **kwargs: Any,
    ) -> DataArray: ...
    def kde2d(
        self,
        da_x: DataArray,
        da_y: DataArray,
        dim: str | Sequence[str] | None = ...,
        gridsize: tuple[int] = ...,
        circular: bool = ...,
        hdi_probs: ArrayLike | None = ...,
    ) -> Dataset: ...
    def qds(
        self,
        da: DataArray,
        nquantiles: int = ...,
        binwidth: float | None = ...,
        dotsize: float = ...,
        stackratio: float = ...,
        top_only: bool = ...,
        dim: Hashable | Sequence[Hashable] | None = ...,
        **kwargs: Incomplete,
    ) -> DataArray: ...
    def ecdf(
        self,
        da: DataArray,
        npoints: int = ...,
        pit: bool = ...,
        dim: Hashable | Sequence[Hashable] | None = ...,
        **kwargs: Incomplete,
    ) -> DataArray: ...
    def uniformity_test(
        self,
        da: DataArray,
        dim: Hashable | Sequence[Hashable] | None = ...,
        method: str = ...,
        **kwargs: Incomplete,
    ) -> tuple[DataArray, DataArray, DataArray]: ...
    def mchain_uniformity_test(
        self, da: DataArray, dim: Sequence[Hashable] | None = ..., **kwargs: Incomplete
    ) -> tuple[DataArray, DataArray, DataArray]: ...
    def thin_factor(
        self,
        da: DataArray,
        target_ess: int | None = ...,
        reduce_func: Literal["mean", "min"] = ...,
    ) -> int: ...
    def thin(
        self,
        da: DataArray,
        factor: int | str = ...,
        sample_dims: list[str] | None = ...,
    ) -> DataArray: ...
    def pareto_min_ss(
        self, da: DataArray, sample_dims: list[str] | None = ...
    ) -> DataArray: ...
    def psislw(
        self,
        da: DataArray,
        r_eff: float = ...,
        dim: Hashable | Sequence[Hashable] | None = ...,
    ) -> tuple[DataArray, DataArray]: ...
    def bfmi(self, da: DataArray, sample_dims: list[str] | None = ...) -> DataArray: ...
    def pareto_khat(
        self,
        da: DataArray,
        sample_dims: list[str] | None = ...,
        r_eff: float | None = ...,
        tail: str = ...,
        log_weights: bool = ...,
    ) -> DataArray: ...
    def loo(
        self,
        da: DataArray,
        sample_dims: list[str] | None = ...,
        r_eff: float = ...,
        log_weights: DataArray | None = ...,
        pareto_k: DataArray | None = ...,
        log_jacobian: DataArray | None = ...,
    ) -> tuple[DataArray, DataArray, DataArray]: ...
    def loo_approximate_posterior(
        self,
        da: DataArray,
        log_p: DataArray,
        log_q: DataArray,
        sample_dims: list[str] | None = ...,
        log_jacobian: DataArray | None = ...,
    ) -> tuple[DataArray, DataArray, DataArray]: ...
    def loo_mixture(
        self,
        da: DataArray,
        sample_dims: list[str] | None = ...,
        log_jacobian: DataArray | None = ...,
    ) -> tuple[DataArray, DataArray, DataArray]: ...
    def loo_score(
        self,
        da: DataArray,
        y_obs: DataArray | numbers.Number,
        log_ratios: DataArray | None = ...,
        kind: str = ...,
        r_eff: float = ...,
        log_weights: DataArray | None = ...,
        pareto_k: DataArray | None = ...,
        sample_dims: list[str] | None = ...,
    ) -> tuple[DataArray, DataArray]: ...
    def loo_pit(
        self,
        da: DataArray,
        y_obs: DataArray,
        log_ratios: DataArray | None = ...,
        r_eff: float = ...,
        log_weights: DataArray | None = ...,
        pareto_k: DataArray | None = ...,
        sample_dims: list[str] | None = ...,
        random_state: int | Generator | None = ...,
        pareto_pit: bool = ...,
    ) -> tuple[DataArray, DataArray]: ...
    def loo_expectation(
        self,
        da: DataArray,
        log_ratios: DataArray | None = ...,
        kind: str = ...,
        r_eff: float = ...,
        log_weights: DataArray | None = ...,
        pareto_k: DataArray | None = ...,
        sample_dims: list[str] | None = ...,
    ) -> tuple[DataArray, DataArray]: ...
    def loo_quantile(
        self,
        da: DataArray,
        log_ratios: DataArray | None = ...,
        probs: float | ArrayLike | None = ...,
        r_eff: float = ...,
        log_weights: DataArray | None = ...,
        pareto_k: DataArray | None = ...,
        sample_dims: list[str] | None = ...,
    ) -> tuple[DataArray, DataArray]: ...
    def loo_summary(
        self, da: DataArray, p_loo_i: DataArray
    ) -> tuple[float, float, float, float]: ...
    def loo_r2(
        self,
        da: DataArray,
        ypred_loo: DataArray,
        n_simulations: int = ...,
        circular: bool = ...,
        random_state: int = ...,
    ) -> NDArray: ...
    def power_scale_lw(
        self,
        da: DataArray,
        alpha: float = ...,
        dim: Hashable | Sequence[Hashable] | None = ...,
    ) -> DataArray: ...
    def power_scale_sense(
        self,
        da: DataArray,
        lower_w: ArrayLike,
        upper_w: ArrayLike,
        lower_alpha: float,
        upper_alpha: float,
        sample_dims: list[str] | None = ...,
    ) -> DataArray: ...
    def autocorr(
        self, da: DataArray, dim: Hashable | Sequence[Hashable] | None = ...
    ) -> DataArray: ...
    def mean(
        self,
        da: DataArray,
        round_to: int | str | None = ...,
        skipna: bool = ...,
        dim: Hashable | Sequence[Hashable] | None = ...,
    ) -> DataArray: ...
    def median(
        self,
        da: DataArray,
        round_to: int | str | None = ...,
        skipna: bool = ...,
        dim: Hashable | Sequence[Hashable] | None = ...,
    ) -> DataArray: ...
    def mode(
        self,
        da: DataArray,
        round_to: int | str | None = ...,
        skipna: bool = ...,
        dim: Hashable | Sequence[Hashable] | None = ...,
    ) -> DataArray: ...
    def std(
        self,
        da: DataArray,
        round_to: int | str | None = ...,
        skipna: bool = ...,
        dim: Hashable | Sequence[Hashable] | None = ...,
    ) -> DataArray: ...
    def var(
        self,
        da: DataArray,
        round_to: int | str | None = ...,
        skipna: bool = ...,
        dim: Hashable | Sequence[Hashable] | None = ...,
    ) -> DataArray: ...
    def mad(
        self,
        da: DataArray,
        round_to: int | str | None = ...,
        skipna: bool = ...,
        dim: Hashable | Sequence[Hashable] | None = ...,
    ) -> DataArray: ...
    def iqr(
        self,
        da: DataArray,
        quantiles: tuple[float, float] = ...,
        round_to: int | str | None = ...,
        skipna: bool = ...,
        dim: Hashable | Sequence[Hashable] | None = ...,
    ) -> DataArray: ...
    def srs_estimator(
        self, da: DataArray, n_data_points: int
    ) -> tuple[float, float, float]: ...
    def diff_srs_estimator(
        self,
        elpd_loo_i_da: DataArray,
        lpd_approx_sample_da: DataArray,
        lpd_approx_all_da: DataArray,
        n_data_points: int,
    ) -> tuple[float, float, float]: ...

dataarray_stats: BaseDataArray
