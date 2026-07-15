# File generated with docstub

import numbers
from collections.abc import Generator, Hashable, Sequence
from typing import Any, Literal

from _typeshed import Incomplete
from numpy.typing import ArrayLike, NDArray
from xarray import DataArray, Dataset

class BaseDataArray:
    def __init__(self, array_class: Incomplete = ...) -> None: ...
    def eti(
        self,
        da: DataArray,
        prob: float | None = ...,
        dim: Sequence[Hashable] | None = ...,
        method: str = ...,
        **kwargs: dict,
    ) -> None: ...
    def hdi(
        self,
        da: DataArray,
        prob: float | None = ...,
        dim: Sequence[Hashable] | None = ...,
        method: str = ...,
        **kwargs: dict,
    ) -> None: ...
    def ess(
        self,
        da: DataArray,
        sample_dims: list[str] | None = ...,
        method: str = ...,
        relative: bool = ...,
        prob: float | None = ...,
    ) -> None: ...
    def compute_ranks(
        self, da: DataArray, dim: Sequence[Hashable] | None = ..., relative: bool = ...
    ) -> None: ...
    def rhat(
        self, da: DataArray, sample_dims: list[str] | None = ..., method: str = ...
    ) -> None: ...
    def rhat_nested(
        self,
        da: DataArray,
        superchain_ids: ArrayLike,
        sample_dims: list[str] | None = ...,
        method: str = ...,
    ) -> None: ...
    def mcse(
        self,
        da: DataArray,
        sample_dims: list[str] | None = ...,
        method: str = ...,
        prob: float | None = ...,
        circular: bool = ...,
    ) -> None: ...
    def get_bins(
        self, da: DataArray, dim: Sequence[Hashable] | None = ..., bins: str = ...
    ) -> None: ...
    def histogram(
        self,
        da: DataArray,
        dim: Sequence[Hashable] | None = ...,
        bins: str | None = ...,
        range: tuple[float, float] | None = ...,
        weights: ArrayLike | None = ...,
        density: bool = ...,
    ) -> None: ...
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
        dim: Sequence[Hashable] | None = ...,
        **kwargs: dict,
    ) -> None: ...
    def ecdf(
        self,
        da: DataArray,
        npoints: int = ...,
        pit: bool = ...,
        dim: Sequence[Hashable] | None = ...,
        **kwargs: dict,
    ) -> None: ...
    def uniformity_test(
        self,
        da: DataArray,
        dim: Sequence[Hashable] | None = ...,
        method: str = ...,
        **kwargs: dict,
    ) -> None: ...
    def mchain_uniformity_test(
        self, da: DataArray, dim: Sequence[Hashable] | None = ..., **kwargs: dict
    ) -> tuple[DataArray, DataArray, DataArray]: ...
    def thin_factor(
        self,
        da: DataArray,
        target_ess: int | None = ...,
        reduce_func: Literal["mean", "min"] = ...,
    ) -> None: ...
    def thin(
        self,
        da: DataArray,
        factor: int | str = ...,
        sample_dims: list[str] | None = ...,
    ) -> None: ...
    def pareto_min_ss(self, da: DataArray, sample_dims: list[str] | None = ...) -> None: ...
    def psislw(
        self, da: DataArray, r_eff: float = ..., dim: Sequence[Hashable] | None = ...
    ) -> None: ...
    def bfmi(self, da: DataArray, sample_dims: list[str] | None = ...) -> None: ...
    def pareto_khat(
        self,
        da: DataArray,
        sample_dims: list[str] | None = ...,
        r_eff: float | None = ...,
        tail: str = ...,
        log_weights: bool = ...,
    ) -> None: ...
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
        self, da: DataArray, alpha: float = ..., dim: Sequence[Hashable] | None = ...
    ) -> None: ...
    def power_scale_sense(
        self,
        da: DataArray,
        lower_w: ArrayLike,
        upper_w: ArrayLike,
        lower_alpha: float,
        upper_alpha: float,
        sample_dims: list[str] | None = ...,
    ) -> None: ...
    def autocorr(self, da: DataArray, dim: Sequence[Hashable] | None = ...) -> None: ...
    def mean(
        self,
        da: DataArray,
        round_to: int | str | None = ...,
        skipna: bool = ...,
        dim: Sequence[Hashable] | None = ...,
    ) -> None: ...
    def median(
        self,
        da: DataArray,
        round_to: int | str | None = ...,
        skipna: bool = ...,
        dim: Sequence[Hashable] | None = ...,
    ) -> None: ...
    def mode(
        self,
        da: DataArray,
        round_to: int | str | None = ...,
        skipna: bool = ...,
        dim: Sequence[Hashable] | None = ...,
    ) -> None: ...
    def std(
        self,
        da: DataArray,
        round_to: int | str | None = ...,
        skipna: bool = ...,
        dim: Sequence[Hashable] | None = ...,
    ) -> None: ...
    def var(
        self,
        da: DataArray,
        round_to: int | str | None = ...,
        skipna: bool = ...,
        dim: Sequence[Hashable] | None = ...,
    ) -> None: ...
    def mad(
        self,
        da: DataArray,
        round_to: int | str | None = ...,
        skipna: bool = ...,
        dim: Sequence[Hashable] | None = ...,
    ) -> None: ...
    def iqr(
        self,
        da: DataArray,
        quantiles: tuple[float, float] = ...,
        round_to: int | str | None = ...,
        skipna: bool = ...,
        dim: Sequence[Hashable] | None = ...,
    ) -> None: ...
    def srs_estimator(self, da: DataArray, n_data_points: int) -> tuple[float, float, float]: ...
    def diff_srs_estimator(
        self,
        elpd_loo_i_da: DataArray,
        lpd_approx_sample_da: DataArray,
        lpd_approx_all_da: DataArray,
        n_data_points: int,
    ) -> tuple[float, float, float]: ...

dataarray_stats: BaseDataArray
