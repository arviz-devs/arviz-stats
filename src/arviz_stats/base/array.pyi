# File generated with docstub

from collections.abc import Generator, Sequence
from typing import Literal

from numpy.typing import ArrayLike

from arviz_stats.base.density import _DensityBase
from arviz_stats.base.diagnostics import _DiagnosticsBase

def process_chain_none(ary: ArrayLike, chain_axis: int, draw_axis: int) -> None: ...
def process_chain_none_multi(*arys: ArrayLike, chain_axis: int, draw_axis: int) -> None: ...
def process_ary_axes(ary: ArrayLike, axes: int | Sequence[int]) -> None: ...

class BaseArray(_DensityBase, _DiagnosticsBase):
    def hdi(
        self,
        ary: ArrayLike,
        prob: float,
        axis: int = ...,
        method: str = ...,
        circular: bool = ...,
        max_modes: int = ...,
        skipna: bool = ...,
        **kwargs: dict,
    ) -> None: ...
    def ess(
        self,
        ary: ArrayLike,
        chain_axis: int = ...,
        draw_axis: int = ...,
        method: str = ...,
        relative: bool = ...,
        prob: float | tuple[float, float] | None = ...,
    ) -> None: ...
    def rhat(
        self,
        ary: ArrayLike,
        chain_axis: int = ...,
        draw_axis: int = ...,
        method: str = ...,
    ) -> None: ...
    def rhat_nested(
        self,
        ary: ArrayLike,
        superchain_ids: ArrayLike,
        method: str = ...,
        chain_axis: int = ...,
        draw_axis: int = ...,
    ) -> None: ...
    def mcse(
        self,
        ary: ArrayLike,
        chain_axis: int = ...,
        draw_axis: int = ...,
        method: str = ...,
        prob: float | None = ...,
        circular: bool = ...,
    ) -> None: ...
    def pareto_min_ss(
        self, ary: ArrayLike, chain_axis: int = ..., draw_axis: int = ...
    ) -> None: ...
    def psislw(
        self, ary: ArrayLike, r_eff: float = ..., axis: int = ...
    ) -> tuple[ArrayLike, ArrayLike]: ...
    def bfmi(self, ary: ArrayLike, chain_axis: int = ..., draw_axis: int = ...) -> None: ...
    def pareto_khat(
        self,
        ary: ArrayLike,
        chain_axis: int = ...,
        draw_axis: int = ...,
        r_eff: float | None = ...,
        tail: str = ...,
        log_weights: bool = ...,
    ) -> None: ...
    def power_scale_lw(self, ary: ArrayLike, alpha: float = ..., axis: int = ...) -> None: ...
    def power_scale_sense(
        self,
        ary: ArrayLike,
        lower_w: ArrayLike,
        upper_w: ArrayLike,
        lower_alpha: float,
        upper_alpha: float,
        chain_axis: int = ...,
        draw_axis: int = ...,
    ) -> None: ...
    def compute_ranks(self, ary: ArrayLike, axis: int = ..., relative: bool = ...) -> None: ...
    def get_bins(self, ary: ArrayLike, axis: int = ..., bins: str = ...) -> None: ...
    def histogram(
        self,
        ary: ArrayLike,
        bins: str | None = ...,
        range: tuple[float, float] | None = ...,
        weights: ArrayLike | None = ...,
        axis: int = ...,
        density: bool = ...,
    ) -> tuple[ArrayLike, ArrayLike]: ...
    def kde(
        self,
        ary: ArrayLike,
        axis: int = ...,
        circular: bool = ...,
        grid_len: int = ...,
        **kwargs: dict,
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike]: ...
    def kde2d(
        self,
        x: ArrayLike,
        y: ArrayLike,
        gridsize: tuple[int] = ...,
        circular: bool = ...,
        hdi_probs: ArrayLike | None = ...,
        axis: int | Sequence[int] = ...,
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]: ...
    def qds(
        self,
        ary: ArrayLike,
        nquantiles: int = ...,
        binwidth: float | None = ...,
        dotsize: float = ...,
        stackratio: float = ...,
        top_only: bool = ...,
        axis: int = ...,
        **kwargs: dict,
    ) -> None: ...
    def ecdf(
        self,
        ary: ArrayLike,
        npoints: int = ...,
        pit: bool = ...,
        axis: int = ...,
        **kwargs: dict,
    ) -> tuple[ArrayLike, ArrayLike]: ...
    def uniformity_test(
        self, ary: ArrayLike, axis: int = ..., method: str = ..., **kwargs: dict
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike]: ...
    def mchain_uniformity_test(
        self,
        ary: ArrayLike,
        chain_axis: int = ...,
        draw_axis: int = ...,
        **kwargs: dict,
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike]: ...
    def bayesian_r2(
        self,
        mu_pred: ArrayLike,
        scale: ArrayLike | None = ...,
        scale_kind: str = ...,
        circular: bool = ...,
    ) -> None: ...
    def residual_r2(self, y_obs: ArrayLike, mu_pred: ArrayLike, circular: bool = ...) -> None: ...
    def metrics(self, observed: ArrayLike, predicted: ArrayLike, kind: str) -> None: ...
    def thin_factor(
        self,
        ary: ArrayLike,
        target_ess: int | None = ...,
        reduce_func: Literal["mean", "min"] = ...,
        chain_axis: int = ...,
        draw_axis: int = ...,
    ) -> int: ...
    def thin(
        self,
        ary: ArrayLike,
        factor: str | int = ...,
        chain_axis: int = ...,
        draw_axis: int = ...,
    ) -> ArrayLike: ...
    def mean(
        self,
        ary: ArrayLike,
        round_to: int | str | None = ...,
        skipna: bool = ...,
        axis: int | None = ...,
    ) -> ArrayLike: ...
    def median(
        self,
        ary: ArrayLike,
        round_to: int | str | None = ...,
        skipna: bool = ...,
        axis: int | None = ...,
    ) -> ArrayLike: ...
    def mode(
        self,
        ary: ArrayLike,
        round_to: int | str | None = ...,
        skipna: bool = ...,
        axis: int | None = ...,
    ) -> ArrayLike: ...
    def std(
        self,
        ary: ArrayLike,
        round_to: int | str | None = ...,
        skipna: bool = ...,
        axis: int | None = ...,
    ) -> ArrayLike: ...
    def var(
        self,
        ary: ArrayLike,
        round_to: int | str | None = ...,
        skipna: bool = ...,
        axis: int | None = ...,
    ) -> ArrayLike: ...
    def mad(
        self,
        ary: ArrayLike,
        round_to: int | str | None = ...,
        skipna: bool = ...,
        axis: int | None = ...,
    ) -> ArrayLike: ...
    def iqr(
        self,
        ary: ArrayLike,
        quantiles: tuple[float] = ...,
        round_to: int | str | None = ...,
        skipna: bool = ...,
        axis: int | None = ...,
    ) -> ArrayLike: ...
    def loo(
        self,
        ary: ArrayLike,
        chain_axis: int = ...,
        draw_axis: int = ...,
        r_eff: float = ...,
        log_weights: ArrayLike | None = ...,
        pareto_k: ArrayLike | None = ...,
        log_jacobian: ArrayLike | None = ...,
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike]: ...
    def loo_approximate_posterior(
        self,
        ary: ArrayLike,
        log_p: ArrayLike,
        log_q: ArrayLike,
        chain_axis: int = ...,
        draw_axis: int = ...,
        log_jacobian: float | None = ...,
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike]: ...
    def loo_mixture(
        self,
        ary: ArrayLike,
        obs_axes: tuple[int],
        chain_axis: int = ...,
        draw_axis: int = ...,
        log_jacobian: ArrayLike | None = ...,
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike]: ...
    def loo_score(
        self,
        ary: ArrayLike,
        y_obs: ArrayLike,
        log_weights: ArrayLike,
        kind: str = ...,
        chain_axis: int = ...,
        draw_axis: int = ...,
    ) -> ArrayLike: ...
    def loo_pit(
        self,
        ary: ArrayLike,
        y_obs: ArrayLike,
        log_weights: ArrayLike,
        chain_axis: int = ...,
        draw_axis: int = ...,
        random_state: int | Generator | None = ...,
        pareto_pit: bool = ...,
    ) -> ArrayLike: ...
    def loo_expectation(
        self,
        ary: ArrayLike,
        log_weights: ArrayLike,
        kind: str = ...,
        chain_axis: int = ...,
        draw_axis: int = ...,
    ) -> ArrayLike: ...
    def loo_quantile(
        self,
        ary: ArrayLike,
        log_weights: ArrayLike,
        prob: float,
        chain_axis: int = ...,
        draw_axis: int = ...,
    ) -> ArrayLike: ...
    def loo_summary(
        self, elpd_i: ArrayLike, p_loo_i: ArrayLike
    ) -> tuple[float, float, float, float]: ...
    def loo_r2(
        self,
        y_obs: ArrayLike,
        ypred_loo: ArrayLike,
        n_simulations: int = ...,
        circular: bool = ...,
        random_state: int = ...,
    ) -> ArrayLike: ...
    def srs_estimator(
        self, y_sample: ArrayLike, n_data_points: int
    ) -> tuple[float, float, float]: ...
    def diff_srs_estimator(
        self,
        elpd_loo_i: ArrayLike,
        lpd_approx_sample: ArrayLike,
        lpd_approx_all: ArrayLike,
        n_data_points: int,
    ) -> tuple[float, float, float]: ...

array_stats: BaseArray
