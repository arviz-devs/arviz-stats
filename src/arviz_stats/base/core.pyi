# File generated with docstub

from collections.abc import Sequence

from _typeshed import Incomplete
from numpy.typing import ArrayLike

class _CoreBase:
    def fft(self, x: ArrayLike) -> None: ...
    def rfft(self, ary: ArrayLike, n: int, axis: int = ...) -> None: ...
    def irfft(self, ary: ArrayLike, n: int, axis: int = ...) -> None: ...
    def autocov(self, ary: ArrayLike, axis: int = ...) -> None: ...
    def autocorr(self, ary: ArrayLike, axis: int = ...) -> None: ...
    def circular_mean(self, ary: ArrayLike) -> None: ...
    def _circular_standardize(self, ary: Incomplete) -> None: ...
    def quantile(
        self,
        ary: ArrayLike,
        quantile: float | ArrayLike,
        axis: int | Sequence[int] | None = ...,
        method: str = ...,
        skipna: bool = ...,
        weights: ArrayLike | None = ...,
    ) -> None: ...
    def eti(
        self,
        ary: ArrayLike,
        prob: float,
        axis: int | Sequence[int] | None = ...,
        method: str = ...,
        skipna: bool = ...,
        weights: ArrayLike | None = ...,
    ) -> None: ...
    def _float_rankdata(self, ary: Incomplete) -> None: ...
    def _compute_ranks(self, ary: Incomplete, relative: Incomplete = ...) -> None: ...
    def _get_bininfo(self, values: Incomplete, bins: Incomplete = ...) -> None: ...
    def _get_bins(self, values: ArrayLike, bins: int = ...) -> ArrayLike: ...
    def _histogram(
        self,
        ary: Incomplete,
        bins: Incomplete = ...,
        range: Incomplete = ...,
        weights: Incomplete = ...,
        density: Incomplete = ...,
    ) -> None: ...
    def _hdi_linear_nearest_common(self, ary: Incomplete, prob: Incomplete) -> None: ...
    def _hdi_nearest(
        self,
        ary: Incomplete,
        prob: Incomplete,
        circular: Incomplete,
        skipna: Incomplete,
    ) -> None: ...
    def _hdi_multimodal_continuous(
        self,
        ary: Incomplete,
        prob: Incomplete,
        skipna: Incomplete,
        max_modes: Incomplete,
        circular: Incomplete,
        from_sample: Incomplete = ...,
        **kwargs: Incomplete,
    ) -> None: ...
    def _hdi_multimodal_discrete(
        self,
        ary: Incomplete,
        prob: Incomplete,
        max_modes: Incomplete,
        bins: Incomplete = ...,
    ) -> None: ...
    def _hdi_from_point_densities(
        self,
        points: Incomplete,
        densities: Incomplete,
        prob: Incomplete,
        circular: Incomplete,
    ) -> None: ...
    def _hdi_from_bin_probabilities(
        self,
        bins: Incomplete,
        bin_probs: Incomplete,
        prob: Incomplete,
        circular: Incomplete,
        dx: Incomplete,
    ) -> None: ...
    def _interval_points_to_bounds(
        self,
        points: Incomplete,
        probs: Incomplete,
        dx: Incomplete,
        circular: Incomplete,
        period: Incomplete = ...,
    ) -> None: ...
    def _pad_hdi_to_maxmodes(
        self,
        hdi_intervals: Incomplete,
        interval_probs: Incomplete,
        max_modes: Incomplete,
    ) -> None: ...
    def _mean(
        self,
        ary: Incomplete,
        round_to: int | str | None = ...,
        skipna: bool = ...,
        axis: int | None = ...,
    ) -> None: ...
    def _median(
        self,
        ary: Incomplete,
        round_to: int | str | None = ...,
        skipna: bool = ...,
        axis: int | None = ...,
    ) -> None: ...
    def _mode(
        self, ary: Incomplete, round_to: Incomplete = ..., skipna: Incomplete = ...
    ) -> None: ...
    def _std(
        self,
        ary: Incomplete,
        round_to: int | str | None = ...,
        skipna: bool = ...,
        axis: int | None = ...,
    ) -> None: ...
    def _var(
        self,
        ary: Incomplete,
        round_to: int | str | None = ...,
        skipna: bool = ...,
        axis: int | None = ...,
    ) -> None: ...
    def _mad(
        self,
        ary: Incomplete,
        round_to: int | str | None = ...,
        skipna: bool = ...,
        axis: int | None = ...,
    ) -> None: ...
    def _iqr(
        self,
        ary: Incomplete,
        quantiles: tuple[float, float] = ...,
        round_to: int | str | None = ...,
        skipna: bool = ...,
        axis: int | None = ...,
    ) -> None: ...
