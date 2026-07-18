# File generated with docstub

import warnings
from collections.abc import Sequence

import numpy as np
from _typeshed import Incomplete
from numpy.typing import ArrayLike, NDArray
from scipy.fftpack import next_fast_len
from scipy.interpolate import CubicSpline
from scipy.stats import circmean

from arviz_stats.base.stats_utils import round_num

class _CoreBase:
    def fft(self, x: ArrayLike) -> None: ...
    def rfft(self, ary: ArrayLike, n: int, axis: int = ...) -> None: ...
    def irfft(self, ary: ArrayLike, n: int, axis: int = ...) -> None: ...
    def autocov(self, ary: ArrayLike, axis: int = ...) -> None: ...
    def autocorr(self, ary: ArrayLike, axis: int = ...) -> None: ...
    def circular_mean(self, ary: ArrayLike) -> None: ...
    def _circular_standardize(self, ary: ArrayLike) -> None: ...
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
    def _float_rankdata(self, ary: ArrayLike) -> None: ...
    def _compute_ranks(self, ary: ArrayLike, relative: bool = ...) -> None: ...
    def _get_bininfo(self, values: ArrayLike, bins: int | str | ArrayLike = ...) -> None: ...
    def _get_bins(self, values: ArrayLike, bins: int | str | ArrayLike = ...) -> NDArray: ...
    def _histogram(
        self,
        ary: ArrayLike,
        bins: int | str | ArrayLike | None = ...,
        range: tuple | None = ...,
        weights: ArrayLike | None = ...,
        density: bool = ...,
    ) -> None: ...
    def _hdi_linear_nearest_common(self, ary: ArrayLike, prob: float) -> None: ...
    def _hdi_nearest(self, ary: ArrayLike, prob: float, circular: bool, skipna: bool) -> None: ...
    def _hdi_multimodal_continuous(
        self,
        ary: ArrayLike,
        prob: float,
        skipna: bool,
        max_modes: int,
        circular: bool,
        from_sample: bool = ...,
        **kwargs: Incomplete,
    ) -> None: ...
    def _hdi_multimodal_discrete(
        self,
        ary: ArrayLike,
        prob: float,
        max_modes: int,
        bins: int | str | ArrayLike | None = ...,
    ) -> None: ...
    def _hdi_from_point_densities(
        self, points: ArrayLike, densities: ArrayLike, prob: float, circular: bool
    ) -> None: ...
    def _hdi_from_bin_probabilities(
        self,
        bins: ArrayLike,
        bin_probs: ArrayLike,
        prob: float,
        circular: bool,
        dx: float,
    ) -> None: ...
    def _interval_points_to_bounds(
        self,
        points: ArrayLike,
        probs: ArrayLike,
        dx: float,
        circular: bool,
        period: float = ...,
    ) -> None: ...
    def _pad_hdi_to_maxmodes(
        self, hdi_intervals: ArrayLike, interval_probs: ArrayLike, max_modes: int
    ) -> None: ...
    def _mean(
        self,
        ary: ArrayLike,
        round_to: int | str | None = ...,
        skipna: bool = ...,
        axis: int | Sequence[int] | None | None = ...,
    ) -> None: ...
    def _median(
        self,
        ary: ArrayLike,
        round_to: int | str | None = ...,
        skipna: bool = ...,
        axis: int | Sequence[int] | None | None = ...,
    ) -> None: ...
    def _mode(
        self, ary: ArrayLike, round_to: int | str | None = ..., skipna: bool = ...
    ) -> None: ...
    def _std(
        self,
        ary: ArrayLike,
        round_to: int | str | None = ...,
        skipna: bool = ...,
        axis: int | Sequence[int] | None | None = ...,
    ) -> None: ...
    def _var(
        self,
        ary: ArrayLike,
        round_to: int | str | None = ...,
        skipna: bool = ...,
        axis: int | Sequence[int] | None | None = ...,
    ) -> None: ...
    def _mad(
        self,
        ary: ArrayLike,
        round_to: int | str | None = ...,
        skipna: bool = ...,
        axis: int | Sequence[int] | None | None = ...,
    ) -> None: ...
    def _iqr(
        self,
        ary: ArrayLike,
        quantiles: tuple[float, float] = ...,
        round_to: int | str | None = ...,
        skipna: bool = ...,
        axis: int | Sequence[int] | None | None = ...,
    ) -> None: ...
