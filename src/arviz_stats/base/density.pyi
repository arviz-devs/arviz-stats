# File generated with docstub

import math
import warnings
from collections.abc import Callable

import numpy as np
from _typeshed import Incomplete
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.signal import convolve, convolve2d
from scipy.signal.windows import gaussian
from scipy.sparse import coo_matrix
from scipy.special import betainc, ive
from scipy.stats import binom, hypergeom

from arviz_stats.base.core import _CoreBase

class _DensityBase(_CoreBase):
    bw_methods_linear: tuple[str, ...]

    def dct1d(self, x: ArrayLike) -> ArrayLike: ...
    def _fixed_point(self, t: float, n: int, k_sq: ArrayLike, a_sq: ArrayLike) -> None: ...
    def _root(
        self, function: Callable, n: int, args: tuple, x: ArrayLike, grid_range: float
    ) -> None: ...
    def bw_scott(self, x: ArrayLike, x_std: float | None = ..., **kwargs: dict) -> None: ...
    def bw_silverman(self, x: ArrayLike, x_std: float | None = ..., **kwargs: dict) -> None: ...
    def bw_isj(
        self,
        x: ArrayLike,
        grid_counts: ArrayLike | None = ...,
        x_std: float | None = ...,
        grid_range: float | None = ...,
    ) -> None: ...
    def bw_experimental(
        self,
        x: ArrayLike,
        grid_counts: ArrayLike | None = ...,
        x_std: float | None = ...,
        grid_range: float | None = ...,
    ) -> None: ...
    def get_bw(
        self,
        x: ArrayLike,
        bw: int,
        grid_counts: ArrayLike | None = ...,
        x_std: float | None = ...,
        grid_range: float | None = ...,
    ) -> float: ...
    def _normalize_angle(self, x: ArrayLike, zero_centered: bool = ...) -> None: ...
    def _vonmises_pdf(self, x: ArrayLike, mu: float, kappa: float) -> None: ...
    def _a1inv(self, x: float) -> None: ...
    def _kappa_mle(self, x: ArrayLike) -> None: ...
    def bw_taylor(self, x: ArrayLike) -> None: ...
    def check_custom_lims(self, custom_lims: list | tuple, x_min: float, x_max: float) -> None: ...
    def get_grid(
        self,
        x_min: float,
        x_max: float,
        x_std: float,
        extend_fct: bool,
        grid_len: int,
        custom_lims: tuple | list,
        extend: bool = ...,
        bound_correction: bool = ...,
    ) -> tuple[int, float, float]: ...
    def kde_linear(
        self,
        x: ArrayLike,
        bw: int = ...,
        adaptive: bool = ...,
        extend: bool = ...,
        bound_correction: bool = ...,
        extend_fct: float = ...,
        bw_fct: float = ...,
        custom_lims: list | tuple | None = ...,
        cumulative: bool = ...,
        grid_len: int = ...,
    ) -> tuple[ArrayLike, ArrayLike, float]: ...
    def kde_convolution(
        self,
        x: ArrayLike,
        bw: float,
        grid_edges: ArrayLike,
        grid_counts: ArrayLike,
        grid_len: int,
        bound_correction: bool,
    ) -> None: ...
    def kde_adaptive(
        self,
        x: ArrayLike,
        bw: float,
        grid_edges: ArrayLike,
        grid_counts: ArrayLike,
        grid_len: int,
        bound_correction: bool,
    ) -> None: ...
    def kde_circular(
        self,
        x: ArrayLike,
        bw: int = ...,
        bw_fct: float = ...,
        custom_lims: list | tuple | None = ...,
        cumulative: bool = ...,
        grid_len: int = ...,
    ) -> None: ...
    def _kde(
        self, x: ArrayLike, circular: bool = ..., grid_len: int = ..., **kwargs: Incomplete
    ) -> None: ...
    def _fast_kde_2d(
        self, x: ArrayLike, y: ArrayLike, gridsize: tuple = ..., circular: bool = ...
    ) -> tuple[ArrayLike, float, float, float, float]: ...
    def _find_hdi_contours(self, density: ArrayLike, hdi_probs: ArrayLike) -> NDArray: ...
    def _qds(
        self,
        x: ArrayLike,
        nquantiles: int,
        binwidth: float | None,
        dotsize: float,
        stackratio: float,
        top_only: bool,
    ) -> None: ...
    def _ecdf(self, ary: ArrayLike, npoints: int, pit: bool) -> None: ...
    @staticmethod
    def _shapley_mean(values: ArrayLike) -> None: ...
    def _pot_c(self, ary: ArrayLike) -> None: ...
    def _prit_c(self, ary: ArrayLike) -> None: ...
    def _piet_c(self, ary: ArrayLike) -> None: ...
    def _mtc_c(self, ary: ArrayLike) -> tuple[float, NDArray, NDArray]: ...
    def _cauchy_combination(
        self, ps: ArrayLike, cauchy_vals: ArrayLike, truncate: bool
    ) -> None: ...
    def _compute_quantiles_and_binwidth(
        self, values: ArrayLike, nquantiles: int, binwidth: float | None = ...
    ) -> tuple[NDArray, float]: ...
    def _wilkinson_algorithm(
        self, values: ArrayLike, binwidth: float
    ) -> tuple[list[float], list[int]]: ...
    def _layout_stacks(
        self,
        stack_locs: ArrayLike,
        stack_counts: ArrayLike[int],
        nquantiles: int,
        dotheight: float,
        top_only: bool,
    ) -> tuple[NDArray, NDArray]: ...
