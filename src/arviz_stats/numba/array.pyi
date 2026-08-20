# File generated with docstub

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from _typeshed import Incomplete
from numba import guvectorize, jit
from numpy.typing import ArrayLike, NDArray

from arviz_stats.base.array import BaseArray

def process_ary_axes(ary: ArrayLike, axes: int | Sequence[int]) -> ArrayLike: ...
def _quantile_ufunc(a: Incomplete, q: Incomplete, out: Incomplete) -> None: ...
def _histogram_jit(
    ary: Incomplete, bins: Incomplete, range: Incomplete = ...
) -> None: ...

class NumbaArray(BaseArray):
    def __init__(self) -> None: ...
    def quantile(
        self,
        ary: Incomplete,
        quantile: Incomplete,
        axis: Incomplete = ...,
        method: Incomplete = ...,
        skipna: Incomplete = ...,
        weights: Incomplete = ...,
    ) -> NDArray: ...
    def _histogram(
        self,
        ary: Incomplete,
        bins: Incomplete = ...,
        range: Incomplete = ...,
        weights: Incomplete = ...,
        density: Incomplete = ...,
    ) -> tuple[np.ndarray, np.ndarray]: ...
    @property
    def kde_ufunc(self) -> Callable: ...
    def kde(
        self,
        ary: ArrayLike,
        axis: int | Sequence[int] | None = ...,
        circular: bool = ...,
        grid_len: int = ...,
        **kwargs: Any
    ) -> tuple[NDArray, NDArray, NDArray]: ...

def _extend_histogram_docstring() -> None: ...

array_stats: NumbaArray
