# File generated with docstub

from collections.abc import Sequence
from typing import Any

from _typeshed import Incomplete
from numpy.typing import ArrayLike

from arviz_stats.base.array import BaseArray

def process_ary_axes(ary: ArrayLike, axes: int | Sequence[int]) -> None: ...
def _quantile_ufunc(a: Incomplete, q: Incomplete, out: Incomplete) -> None: ...
def _histogram_jit(ary: Incomplete, bins: Incomplete, range: Incomplete = ...) -> None: ...

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
    ) -> None: ...
    def _histogram(
        self,
        ary: Incomplete,
        bins: Incomplete = ...,
        range: Incomplete = ...,
        weights: Incomplete = ...,
        density: Incomplete = ...,
    ) -> None: ...
    @property
    def kde_ufunc(self) -> None: ...
    def kde(
        self,
        ary: ArrayLike,
        axis: int = ...,
        circular: bool = ...,
        grid_len: int = ...,
        **kwargs: Any,
    ) -> None: ...

NumbaArray: Incomplete
histogram: Incomplete
__doc__: Incomplete

array_stats: Incomplete
