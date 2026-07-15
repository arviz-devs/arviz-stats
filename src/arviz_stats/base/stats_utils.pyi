# File generated with docstub

import numbers
from collections.abc import Callable, Sequence

import numpy
from _typeshed import Incomplete
from numpy.typing import ArrayLike

__all__ = ["make_ufunc", "calculate_khat_bin_edges"]

_log: Incomplete

def make_ufunc(
    func: Callable,
    n_dims: int = ...,
    n_output: int = ...,
    n_input: int = ...,
    index: int = ...,
    ravel: bool = ...,
    check_shape: bool | None = ...,
) -> Callable: ...
def update_docstring(ufunc: Callable, func: Callable, n_output: int = ...) -> None: ...
def logsumexp(
    ary: ArrayLike,
    *,
    b: numbers.Number | None = ...,
    b_inv: numbers.Number | None = ...,
    axis: int | Sequence[int] | None = ...,
    keepdims: bool = ...,
    out: ArrayLike | None = ...,
    copy: bool = ...,
) -> None: ...
def not_valid(
    ary: numpy.ndarray,
    check_nan: bool = ...,
    check_shape: bool = ...,
    nan_kwargs: dict | None = ...,
    shape_kwargs: dict | None = ...,
) -> bool: ...
def calculate_khat_bin_edges(
    ary: ArrayLike, thresholds: Sequence[float], tolerance: float = ...
) -> list[float] | None: ...
def round_num(value: float, precision: int | str) -> None: ...
def get_decimal_places_from_se(se_val: float) -> None: ...
