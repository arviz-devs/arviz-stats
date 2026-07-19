# File generated with docstub

import warnings
from collections.abc import Hashable, Sequence

import numpy as np
import xarray as xr
from _typeshed import Incomplete
from arviz_base import rcParams
from numba import guvectorize
from numpy.typing import ArrayLike
from xarray import DataArray
from xarray_einstats import _remove_indexes_to_reduce

__all__ = ["eti", "quantile"]

def _quantile(ary: Incomplete, q: Incomplete, result: Incomplete) -> None: ...
def quantile(da: DataArray, q: ArrayLike, dim: str | Sequence[Hashable]) -> None: ...
def eti(
    da: DataArray,
    prob: float | None = ...,
    dims: str | Sequence[Hashable] | None = ...,
    method: str | None = ...,
    skipna: bool | None = ...,
) -> None: ...
