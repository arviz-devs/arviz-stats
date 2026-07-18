# File generated with docstub

import warnings

import numpy as np
import xarray as xr
from _typeshed import Incomplete
from arviz_base import rcParams
from numba import guvectorize
from xarray_einstats import _remove_indexes_to_reduce

__all__ = ["eti", "quantile"]

def _quantile(ary: Incomplete, q: Incomplete, result: Incomplete) -> None: ...
def quantile(da: Incomplete, q: Incomplete, dim: Incomplete) -> None: ...
def eti(
    da: Incomplete,
    prob: Incomplete = ...,
    dims: Incomplete = ...,
    method: Incomplete = ...,
    skipna: Incomplete = ...,
) -> None: ...
