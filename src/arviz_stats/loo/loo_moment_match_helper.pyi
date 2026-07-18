# File generated with docstub

from collections.abc import Callable

import numpy as np
import xarray
from _typeshed import Incomplete
from _typeshed import Incomplete as Model
from _typeshed import Incomplete as RandomVariable
from xarray import DataArray

__all__ = ["mm_from_pymc"]

def mm_from_pymc(
    idata: xarray.DataTree,
    *,
    model: Model | None = ...,
    var_name: str | None = ...,
) -> tuple[Callable, Callable, DataArray]: ...
def _validate_model(model: Model) -> None: ...
def _get_observed_rv(
    model: Model, idata: xarray.DataTree, var_name: str | None
) -> tuple[RandomVariable, str]: ...
def _get_upars_da(
    idata: xarray.DataTree, model: Model, initial_point: dict, value_vars: list
) -> tuple[DataArray, int]: ...
def _get_batched_func(
    inputs: Incomplete,
    outputs: Incomplete,
    initial_point: dict,
    *,
    dtype: Incomplete = ...,
    on_unused_input: Incomplete = ...,
) -> Callable: ...
