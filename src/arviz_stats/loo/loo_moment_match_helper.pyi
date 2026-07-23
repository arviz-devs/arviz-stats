# File generated with docstub

from collections.abc import Callable

import numpy as np
import pymc
import pytensor
import xarray
from xarray import DataArray

__all__ = ["mm_from_pymc"]

def mm_from_pymc(
    idata: xarray.DataTree,
    *,
    model: pymc.Model | None = ...,
    var_name: str | None = ...,
) -> tuple[Callable, Callable, DataArray]: ...
def _validate_model(model: pymc.Model) -> None: ...
def _get_observed_rv(
    model: pymc.Model, idata: xarray.DataTree, var_name: str | None
) -> tuple[pytensor.tensor.TensorVariable, str]: ...
def _get_upars_da(
    idata: xarray.DataTree, model: pymc.Model, initial_point: dict, value_vars: list
) -> tuple[DataArray, int]: ...
def _get_batched_func(
    inputs: list[pytensor.tensor.TensorVariable],
    outputs: list[pytensor.tensor.TensorVariable],
    initial_point: dict,
    *,
    dtype: str = ...,
    on_unused_input: str = ...,
) -> Callable: ...
