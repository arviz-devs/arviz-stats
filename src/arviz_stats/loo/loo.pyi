# File generated with docstub

import numbers
from collections.abc import Callable

import xarray
from xarray import DataArray

from arviz_stats.utils import ELPDData

def loo(
    data: xarray.DataTree | xarray.DataTree,
    pointwise: bool | None = ...,
    var_name: str | None = ...,
    reff: float | None = ...,
    log_lik_fn: Callable | None = ...,
    log_weights: DataArray | None = ...,
    pareto_k: DataArray | None = ...,
    log_jacobian: DataArray | None = ...,
    mixture: bool = ...,
) -> ELPDData: ...
def loo_i(
    i: int | dict | numbers.Number,
    data: xarray.DataTree | xarray.DataTree,
    var_name: str | None = ...,
    reff: float | None = ...,
    log_lik_fn: Callable | None = ...,
    log_weights: DataArray | None = ...,
    pareto_k: DataArray | None = ...,
    log_jacobian: DataArray | None = ...,
) -> ELPDData: ...
