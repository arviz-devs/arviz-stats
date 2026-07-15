# File generated with docstub

from collections.abc import Callable

import xarray
from numpy.typing import NDArray
from xarray import DataArray

from arviz_stats.utils import ELPDData

def loo_subsample(
    data: xarray.DataTree | xarray.DataTree,
    observations: int | NDArray,
    pointwise: bool | None = ...,
    var_name: str | None = ...,
    reff: float | None = ...,
    log_weights: DataArray | ELPDData | None = ...,
    log_p: NDArray | DataArray | None = ...,
    log_q: NDArray | DataArray | None = ...,
    seed: int = ...,
    method: str = ...,
    thin: int | str | None = ...,
    log_lik_fn: Callable | None = ...,
    param_names: list | None = ...,
    log: bool = ...,
    log_jacobian: DataArray | None = ...,
) -> ELPDData: ...
def update_subsample(
    loo_orig: ELPDData,
    data: xarray.DataTree | xarray.DataTree,
    observations: int | NDArray | None = ...,
    var_name: str | None = ...,
    reff: float | None = ...,
    log_weights: DataArray | ELPDData | None = ...,
    seed: int = ...,
    method: str = ...,
    log_lik_fn: Callable | None = ...,
    param_names: list | None = ...,
    log: bool = ...,
) -> ELPDData: ...
