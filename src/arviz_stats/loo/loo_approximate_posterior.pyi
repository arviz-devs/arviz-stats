# File generated with docstub

import xarray
from arviz_base import rcParams
from numpy.typing import NDArray
from xarray import DataArray

from arviz_stats.loo.loo_helper import (
    _check_log_density,
    _check_log_jacobian,
    _prepare_loo_inputs,
    _warn_pareto_k,
    _warn_pointwise_loo,
)
from arviz_stats.utils import ELPDData

def loo_approximate_posterior(
    data: xarray.DataTree | xarray.DataTree,
    log_p: NDArray | DataArray,
    log_q: NDArray | DataArray,
    pointwise: bool | None = ...,
    var_name: str | None = ...,
    log_jacobian: DataArray | None = ...,
) -> ELPDData: ...
