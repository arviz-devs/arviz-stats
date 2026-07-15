# File generated with docstub

import xarray
from numpy.typing import NDArray
from xarray import DataArray

from arviz_stats.utils import ELPDData

def loo_approximate_posterior(
    data: xarray.DataTree | xarray.DataTree,
    log_p: NDArray | DataArray,
    log_q: NDArray | DataArray,
    pointwise: bool | None = ...,
    var_name: str | None = ...,
    log_jacobian: DataArray | None = ...,
) -> ELPDData: ...
