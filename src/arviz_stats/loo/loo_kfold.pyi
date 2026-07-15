# File generated with docstub

import xarray
from numpy.typing import NDArray
from xarray import DataArray

from arviz_stats.utils import ELPDData

from .wrapper import SamplingWrapper

def loo_kfold(
    data: xarray.DataTree | xarray.DataTree,
    wrapper: SamplingWrapper,
    pointwise: bool | None = ...,
    var_name: str | None = ...,
    k: int = ...,
    folds: NDArray | DataArray | None = ...,
    stratify_by: NDArray | DataArray | None = ...,
    group_by: NDArray | DataArray | None = ...,
    save_fits: bool = ...,
) -> ELPDData: ...
