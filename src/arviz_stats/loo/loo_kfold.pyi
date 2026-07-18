# File generated with docstub

import numpy as np
import xarray
from arviz_base import rcParams
from numpy.typing import NDArray
from xarray import DataArray

from arviz_stats.loo.helper_loo_kfold import (
    _combine_fold_elpds,
    _compute_kfold_results,
    _prepare_kfold_inputs,
)
from arviz_stats.utils import ELPDData

from . import SamplingWrapper

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
