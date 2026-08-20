# File generated with docstub

from collections.abc import Callable

import numpy as np
import xarray
import xarray as xr
from arviz_base import rcParams
from numpy.typing import NDArray
from xarray import DataArray
from xarray_einstats.stats import logsumexp

from arviz_stats.loo.loo_approximate_posterior import loo_approximate_posterior
from arviz_stats.loo.loo_helper import (
    _check_log_jacobian,
    _compute_loo_results,
    _get_r_eff,
    _prepare_full_arrays,
    _prepare_loo_inputs,
    _prepare_subsample,
    _prepare_update_subsample,
    _select_obs_by_coords,
    _select_obs_by_indices,
    _warn_pareto_k,
)
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
