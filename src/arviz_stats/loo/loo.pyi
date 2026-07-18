# File generated with docstub

import numbers
import warnings
from collections.abc import Callable

import xarray
from arviz_base import rcParams
from xarray import DataArray
from xarray_einstats.stats import logsumexp

from arviz_stats.loo.helper_loo import (
    _check_log_jacobian,
    _compute_loo_results,
    _get_log_likelihood_i,
    _get_r_eff,
    _get_weights_and_k_i,
    _log_lik_i,
    _prepare_loo_inputs,
    _warn_pareto_k,
    _warn_pointwise_loo,
)
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
