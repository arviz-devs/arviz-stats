# File generated with docstub

import warnings
from collections import namedtuple
from collections.abc import Callable
from copy import deepcopy

import arviz_base as azb
import numpy as np
import pymc
import xarray
import xarray as xr
from _typeshed import Incomplete
from _typeshed import Incomplete as SplitMomentMatch
from arviz_base import dataset_to_dataarray, rcParams
from numpy.typing import NDArray
from xarray import DataArray
from xarray_einstats.stats import logsumexp

from arviz_stats.loo.helper_loo import (
    _get_log_likelihood_i,
    _get_r_eff,
    _get_r_eff_i,
    _get_weights_and_k_i,
    _prepare_loo_inputs,
    _shift,
    _shift_and_cov,
    _shift_and_scale,
    _warn_pareto_k,
)
from arviz_stats.sampling_diagnostics import ess
from arviz_stats.utils import ELPDData

SplitMomentMatch: Incomplete
UpdateQuantities: Incomplete
LooMomentMatchResult: Incomplete

def loo_moment_match(
    data: xarray.DataTree | xarray.DataTree,
    loo_orig: ELPDData,
    log_prob_upars_fn: Callable | None = ...,
    log_lik_i_upars_fn: Callable | None = ...,
    upars: DataArray | None = ...,
    var_name: str | None = ...,
    reff: float | None = ...,
    max_iters: int = ...,
    k_threshold: float | None = ...,
    split: bool = ...,
    cov: bool = ...,
    pointwise: bool | None = ...,
    model: pymc.Model | None = ...,
) -> ELPDData: ...
def _split_moment_match(
    upars: DataArray,
    cov: bool,
    total_shift: NDArray,
    total_scaling: NDArray,
    total_mapping: NDArray,
    i: int,
    reff: float,
    log_prob_upars_fn: Callable,
    log_lik_i_upars_fn: Callable,
) -> SplitMomentMatch: ...
def _loo_moment_match_i(
    i: int,
    upars: xr.DataArray,
    log_likelihood: xr.DataArray,
    log_prob_upars_fn: Callable,
    log_lik_i_upars_fn: Callable,
    max_iters: int,
    k_threshold: float,
    split: bool,
    cov: bool,
    orig_log_prob: xr.DataArray,
    ks: np.ndarray,
    log_weights: xr.DataArray | None,
    pareto_k: xr.DataArray,
    r_eff: xr.DataArray | float | None,
    sample_dims: list,
    obs_dims: list,
    n_samples: int,
    n_params: int,
    param_dim_name: str,
    var_name: str,
) -> None: ...
def _update_loo_data_i(
    loo_data: ELPDData,
    i: int,
    new_elpd_i: float,
    new_pareto_k: float,
    log_liki: xr.DataArray,
    sample_dims: list,
    obs_dims: list,
    n_samples: int,
    n_eff_i: float | None = ...,
    original_log_liki: xr.DataArray | None = ...,
    log_weights_i: xr.DataArray | None = ...,
    suppress_warnings: bool = ...,
) -> None: ...
def _update_quantities_i(
    upars: xr.DataArray,
    i: int,
    orig_log_prob: xr.DataArray,
    log_prob_upars_fn: Callable,
    log_lik_i_upars_fn: Callable,
    reff_i: float,
    sample_dims: list,
) -> None: ...
def _wrap__psislw(log_weights: xr.DataArray, sample_dims: list, r_eff: float) -> None: ...
