# File generated with docstub

import warnings
from collections import namedtuple
from collections.abc import Callable
from copy import deepcopy

import arviz_base as azb
import numpy as np
import xarray
import xarray as xr
from _typeshed import Incomplete
from _typeshed import Incomplete as Model
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
    model: Model | None = ...,
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
    i: Incomplete,
    upars: Incomplete,
    log_likelihood: Incomplete,
    log_prob_upars_fn: Incomplete,
    log_lik_i_upars_fn: Incomplete,
    max_iters: Incomplete,
    k_threshold: Incomplete,
    split: Incomplete,
    cov: Incomplete,
    orig_log_prob: Incomplete,
    ks: Incomplete,
    log_weights: Incomplete,
    pareto_k: Incomplete,
    r_eff: Incomplete,
    sample_dims: Incomplete,
    obs_dims: Incomplete,
    n_samples: Incomplete,
    n_params: Incomplete,
    param_dim_name: Incomplete,
    var_name: Incomplete,
) -> None: ...
def _update_loo_data_i(
    loo_data: Incomplete,
    i: Incomplete,
    new_elpd_i: Incomplete,
    new_pareto_k: Incomplete,
    log_liki: Incomplete,
    sample_dims: Incomplete,
    obs_dims: Incomplete,
    n_samples: Incomplete,
    n_eff_i: Incomplete = ...,
    original_log_liki: Incomplete = ...,
    log_weights_i: Incomplete = ...,
    suppress_warnings: Incomplete = ...,
) -> None: ...
def _update_quantities_i(
    upars: Incomplete,
    i: Incomplete,
    orig_log_prob: Incomplete,
    log_prob_upars_fn: Incomplete,
    log_lik_i_upars_fn: Incomplete,
    reff_i: Incomplete,
    sample_dims: Incomplete,
) -> None: ...
def _wrap__psislw(log_weights: Incomplete, sample_dims: Incomplete, r_eff: Incomplete) -> None: ...
