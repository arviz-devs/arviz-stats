# File generated with docstub

from collections.abc import Callable

import xarray
from _typeshed import Incomplete
from _typeshed import Incomplete as SplitMomentMatch
from numpy.typing import NDArray
from xarray import DataArray

from arviz_stats.utils import ELPDData

SplitMomentMatch: Incomplete
UpdateQuantities: Incomplete
LooMomentMatchResult: Incomplete

def loo_moment_match(
    data: xarray.DataTree | xarray.DataTree,
    loo_orig: ELPDData,
    log_prob_upars_fn: Callable,
    log_lik_i_upars_fn: Callable,
    upars: DataArray | None = ...,
    var_name: str | None = ...,
    reff: float | None = ...,
    max_iters: int = ...,
    k_threshold: float | None = ...,
    split: bool = ...,
    cov: bool = ...,
    pointwise: bool | None = ...,
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
