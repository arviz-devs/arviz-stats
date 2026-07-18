# File generated with docstub

import warnings
from collections import namedtuple
from collections.abc import Mapping
from copy import deepcopy

import numpy as np
import xarray as xr
from _typeshed import Incomplete
from arviz_base import convert_to_datatree, extract, ndarray_to_dataarray, rcParams
from numpy.typing import ArrayLike
from xarray import DataArray
from xarray_einstats.stats import logsumexp

from arviz_stats.utils import ELPDData, get_log_likelihood

__all__ = [
    "_shift",
    "_shift_and_scale",
    "_shift_and_cov",
    "_get_log_likelihood_i",
    "_get_weights_and_k_i",
    "_compute_loo_approximation",
    "_diff_srs_estimator",
    "_srs_estimator",
    "_compute_loo_results",
    "_generate_subsample_indices",
    "_get_r_eff",
    "_get_r_eff_i",
    "_prepare_loo_inputs",
    "_extract_loo_data",
    "_check_log_density",
    "_check_log_jacobian",
    "_warn_pareto_k",
    "_warn_pointwise_loo",
    "_prepare_subsample",
    "_prepare_update_subsample",
    "_select_obs_by_indices",
    "_select_obs_by_coords",
    "_prepare_full_arrays",
    "_align_data_to_obs",
    "_validate_crps_input",
    "_log_lik_i",
    "_validate_sample_dims",
    "_get_sample_coords",
    "_var_name_custom_ll",
    "_validate_log_lik_fn_result",
]

LooInputs: Incomplete

SubsampleData: Incomplete
UpdateSubsampleData: Incomplete

ShiftResult: Incomplete
ShiftAndScaleResult: Incomplete
ShiftAndCovResult: Incomplete

def _compute_loo_results(
    log_likelihood: Incomplete,
    var_name: Incomplete,
    sample_dims: Incomplete,
    n_samples: Incomplete,
    n_data_points: Incomplete,
    pointwise: Incomplete = ...,
    log_weights: Incomplete = ...,
    pareto_k: Incomplete = ...,
    reff: Incomplete = ...,
    approx_posterior: Incomplete = ...,
    return_pointwise: Incomplete = ...,
    log_jacobian: Incomplete = ...,
) -> None: ...
def _prepare_loo_inputs(
    data: Incomplete,
    var_name: Incomplete,
    thin_factor: Incomplete = ...,
    log_lik_fn: Incomplete = ...,
) -> None: ...
def _log_lik_i(
    i: Incomplete, data: Incomplete, var_name: Incomplete, log_lik_fn: Incomplete
) -> None: ...
def _extract_loo_data(loo_orig: Incomplete) -> None: ...
def _shift(upars: Incomplete, lwi: Incomplete) -> None: ...
def _shift_and_scale(upars: Incomplete, lwi: Incomplete) -> None: ...
def _shift_and_cov(upars: Incomplete, lwi: Incomplete) -> None: ...
def _get_log_likelihood_i(
    log_likelihood: Incomplete, i: Incomplete, obs_dims: Incomplete
) -> None: ...
def _get_weights_and_k_i(
    log_weights: Incomplete,
    pareto_k: Incomplete,
    i: Incomplete,
    obs_dims: Incomplete,
    sample_dims: Incomplete,
    data: Incomplete,
    n_samples: Incomplete,
    reff: Incomplete,
    log_lik_i: Incomplete,
    var_name: Incomplete,
) -> None: ...
def _prepare_subsample(
    data: Incomplete,
    log_likelihood_da: Incomplete,
    var_name: Incomplete,
    observations: Incomplete,
    seed: Incomplete,
    method: Incomplete,
    log_lik_fn: Incomplete,
    param_names: Incomplete,
    log: Incomplete,
    obs_dims: Incomplete,
    sample_dims: Incomplete,
    n_data_points: Incomplete,
    n_samples: Incomplete,
    thin_factor: Incomplete = ...,
) -> None: ...
def _prepare_update_subsample(
    loo_orig: Incomplete,
    data: Incomplete,
    observations: Incomplete,
    var_name: Incomplete,
    seed: Incomplete,
    method: Incomplete,
    log_lik_fn: Incomplete,
    param_names: Incomplete,
    log: Incomplete,
    thin_factor: Incomplete = ...,
) -> None: ...
def _compute_loo_approximation(
    data: Incomplete,
    var_name: Incomplete,
    log_lik_fn: Incomplete = ...,
    param_names: Incomplete = ...,
    method: Incomplete = ...,
    log: Incomplete = ...,
) -> None: ...
def _diff_srs_estimator(
    elpd_loo_i_sample: DataArray | ArrayLike,
    lpd_approx_sample: DataArray | ArrayLike,
    lpd_approx_all: DataArray | ArrayLike,
    n_data_points: int,
) -> tuple[float, float, float]: ...
def _srs_estimator(
    y_sample: DataArray,
    n_data_points: int,
) -> tuple[float, float, float]: ...
def _align_group(group: Incomplete, observed: Incomplete, primary_obs_dim: Incomplete) -> None: ...
def _align_data_to_obs(data: Incomplete, observed: Incomplete) -> None: ...
def _generate_subsample_indices(
    n_data_points: Incomplete, observations: Incomplete, seed: Incomplete
) -> None: ...
def _get_r_eff(data: Incomplete, n_samples: Incomplete) -> None: ...
def _get_r_eff_i(r_eff: Incomplete, i: Incomplete, obs_dims: Incomplete) -> None: ...
def _prepare_full_arrays(
    pointwise_values: Incomplete,
    pareto_k_values: Incomplete,
    ref_array: Incomplete,
    indices: Incomplete,
    obs_dims: Incomplete,
    elpd_loo_hat: Incomplete = ...,
) -> None: ...
def _select_obs_by_indices(
    data_array: Incomplete, indices: Incomplete, dims: Incomplete, dim_name: Incomplete
) -> None: ...
def _select_obs_by_coords(
    data_array: Incomplete,
    coord_array: Incomplete,
    dims: Incomplete,
    dim_name: Incomplete,
) -> None: ...
def _warn_pareto_k(
    pareto_k_values: Incomplete, n_samples: Incomplete, suppress: Incomplete = ...
) -> None: ...
def _warn_pointwise_loo(elpd: Incomplete, elpd_i_values: Incomplete) -> None: ...
def _check_log_density(
    log_dens: Incomplete,
    name: Incomplete,
    log_likelihood: Incomplete,
    n_samples: Incomplete,
    sample_dims: Incomplete,
) -> None: ...
def _check_log_jacobian(log_jacobian: Incomplete, obs_dims: Incomplete) -> None: ...
def _get_upars_info(upars: Incomplete, param_dim: Incomplete) -> None: ...
def _reconstruct_upars(upars_new_values: Incomplete, props: Incomplete) -> None: ...
def _has_nan_slice(da: Incomplete, dim: Incomplete) -> None: ...
def _validate_crps_input(
    y_pred: Incomplete,
    y_obs: Incomplete,
    log_likelihood: Incomplete,
    *,
    sample_dims: Incomplete,
    obs_dims: Incomplete,
) -> None: ...
def _validate_sample_dims(
    data: Incomplete,
    *,
    sample_dims: Incomplete = ...,
    ref_sizes: Incomplete = ...,
    obs_dims: Incomplete = ...,
) -> None: ...
def _get_sample_coords(sample_dims: Incomplete, data_for_fn: Incomplete) -> None: ...
def _validate_log_lik_fn_result(
    log_likelihood: Incomplete,
    sample_dims: Incomplete,
    obs_dims: Incomplete,
    observed: Incomplete,
) -> None: ...
def _var_name_custom_ll(data: Incomplete, var_name: Incomplete) -> None: ...
