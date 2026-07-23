# File generated with docstub

import warnings
from collections import namedtuple
from collections.abc import Callable, Mapping
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

LooInputs: type

SubsampleData: type
UpdateSubsampleData: type

ShiftResult: type
ShiftAndScaleResult: type
ShiftAndCovResult: type

def _compute_loo_results(
    log_likelihood: xr.DataArray | xr.Dataset,
    var_name: str | None,
    sample_dims: list,
    n_samples: int,
    n_data_points: int,
    pointwise: bool | None = ...,
    log_weights: xr.DataArray | xr.Dataset | None = ...,
    pareto_k: xr.DataArray | xr.Dataset | None = ...,
    reff: float | None = ...,
    approx_posterior: bool = ...,
    return_pointwise: bool = ...,
    log_jacobian: xr.DataArray | None = ...,
) -> None: ...
def _prepare_loo_inputs(
    data: xr.DataTree,
    var_name: str | None,
    thin_factor: int | None = ...,
    log_lik_fn: Callable | None = ...,
) -> None: ...
def _log_lik_i(
    i: int, data: xr.DataTree, var_name: str | None, log_lik_fn: Callable | None
) -> None: ...
def _extract_loo_data(loo_orig: ELPDData) -> None: ...
def _shift(upars: xr.DataArray, lwi: xr.DataArray) -> None: ...
def _shift_and_scale(upars: xr.DataArray, lwi: xr.DataArray) -> None: ...
def _shift_and_cov(upars: xr.DataArray, lwi: xr.DataArray) -> None: ...
def _get_log_likelihood_i(log_likelihood: xr.DataArray, i: Incomplete, obs_dims: list) -> None: ...
def _get_weights_and_k_i(
    log_weights: xr.DataArray | xr.Dataset | None,
    pareto_k: xr.DataArray | xr.Dataset | None,
    i: Incomplete,
    obs_dims: list,
    sample_dims: list,
    data: xr.DataTree,
    n_samples: int,
    reff: float | None,
    log_lik_i: xr.DataArray,
    var_name: str | None,
) -> None: ...
def _prepare_subsample(
    data: xr.DataTree,
    log_likelihood_da: xr.DataArray,
    var_name: str | None,
    observations: int | np.ndarray,
    seed: int | None,
    method: str,
    log_lik_fn: Callable | None,
    param_names: list | None,
    log: bool,
    obs_dims: list,
    sample_dims: list,
    n_data_points: int,
    n_samples: int,
    thin_factor: int | None = ...,
) -> None: ...
def _prepare_update_subsample(
    loo_orig: ELPDData,
    data: xr.DataTree,
    observations: int | np.ndarray,
    var_name: str | None,
    seed: int | None,
    method: str,
    log_lik_fn: Callable | None,
    param_names: list | None,
    log: bool,
    thin_factor: int | None = ...,
) -> None: ...
def _compute_loo_approximation(
    data: xr.DataTree,
    var_name: str,
    log_lik_fn: Callable | None = ...,
    param_names: list | None = ...,
    method: str = ...,
    log: bool = ...,
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
def _align_group(group: xr.Dataset, observed: xr.DataArray, primary_obs_dim: str) -> None: ...
def _align_data_to_obs(data: xr.DataTree, observed: xr.DataArray) -> None: ...
def _generate_subsample_indices(
    n_data_points: int, observations: int | np.ndarray, seed: int | None
) -> None: ...
def _get_r_eff(data: xr.DataTree, n_samples: int) -> None: ...
def _get_r_eff_i(r_eff: xr.DataArray, i: Incomplete, obs_dims: list) -> None: ...
def _prepare_full_arrays(
    pointwise_values: xr.DataArray,
    pareto_k_values: xr.DataArray,
    ref_array: xr.DataArray,
    indices: np.ndarray,
    obs_dims: list,
    elpd_loo_hat: float | None = ...,
) -> None: ...
def _select_obs_by_indices(
    data_array: xr.DataArray, indices: np.ndarray, dims: list, dim_name: str
) -> None: ...
def _select_obs_by_coords(
    data_array: xr.DataArray, coord_array: xr.DataArray, dims: list, dim_name: str
) -> None: ...
def _warn_pareto_k(pareto_k_values: xr.DataArray, n_samples: int, suppress: bool = ...) -> None: ...
def _warn_pointwise_loo(elpd: float, elpd_i_values: np.ndarray) -> None: ...
def _check_log_density(
    log_dens: np.ndarray | xr.DataArray,
    name: str,
    log_likelihood: xr.DataArray,
    n_samples: int,
    sample_dims: list,
) -> None: ...
def _check_log_jacobian(log_jacobian: xr.DataArray | None, obs_dims: list) -> None: ...
def _get_upars_info(upars: xr.DataArray, param_dim: str) -> None: ...
def _reconstruct_upars(upars_new_values: np.ndarray, props: dict) -> None: ...
def _has_nan_slice(da: xr.DataArray, dim: str) -> None: ...
def _validate_crps_input(
    y_pred: xr.DataArray,
    y_obs: xr.DataArray,
    log_likelihood: xr.DataArray,
    *,
    sample_dims: list,
    obs_dims: list,
) -> None: ...
def _validate_sample_dims(
    data: xr.DataArray,
    *,
    sample_dims: list | None = ...,
    ref_sizes: Incomplete = ...,
    obs_dims: list | None = ...,
) -> None: ...
def _get_sample_coords(sample_dims: list, data_for_fn: xr.DataTree) -> None: ...
def _validate_log_lik_fn_result(
    log_likelihood: xr.DataArray,
    sample_dims: list,
    obs_dims: list,
    observed: xr.DataArray,
) -> None: ...
def _var_name_custom_ll(data: xr.DataTree, var_name: str | None) -> None: ...
