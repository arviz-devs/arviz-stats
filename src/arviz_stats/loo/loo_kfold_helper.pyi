# File generated with docstub

from collections import namedtuple

import numpy as np
import xarray as xr
from _typeshed import Incomplete
from arviz_base import convert_to_datatree
from numpy.typing import ArrayLike
from xarray_einstats.stats import logsumexp

from arviz_stats.loo.wrapper import SamplingWrapper
from arviz_stats.utils import get_log_likelihood

__all__ = [
    "_prepare_kfold_inputs",
    "_compute_kfold_results",
    "_kfold_split_random",
    "_kfold_split_stratified",
    "_kfold_split_grouped",
    "_extract_fold_data",
    "_get_fold_indices",
    "_combine_fold_elpds",
    "_validate_k_value",
    "_validate_array_length",
    "_validate_fold_parameters",
]

FoldData: type

KfoldResults: type

KfoldInputs: type

def _prepare_kfold_inputs(
    data: Incomplete,
    var_name: str | None,
    wrapper: SamplingWrapper,
    k: int,
    folds: ArrayLike | None,
    stratify_by: ArrayLike | None,
    group_by: ArrayLike | None,
    seed: int | None = ...,
) -> None: ...
def _compute_kfold_results(
    kfold_inputs: KfoldInputs, wrapper: SamplingWrapper, save_fits: bool
) -> None: ...
def _kfold_split_random(
    k: int = ..., n: int | None = ..., rng: np.random.Generator | None = ...
) -> None: ...
def _kfold_split_stratified(
    k: int = ..., x: ArrayLike | None = ..., rng: np.random.Generator | None = ...
) -> None: ...
def _kfold_split_grouped(
    k: int = ..., x: ArrayLike | None = ..., rng: np.random.Generator | None = ...
) -> None: ...
def _extract_fold_data(data: xr.DataArray, fold_indices: ArrayLike, train: bool = ...) -> None: ...
def _get_fold_indices(fold_assignments: ArrayLike, k: int) -> None: ...
def _combine_fold_elpds(fold_elpds: list, n_data_points: int) -> None: ...
def _validate_k_value(k: int, n: int, param_name: str = ...) -> None: ...
def _validate_array_length(array: ArrayLike, expected_length: int, param_name: str) -> None: ...
def _validate_fold_parameters(
    folds: ArrayLike | None, stratify_by: ArrayLike | None, group_by: ArrayLike | None
) -> None: ...
