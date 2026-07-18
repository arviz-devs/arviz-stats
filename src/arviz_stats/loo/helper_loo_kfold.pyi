# File generated with docstub

from collections import namedtuple

import numpy as np
import xarray as xr
from _typeshed import Incomplete
from arviz_base import convert_to_datatree
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

FoldData: Incomplete

KfoldResults: Incomplete

KfoldInputs: Incomplete

def _prepare_kfold_inputs(
    data: Incomplete,
    var_name: Incomplete,
    wrapper: Incomplete,
    k: Incomplete,
    folds: Incomplete,
    stratify_by: Incomplete,
    group_by: Incomplete,
) -> None: ...
def _compute_kfold_results(
    kfold_inputs: Incomplete, wrapper: Incomplete, save_fits: Incomplete
) -> None: ...
def _kfold_split_random(k: Incomplete = ..., n: Incomplete = ...) -> None: ...
def _kfold_split_stratified(k: Incomplete = ..., x: Incomplete = ...) -> None: ...
def _kfold_split_grouped(k: Incomplete = ..., x: Incomplete = ...) -> None: ...
def _extract_fold_data(
    data: Incomplete, fold_indices: Incomplete, train: Incomplete = ...
) -> None: ...
def _get_fold_indices(fold_assignments: Incomplete, k: Incomplete) -> None: ...
def _combine_fold_elpds(fold_elpds: Incomplete, n_data_points: Incomplete) -> None: ...
def _validate_k_value(k: Incomplete, n: Incomplete, param_name: Incomplete = ...) -> None: ...
def _validate_array_length(
    array: Incomplete, expected_length: Incomplete, param_name: Incomplete
) -> None: ...
def _validate_fold_parameters(
    folds: Incomplete, stratify_by: Incomplete, group_by: Incomplete
) -> None: ...
