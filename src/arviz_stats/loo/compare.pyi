# File generated with docstub

import warnings
from copy import deepcopy
from typing import Literal

import numpy as np
import pandas
import pandas as pd
import xarray
from _typeshed import Incomplete
from arviz_base import rcParams
from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.stats import dirichlet, norm

from arviz_stats.base.stats_utils import get_decimal_places_from_se, round_num
from arviz_stats.loo import loo
from arviz_stats.loo.helper_loo import _diff_srs_estimator
from arviz_stats.utils import ELPDData

def compare(
    compare_dict: dict[str, xarray.DataTree | ELPDData],
    method: str = ...,
    var_name: str | None = ...,
    reference: str | None = ...,
    round_to: int | Literal["auto", "none"] = ...,
) -> pandas.DataFrame: ...
def _compute_elpd_diff_subsampled(elpd_a: Incomplete, elpd_b: Incomplete) -> None: ...
def _difference_estimator(
    elpd_a: Incomplete,
    elpd_b: Incomplete,
    shared_indices: Incomplete,
    subsample_a: Incomplete = ...,
    subsample_b: Incomplete = ...,
) -> None: ...
def _compute_naive_diff(elpd_a: Incomplete, elpd_b: Incomplete) -> None: ...
def _ic_matrix(ics: Incomplete) -> None: ...
def _calculate_ics(
    compare_dict: dict[str, xarray.DataTree | ELPDData],
    var_name: str | None = ...,
) -> dict[ELPDData]: ...
def _order_stat_check(
    ics_dict: Incomplete, model_order: Incomplete, has_subsampling: Incomplete
) -> None: ...
def _round_compare(
    result: pandas.DataFrame, round_val: int | str, prob_direction: str
) -> pandas.DataFrame: ...
