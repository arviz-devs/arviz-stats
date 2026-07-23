# File generated with docstub

from collections.abc import Hashable, Sequence
from typing import NamedTuple

import numpy as np
import xarray
import xarray as xr
from arviz_base import convert_to_datatree, extract, rcParams
from numpy.typing import NDArray
from xarray import DataArray, Dataset, apply_ufunc

from arviz_stats.loo.helper_loo import _get_r_eff, _warn_pareto_k
from arviz_stats.metrics import _metrics, _summary_r2
from arviz_stats.utils import get_log_likelihood_dataset
from arviz_stats.validate import validate_dims

def loo_expectations(
    data: xarray.DataTree | xarray.DataTree,
    var_name: str | None = ...,
    group: str = ...,
    sample_dims: str | Sequence[Hashable] | None = ...,
    log_likelihood_var_name: str | None = ...,
    kind: str = ...,
    probs: float | list[float] | None = ...,
    log_weights: DataArray | None = ...,
    pareto_k: DataArray | None = ...,
) -> tuple[DataArray | Dataset, DataArray | Dataset]: ...
def loo_metrics(
    data: xarray.DataTree | xarray.DataTree,
    kind: str = ...,
    var_name: str | None = ...,
    round_to: int | str | None | None = ...,
) -> NamedTuple: ...
def loo_r2(
    data: xarray.DataTree | xarray.DataTree,
    var_name: str,
    n_simulations: int = ...,
    summary: bool = ...,
    point_estimate: str | None = ...,
    ci_kind: str | None = ...,
    ci_prob: float | None = ...,
    circular: bool = ...,
    round_to: int | str | None | None = ...,
) -> NamedTuple | NDArray: ...
def _get_function_khat(
    values: NDArray,
    log_weights: NDArray,
    kind: str,
) -> float: ...
