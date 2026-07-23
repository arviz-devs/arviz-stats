# File generated with docstub

from collections.abc import Hashable, Sequence

import numpy as np
import xarray
from arviz_base import extract
from xarray import DataArray, Dataset

from arviz_stats.loo import loo_expectations
from arviz_stats.summary import mad, mean, median, std, var
from arviz_stats.validate import validate_dims

def loo_influence(
    data: xarray.DataTree | xarray.DataTree,
    var_names: str | list[str] | None = ...,
    group: str = ...,
    sample_dims: str | Sequence[Hashable] | None = ...,
    log_likelihood_var_name: str | None = ...,
    kind: str = ...,
    standardize: bool = ...,
    probs: float | list[float] | None = ...,
    log_weights: DataArray | None = ...,
    pareto_k: DataArray | None = ...,
) -> tuple[DataArray | Dataset, DataArray | Dataset]: ...
