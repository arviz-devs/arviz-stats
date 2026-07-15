# File generated with docstub

from collections.abc import Hashable, Sequence

import xarray
from xarray import DataArray, Dataset

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
