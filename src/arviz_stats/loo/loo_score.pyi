# File generated with docstub

from typing import NamedTuple

import xarray
from xarray import DataArray

def loo_score(
    data: xarray.DataTree | xarray.DataTree,
    var_name: str | None = ...,
    kind: str = ...,
    pointwise: bool = ...,
    round_to: int | str | None = ...,
    log_weights: DataArray | None = ...,
    pareto_k: DataArray | None = ...,
) -> NamedTuple: ...
