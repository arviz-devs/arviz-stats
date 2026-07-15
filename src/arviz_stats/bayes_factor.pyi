# File generated with docstub

import xarray
import xarray as xr
from _typeshed import Incomplete

def bayes_factor(
    data: xarray.DataTree,
    var_names: str | list[str],
    ref_vals: float | list[float] = ...,
    return_ref_vals: bool = ...,
    prior: dict | None = ...,
    circular: bool = ...,
) -> xr.Dataset: ...
def _eval_kde_at_ref(kde_da: Incomplete, ref_val: Incomplete) -> None: ...
