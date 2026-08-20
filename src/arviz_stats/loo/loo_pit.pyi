# File generated with docstub

from collections.abc import Generator

import xarray
import xarray as xr
from arviz_base import convert_to_datatree, extract
from numpy.typing import NDArray
from xarray import DataArray, Dataset

from arviz_stats.loo.loo_helper import _get_r_eff
from arviz_stats.utils import get_log_likelihood_dataset

def loo_pit(
    data: xarray.DataTree | xarray.DataTree,
    var_names: str | list[str] | None = ...,
    log_weights: Dataset | None = ...,
    pareto_k: Dataset | None = ...,
    random_state: int | Generator | None = ...,
    pareto_pit: bool = ...,
) -> NDArray | DataArray: ...
