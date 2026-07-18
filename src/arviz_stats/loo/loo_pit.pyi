# File generated with docstub

from collections.abc import Generator

import xarray
import xarray as xr
from _typeshed import Incomplete
from arviz_base import convert_to_datatree, extract
from xarray import Dataset

from arviz_stats.loo.helper_loo import _get_r_eff
from arviz_stats.utils import get_log_likelihood_dataset

def loo_pit(
    data: xarray.DataTree | xarray.DataTree,
    var_names: str | list[str] | None = ...,
    log_weights: Dataset | None = ...,
    pareto_k: Dataset | None = ...,
    random_state: int | Generator | None = ...,
    pareto_pit: Incomplete = ...,
) -> Incomplete: ...
