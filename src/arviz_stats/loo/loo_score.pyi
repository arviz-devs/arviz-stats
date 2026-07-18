# File generated with docstub

from collections import namedtuple
from typing import NamedTuple

import numpy as np
import xarray
from arviz_base import convert_to_datatree, extract
from xarray import DataArray

from arviz_stats.base.stats_utils import round_num
from arviz_stats.loo.helper_loo import (
    _get_r_eff,
    _prepare_loo_inputs,
    _validate_crps_input,
    _warn_pareto_k,
)

def loo_score(
    data: xarray.DataTree | xarray.DataTree,
    var_name: str | None = ...,
    kind: str = ...,
    pointwise: bool = ...,
    round_to: int | str | None = ...,
    log_weights: DataArray | None = ...,
    pareto_k: DataArray | None = ...,
) -> NamedTuple: ...
