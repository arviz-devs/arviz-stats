# File generated with docstub

import warnings

import numpy as np
import xarray
from arviz_base import dict_to_dataset, extract
from xarray import Dataset

def kaplan_meier(
    dt: xarray.DataTree,
    var_names: str | list[str],
) -> Dataset: ...
def generate_survival_curves(
    dt: xarray.DataTree,
    var_names: str | list[str],
    group: str = ...,
    num_samples: int | None = ...,
    extrapolation_factor: float = ...,
) -> Dataset: ...
