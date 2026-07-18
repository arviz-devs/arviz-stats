# File generated with docstub

from collections.abc import Hashable, Iterable
from typing import Literal

import numpy as np
import xarray
import xarray as xr
from _typeshed import Incomplete
from arviz_base import convert_to_dataset, convert_to_datatree, extract, rcParams
from numpy.typing import ArrayLike, NDArray

from arviz_stats.utils import get_array_function
from arviz_stats.validate import validate_dims

def thin(
    data: ArrayLike,
    sample_dims: Iterable[Hashable] = ...,
    group: Hashable = ...,
    var_names: str | list[str] | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    coords: dict | None = ...,
    factor: str | int = ...,
    chain_axis: int = ...,
    draw_axis: int = ...,
) -> NDArray: ...
def weight_predictions(
    dts: list[xarray.DataTree] | list[xarray.DataTree],
    weights: ArrayLike | None = ...,
    group: str = ...,
    sample_dims: Iterable[Hashable] | None = ...,
    random_seed: int | None = ...,
) -> Incomplete: ...
