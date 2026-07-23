# File generated with docstub

from collections.abc import Hashable, Sequence
from typing import Any, Literal

import numpy as np
import xarray
import xarray as xr
from _typeshed import Incomplete
from arviz_base import convert_to_dataset
from numpy.typing import ArrayLike, NDArray
from xarray import DataArray, Dataset

from arviz_stats.utils import _apply_multi_input_function, get_function
from arviz_stats.validate import validate_ci_prob, validate_dims

def hdi(
    data: ArrayLike,
    prob: float | None = ...,
    dim: Sequence[Hashable] | None = ...,
    group: Hashable = ...,
    var_names: str | list[str] | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    coords: dict | None = ...,
    method: str = ...,
    circular: bool = ...,
    max_modes: int = ...,
    skipna: bool = ...,
    **kwargs: Any,
) -> NDArray: ...
def eti(
    data: ArrayLike,
    prob: float | None = ...,
    dim: Sequence[Hashable] | None = ...,
    group: Hashable = ...,
    var_names: str | list[str] | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    coords: dict | None = ...,
    method: str = ...,
    skipna: bool = ...,
    **kwargs: Any,
) -> NDArray: ...
def ecdf(
    data: ArrayLike,
    dim: Sequence[Hashable] | None = ...,
    group: Hashable = ...,
    var_names: str | list[str] | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    coords: dict | None = ...,
    pit: bool = ...,
    **kwargs: Any,
) -> NDArray: ...
def histogram(
    data: ArrayLike,
    dim: Sequence[Hashable] | None = ...,
    group: Hashable = ...,
    var_names: str | list[str] | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    coords: dict | None = ...,
    bins: Incomplete = ...,
    range: ArrayLike | None = ...,
    weights: ArrayLike | None = ...,
    density: bool = ...,
) -> NDArray: ...
def kde(
    data: ArrayLike,
    dim: Sequence[Hashable] | None = ...,
    group: Hashable = ...,
    var_names: str | list[str] | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    coords: dict | None = ...,
    circular: bool = ...,
    **kwargs: Any,
) -> NDArray: ...
def qds(
    data: ArrayLike,
    dim: Sequence[Hashable] | None = ...,
    group: Hashable = ...,
    var_names: str | list[str] | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    coords: dict | None = ...,
    nquantiles: int = ...,
    binwidth: float | None = ...,
    dotsize: float = ...,
    stackratio: float = ...,
    top_only: bool = ...,
    **kwargs: Any,
) -> NDArray: ...
def kde2d(
    da_x: ArrayLike | DataArray | xarray.DataTree | xarray.DataTree,
    da_y: ArrayLike | DataArray | xarray.DataTree | xarray.DataTree,
    dim: str | Sequence[Hashable] | None = ...,
    group: Hashable = ...,
    gridsize: tuple[int] = ...,
    circular: bool = ...,
    hdi_probs: list[float] | None = ...,
) -> tuple | Dataset: ...
