# File generated with docstub

from collections.abc import Hashable, Sequence
from typing import Literal

import pandas
import xarray
from _typeshed import Incomplete

_log: Incomplete

labeller: Incomplete

__all__ = ["psense", "psense_summary"]

def psense(
    data: xarray.DataTree | xarray.DataTree,
    var_names: list[str] | None = ...,
    filter_vars: Incomplete | None = ...,
    group: Literal["prior", "likelihood"] = ...,
    coords: dict | None = ...,
    sample_dims: str | Sequence[Hashable] | None = ...,
    alphas: tuple = ...,
    group_var_names: str | None = ...,
    group_coords: dict | None = ...,
) -> xarray.DataTree: ...
def psense_summary(
    data: xarray.DataTree | xarray.DataTree,
    var_names: list[str] | None = ...,
    filter_vars: Incomplete | None = ...,
    coords: dict | None = ...,
    sample_dims: str | Sequence[Hashable] | None = ...,
    threshold: float = ...,
    alphas: tuple = ...,
    prior_var_names: str | None = ...,
    likelihood_var_names: str | None = ...,
    prior_coords: dict | None = ...,
    likelihood_coords: dict | None = ...,
    round_to: int = ...,
) -> pandas.DataFrame: ...
def power_scale_dataset(
    data: xarray.DataTree | xarray.DataTree,
    group: str,
    alphas: tuple[float],
    sample_dims: str | Sequence[Hashable],
    group_var_names: str,
    group_coords: dict,
) -> Incomplete: ...
def _get_power_scale_weights(
    dt: Incomplete,
    alphas: Incomplete = ...,
    group: Incomplete = ...,
    sample_dims: Incomplete = ...,
    group_var_names: Incomplete = ...,
    group_coords: Incomplete = ...,
) -> None: ...
