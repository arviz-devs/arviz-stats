# File generated with docstub

import logging
from collections.abc import Hashable, Sequence
from typing import Literal

import numpy as np
import pandas
import xarray
import xarray as xr
from _typeshed import Incomplete
from arviz_base import convert_to_datatree, dataset_to_dataframe, extract
from arviz_base.labels import BaseLabeller

from arviz_stats.utils import get_log_likelihood_dataset, get_log_prior
from arviz_stats.validate import validate_dims

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
