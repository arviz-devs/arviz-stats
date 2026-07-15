# File generated with docstub

from collections.abc import Hashable, Iterable
from typing import NamedTuple

import xarray
from _typeshed import Incomplete
from numpy.typing import NDArray
from xarray import DataArray

def bayesian_r2(
    data: xarray.DataTree | xarray.DataTree,
    pred_mean: str | None = ...,
    scale: str | None = ...,
    scale_kind: str = ...,
    summary: bool = ...,
    group: str = ...,
    point_estimate: str | None = ...,
    ci_kind: str | None = ...,
    ci_prob: float | None = ...,
    circular: bool = ...,
    round_to: int | str | None | None = ...,
) -> NamedTuple | NDArray: ...
def residual_r2(
    data: xarray.DataTree | xarray.DataTree,
    pred_mean: str | None = ...,
    obs_name: str | None = ...,
    summary: bool = ...,
    group: str = ...,
    point_estimate: str | None = ...,
    ci_kind: str | None = ...,
    ci_prob: float | None = ...,
    circular: bool = ...,
    round_to: int | str | None | None = ...,
) -> NamedTuple | NDArray: ...
def metrics(
    data: xarray.DataTree | xarray.DataTree,
    kind: str = ...,
    var_name: str | None = ...,
    sample_dims: Iterable[Hashable] | None = ...,
    round_to: int | str | None | None = ...,
) -> Incomplete: ...
def kl_divergence(
    data1: DataArray,
    data2: DataArray,
    group: Hashable = ...,
    var_names: str | list[str] | None = ...,
    sample_dims: Iterable[Hashable] | None = ...,
    num_samples: int = ...,
    round_to: int | str | None | None = ...,
    random_seed: int = ...,
) -> float: ...
def wasserstein(
    data1: DataArray,
    data2: DataArray,
    group: Hashable = ...,
    var_names: str | list[str] | None = ...,
    sample_dims: Iterable[Hashable] | None = ...,
    joint: bool = ...,
    num_samples: int = ...,
    round_to: int | str | None | None = ...,
    random_seed: int = ...,
) -> float: ...
def _prepare_distribution_pair(
    data1: Incomplete,
    data2: Incomplete,
    group: Incomplete,
    var_names: Incomplete,
    sample_dims: Incomplete,
    num_samples: Incomplete,
    random_seed: Incomplete,
) -> None: ...
def _extract_and_reindex(
    data: Incomplete,
    group: Incomplete,
    var_names: Incomplete,
    sample_dims: Incomplete,
    num_samples: Incomplete,
    random_seed: Incomplete,
) -> None: ...
def _kld(ary0: Incomplete, ary1: Incomplete) -> float: ...
def _metrics(
    observed: DataArray, predicted: DataArray, kind: str, round_to: int | str | None
) -> Incomplete: ...
def _summary_r2(
    name: Incomplete,
    r_squared: Incomplete,
    point_estimate: Incomplete,
    ci_kind: Incomplete,
    ci_prob: Incomplete,
    round_to: Incomplete,
) -> None: ...
