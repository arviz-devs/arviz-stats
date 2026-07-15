# File generated with docstub

from collections.abc import Hashable, Sequence
from typing import Any, Literal

import pandas
import xarray
from _typeshed import Incomplete
from numpy.typing import ArrayLike, NDArray
from xarray import Dataset

__all__ = [
    "summary",
    "ci_in_rope",
    "mean",
    "median",
    "mode",
    "std",
    "var",
    "iqr",
    "mad",
]

def summary(
    data: xarray.DataTree,
    var_names: list[str] | None = ...,
    filter_vars: Incomplete | None = ...,
    group: str = ...,
    coords: dict | None = ...,
    sample_dims: str | Sequence[Hashable] | None = ...,
    kind: Incomplete = ...,
    fmt: Incomplete = ...,
    ci_prob: float | None = ...,
    ci_kind: Literal["hdi", "eti"] | None = ...,
    round_to: int | Literal["auto", "none"] = ...,
    skipna: bool = ...,
) -> pandas.DataFrame | Dataset: ...
def _round_summary(summary_result: pandas.DataFrame, round_val: int | str) -> pandas.DataFrame: ...
def ci_in_rope(
    data: xarray.DataTree,
    rope: Incomplete,
    var_names: list[str] | None = ...,
    filter_vars: Incomplete | None = ...,
    group: str = ...,
    dim: str | Sequence[Hashable] | None = ...,
    ci_prob: float | None = ...,
    ci_kind: Literal["hdi", "eti"] | None = ...,
    rope_dim: str = ...,
) -> Dataset: ...
def mean(
    data: ArrayLike,
    dim: Sequence[Hashable] | None = ...,
    group: Hashable = ...,
    var_names: str | list[str] | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    coords: dict | None = ...,
    round_to: int | str | None | None = ...,
    skipna: bool = ...,
    **kwargs: Any,
) -> NDArray: ...
def median(
    data: ArrayLike,
    dim: Sequence[Hashable] | None = ...,
    group: Hashable = ...,
    var_names: str | list[str] | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    coords: dict | None = ...,
    round_to: int | str | None | None = ...,
    skipna: bool = ...,
    **kwargs: Any,
) -> NDArray: ...
def mode(
    data: ArrayLike,
    dim: Sequence[Hashable] | None = ...,
    group: Hashable = ...,
    var_names: str | list[str] | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    coords: dict | None = ...,
    round_to: int | str | None | None = ...,
    skipna: bool = ...,
    **kwargs: Any,
) -> NDArray: ...
def std(
    data: ArrayLike,
    dim: Sequence[Hashable] | None = ...,
    group: Hashable = ...,
    var_names: str | list[str] | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    coords: dict | None = ...,
    round_to: int | str | None | None = ...,
    skipna: bool = ...,
    **kwargs: Any,
) -> NDArray: ...
def var(
    data: ArrayLike,
    dim: Sequence[Hashable] | None = ...,
    group: Hashable = ...,
    var_names: str | list[str] | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    coords: dict | None = ...,
    round_to: int | str | None | None = ...,
    skipna: bool = ...,
    **kwargs: Any,
) -> NDArray: ...
def mad(
    data: ArrayLike,
    dim: Sequence[Hashable] | None = ...,
    group: Hashable = ...,
    var_names: str | list[str] | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    coords: dict | None = ...,
    round_to: int | str | None | None = ...,
    skipna: bool = ...,
    **kwargs: Any,
) -> NDArray: ...
def iqr(
    data: ArrayLike,
    dim: Sequence[Hashable] | None = ...,
    group: Hashable = ...,
    var_names: str | list[str] | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    coords: dict | None = ...,
    quantiles: tuple[float] = ...,
    round_to: int | str | None | None = ...,
    skipna: bool = ...,
    **kwargs: Any,
) -> NDArray: ...
