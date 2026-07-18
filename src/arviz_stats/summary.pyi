# File generated with docstub

from collections.abc import Hashable, Sequence
from typing import Any, Literal

import numpy as np
import pandas
import pandas as pd
import xarray
import xarray as xr
from _typeshed import Incomplete
from arviz_base import dataset_to_dataframe, extract, rcParams, references_to_dataset
from numpy.typing import ArrayLike, NDArray
from xarray import Dataset
from xarray_einstats import stats

from arviz_stats.base.stats_utils import get_decimal_places_from_se, round_num
from arviz_stats.utils import _apply_multi_input_function
from arviz_stats.validate import validate_dims

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
) -> SummaryDataFrame: ...
def _build_fmt_map(
    summary_result: pandas.DataFrame, round_val: int | str
) -> dict[str, dict[Any, str]]: ...

class SummaryDataFrame(pd.DataFrame):
    _metadata: Incomplete

    def __init__(
        self, *args: Incomplete, fmt_map: Incomplete = ..., **kwargs: Incomplete
    ) -> None: ...
    @property
    def _constructor(self) -> None: ...
    @property
    def T(self) -> None: ...
    def transpose(self, *args: Incomplete, **kwargs: Incomplete) -> None: ...
    def _display_df(self) -> None: ...
    def _repr_html_(self) -> None: ...
    def _repr_latex_(self) -> None: ...
    def to_html(self, *args: Incomplete, **kwargs: Incomplete) -> None: ...
    def to_latex(self, *args: Incomplete, **kwargs: Incomplete) -> None: ...
    def __repr__(self) -> None: ...
    def __str__(self) -> None: ...
    def to_string(self, *args: Incomplete, **kwargs: Incomplete) -> None: ...

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
