# File generated with docstub

from collections.abc import Hashable, Iterable
from typing import Any, Literal

import xarray
from numpy.typing import ArrayLike, NDArray

def ess(
    data: ArrayLike,
    sample_dims: Iterable[Hashable] | None = ...,
    group: Hashable = ...,
    var_names: str | list[str] | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    coords: dict | None = ...,
    method: str = ...,
    relative: bool = ...,
    prob: float | None = ...,
    chain_axis: int = ...,
    draw_axis: int = ...,
) -> NDArray: ...
def rhat(
    data: ArrayLike,
    sample_dims: Iterable[Hashable] | None = ...,
    group: Hashable = ...,
    var_names: str | list[str] | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    coords: dict | None = ...,
    method: Literal["rank", "identity", "split", "z_scale", "folded"] = ...,
    chain_axis: int = ...,
    draw_axis: int = ...,
) -> NDArray: ...
def rhat_nested(
    data: ArrayLike,
    sample_dims: Iterable[Hashable] | None = ...,
    group: Hashable = ...,
    var_names: str | list[str] | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    method: str = ...,
    coords: dict | None = ...,
    superchain_ids: list | None = ...,
    chain_axis: int = ...,
    draw_axis: int = ...,
) -> None: ...
def mcse(
    data: ArrayLike,
    sample_dims: Iterable[Hashable] | None = ...,
    group: Hashable = ...,
    var_names: str | list[str] | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    coords: dict | None = ...,
    method: str = ...,
    prob: float | None = ...,
    circular: bool = ...,
    chain_axis: int = ...,
    draw_axis: int = ...,
) -> NDArray: ...
def bfmi(
    data: ArrayLike,
    sample_dims: Iterable[Hashable] | None = ...,
    group: Hashable = ...,
    var_names: str | list[str] = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    coords: dict | None = ...,
    **kwargs: Any,
) -> NDArray: ...
def diagnose(
    data: xarray.DataTree,
    *,
    var_names: str | list[str] | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    coords: dict | None = ...,
    sample_dims: Iterable[Hashable] | None = ...,
    group: str = ...,
    rhat_max: float = ...,
    ess_min_ratio: float = ...,
    ess_threshold: int | None = ...,
    bfmi_threshold: float = ...,
    show_diagnostics: bool = ...,
    return_diagnostics: bool = ...,
) -> tuple[bool, dict]: ...
