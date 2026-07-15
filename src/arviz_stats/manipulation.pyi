# File generated with docstub

from collections.abc import Hashable, Iterable
from typing import Literal

import xarray
from _typeshed import Incomplete
from numpy.typing import ArrayLike, NDArray

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
