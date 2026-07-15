# File generated with docstub

from typing import Literal

import xarray
from numpy.typing import ArrayLike
from xarray import Dataset

def isotonic_fit(
    dt: xarray.DataTree,
    var_names: str | list[str],
    group: str,
    ci_prob: float,
    data_type: str = ...,
    residuals: bool = ...,
    x_var: ArrayLike | None = ...,
) -> None: ...
def _isotonic_fit(
    pred: ArrayLike,
    obs: ArrayLike,
    ci_prob: float,
    residuals: bool,
    preds: ArrayLike,
    x_var: ArrayLike,
) -> None: ...
def point_interval_unique(
    ds: Dataset, ci_prob: float, point_estimate: Literal["mean", "median", "mode"]
) -> Dataset: ...
def point_unique(ds: Dataset) -> Dataset: ...
