# File generated with docstub

from collections.abc import Callable

import xarray
from arviz_base import convert_to_datatree
from numpy.typing import ArrayLike
from xarray import DataArray, apply_ufunc

class SamplingWrapper:
    def __init__(
        self,
        model: object,
        idata_orig: xarray.DataTree | xarray.DataTree | None = ...,
        log_lik_fun: Callable | None = ...,
        is_ufunc: bool = ...,
        posterior_vars: list[str] | None = ...,
        sample_kwargs: dict | None = ...,
        idata_kwargs: dict | None = ...,
        log_lik_kwargs: dict | None = ...,
        apply_ufunc_kwargs: dict | None = ...,
    ) -> None: ...
    def sel_observations(self, idx: int | ArrayLike) -> tuple[object, object]: ...
    def sample(self, modified_observed_data: object) -> object: ...
    def get_inference_data(self, fitted_model: object) -> xarray.DataTree | xarray.DataTree: ...
    def log_likelihood__i(
        self, excluded_obs: object, idata__i: xarray.DataTree | xarray.DataTree
    ) -> DataArray: ...
    def check_implemented_methods(self, methods: list) -> list: ...
