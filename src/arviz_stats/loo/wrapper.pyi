# File generated with docstub

from collections.abc import Callable

import xarray
from _typeshed import Incomplete
from _typeshed import Incomplete as fitted_model
from _typeshed import Incomplete as modified_observed_data
from _typeshed import Incomplete as optional
from arviz_base import convert_to_datatree
from xarray import apply_ufunc

class SamplingWrapper:
    def __init__(
        self,
        model: Incomplete,
        idata_orig: optional | None = ...,
        log_lik_fun: Callable | None = ...,
        is_ufunc: bool = ...,
        posterior_vars: list[str] | None = ...,
        sample_kwargs: dict | None = ...,
        idata_kwargs: dict | None = ...,
        log_lik_kwargs: dict | None = ...,
        apply_ufunc_kwargs: dict | None = ...,
    ) -> None: ...
    def sel_observations(self, idx: Incomplete) -> modified_observed_data: ...
    def sample(self, modified_observed_data: Incomplete) -> fitted_model: ...
    def get_inference_data(self, fitted_model: Incomplete) -> Incomplete: ...
    def log_likelihood__i(
        self, excluded_obs: Incomplete, idata__i: xarray.DataTree | xarray.DataTree
    ) -> Incomplete: ...
    def check_implemented_methods(self, methods: list) -> Incomplete: ...
