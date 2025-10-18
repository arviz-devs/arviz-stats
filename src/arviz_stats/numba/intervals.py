"""Credible interval estimators."""

import warnings

import numpy as np
import xarray as xr
from arviz_base import rcParams
from numba import guvectorize
from xarray_einstats import _remove_indexes_to_reduce

__all__ = ["eti", "quantile"]


@guvectorize(
    ["void(float64[:], float64[:], float64[:])", "void(int64[:], float64[:], float64[:])"],
    "(n),(q)->(q)",
    cache=True,
    target="parallel",
)
def _quantile(ary, q, result):  # pylint: disable=unused-argument
    result[:] = np.quantile(ary, q)


def quantile(da, q, dim):
    """Numbified and xarray aware quantile function."""
    if not isinstance(dim, str):
        aux_dim = f"__aux_dim__:{','.join(dim)}"
        da = _remove_indexes_to_reduce(da, dim).stack({aux_dim: dim}, create_index=False)
    else:
        aux_dim = dim
    return xr.apply_ufunc(
        _quantile, da, q, input_core_dims=[[aux_dim], ["quantile"]], output_core_dims=[["quantile"]]
    )


def eti(da, prob=None, dims=None, method=None, skipna=None):
    """Compute the equal tail credible interval."""
    if prob is None:
        prob = rcParams["stats.ci_prob"]
    elif not 1 >= prob > 0:
        raise ValueError("The value of hdi_prob should be in the interval (0, 1]")
    if dims is None:
        dims = rcParams["data.sample_dims"]
    if method is not None:
        warnings.warn("method argument is not available with numba and will be ignored.")
    if skipna is not None:
        warnings.warn("skipna argument is not available with numba and will be ignored.")
    q_low = (1 - prob) / 2
    q_high = 1 - q_low

    return quantile(da, (q_low, q_high), dim=dims)
