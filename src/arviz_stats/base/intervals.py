"""Credible interval estimators."""
import warnings

import numpy as np
import scipy.stats as st
import xarray as xr
from arviz_base import rcParams

from .density import _kde
from .density import get_bins as _get_bins
from .stats_utils import wrap_xarray_ufunc as _wrap_xarray_ufunc

__all__ = ["eti", "hdi"]


def eti(
    da,
    prob=None,
    dims=None,
    method="linear",
    skipna=False,
):
    """Compute equal tail credible interval.

    Internal method for DataArrays.

    This method is then used by the multiple accessors to expose this
    function to DataArrays, Datasets and DataTrees as accessor methods.

    See Also
    --------
    AzStatsDaAccessor, AzStatsDsAccessor, AzStatsDtAccessor
    """
    if prob is None:
        prob = rcParams["stats.ci_prob"]
    elif not 1 >= prob > 0:
        raise ValueError("The value of hdi_prob should be in the interval (0, 1]")
    if dims is None:
        dims = rcParams["data.sample_dims"]
    q_low = (1 - prob) / 2
    q_high = 1 - q_low

    return da.quantile((q_low, q_high), dim=dims, method=method, skipna=skipna)


def hdi(
    da,
    prob=None,
    dims=None,
    method="nearest",
    circular=False,
    multimodal=False,
    skipna=False,
    max_modes=10,
    **dask_kwargs,
):
    """Compute highest density interval on a DataArray."""
    if prob is None:
        prob = rcParams["stats.ci_prob"]
    elif not 1 >= prob > 0:
        raise ValueError("The value of hdi_prob should be in the interval (0, 1]")
    if dims is None:
        dims = rcParams["data.sample_dims"]
    if isinstance(dims, str):
        dims = [dims]

    func_kwargs = {
        "prob": prob,
        "skipna": skipna,
        "out_shape": (max_modes, 2) if multimodal else (2,),
    }
    mode_dim = "mode" if da.name is None else f"{da.name}_mode"
    dask_kwargs.setdefault("output_core_dims", [[mode_dim, "hdi"] if multimodal else ["hdi"]])
    if not multimodal:
        func_kwargs["circular"] = circular
    else:
        func_kwargs["max_modes"] = max_modes

    hdi_method_map = {"nearest": _hdi_nearest, "agg": _hdi_agg_nearest}

    func = _hdi_multimodal if multimodal else hdi_method_map[method]

    hdi_coord = xr.DataArray(["lower", "higher"], dims=["hdi"], attrs={"prob": prob})
    hdi_data = _wrap_xarray_ufunc(
        func, da, input_core_dims=[dims], func_kwargs=func_kwargs, **dask_kwargs
    ).assign_coords({"hdi": hdi_coord})
    hdi_data = hdi_data.dropna(mode_dim, how="all") if multimodal else hdi_data
    return hdi_data


def _hdi_agg_nearest(ary, prob, skipna):
    """Approximate the HDI from the kde or histogram."""
    ary = ary.flatten()
    if skipna:
        nans = np.isnan(ary)
        if not nans.all():
            ary = ary[~nans]

    if ary.dtype.kind == "f":
        bins, density, _ = _kde(ary)
    else:
        bins = _get_bins(ary)
        density, _ = np.histogram(ary, bins=bins, density=True)

    sorted_idx = np.argsort(density)[::-1]
    mass_cum = 0
    indices = []
    for idx in sorted_idx:
        mass_cum += density[idx]
        indices.append(idx)
        if mass_cum >= prob:
            break
    return bins[np.sort(indices)[[0, -1]]]


def _hdi_linear_nearest_common(ary, prob, skipna, circular):
    ary = ary.flatten()
    if skipna:
        nans = np.isnan(ary)
        if not nans.all():
            ary = ary[~nans]
    n = len(ary)

    mean = None
    if circular:
        mean = st.circmean(ary, high=np.pi, low=-np.pi)
        ary = ary - mean
        ary = np.arctan2(np.sin(ary), np.cos(ary))

    ary = np.sort(ary)
    interval_idx_inc = int(np.floor(prob * n))
    n_intervals = n - interval_idx_inc
    interval_width = np.subtract(ary[interval_idx_inc:], ary[:n_intervals], dtype=np.float_)

    if len(interval_width) == 0:
        raise ValueError("Too few elements for interval calculation. ")

    min_idx = np.argmin(interval_width)

    return ary, mean, min_idx, interval_idx_inc


def _hdi_nearest(ary, prob, circular, skipna):
    """Compute HDI over the flattened array as closest samples that contain the given prob."""
    ary, mean, min_idx, interval_idx_inc = _hdi_linear_nearest_common(ary, prob, skipna, circular)

    hdi_min = ary[min_idx]
    hdi_max = ary[min_idx + interval_idx_inc]

    if circular:
        hdi_min = hdi_min + mean
        hdi_max = hdi_max + mean
        hdi_min = np.arctan2(np.sin(hdi_min), np.cos(hdi_min))
        hdi_max = np.arctan2(np.sin(hdi_max), np.cos(hdi_max))

    hdi_interval = np.array([hdi_min, hdi_max])

    return hdi_interval


def _hdi_multimodal(ary, prob, skipna, max_modes):
    """Compute HDI if the distribution is multimodal."""
    ary = ary.flatten()
    if skipna:
        ary = ary[~np.isnan(ary)]

    if ary.dtype.kind == "f":
        bins, density, _ = _kde(ary)
        lower, upper = bins[0], bins[-1]
        range_x = upper - lower
        dx = range_x / len(density)
    else:
        bins = _get_bins(ary)
        density, _ = np.histogram(ary, bins=bins, density=True)
        dx = np.diff(bins)[0]

    density *= dx

    idx = np.argsort(-density)
    intervals = bins[idx][density[idx].cumsum() <= prob]
    intervals.sort()

    intervals_splitted = np.split(intervals, np.where(np.diff(intervals) >= dx * 1.1)[0] + 1)

    hdi_intervals = np.full((max_modes, 2), np.nan)
    for i, interval in enumerate(intervals_splitted):
        if i == max_modes:
            warnings.warn(
                f"found more modes than {max_modes}, returning only the first {max_modes} modes"
            )
            break
        if interval.size == 0:
            hdi_intervals[i] = np.asarray([bins[0], bins[0]])
        else:
            hdi_intervals[i] = np.asarray([interval[0], interval[-1]])

    return np.array(hdi_intervals)
