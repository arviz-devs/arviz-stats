"""Diagnostics as gufuncs powered by xarray, numba and einops."""

import numba
import numpy as np
import scipy
import xarray as xr
from scipy.fftpack import next_fast_len
from xarray_einstats import stats
from xarray_einstats.einops import rearrange

__all__ = ["rhat", "ess"]


def _backtransform_ranks(da, c=3 / 8):  # pylint: disable=invalid-name
    """Backtransformation of ranks.

    Parameters
    ----------
    da : xr.DataArray
        Ranks array. It must have dimensions named `chain` and `draw`
    c : float
        Fractional offset. Defaults to c = 3/8 as recommended by Blom (1958).

    Returns
    -------
    xr.DataArray

    References
    ----------
    Blom, G. (1958). Statistical Estimates and Transformed Beta-Variables. Wiley; New York.
    """
    size = len(da.chain) * len(da.draw)
    return (da - c) / (size - 2 * c + 1)


def _z_scale(da, **kwargs):
    """Calculate z_scale.

    Parameters
    ----------
    da : xr.DataArray
        Input array. It must have dimensions named `chain` and `draw`

    Returns
    -------
    xr.DataArray
    """
    kwargs = kwargs.copy()
    if kwargs.get("dask", None) == "allowed":
        # scipy doesn't support dask properly, so even if we use dask="allowed" or dask directly
        # for everything else we need to force "parallelized" mode here
        kwargs["dask"] = "parallelized"
    rank = stats.rankdata(da, dims=("chain", "draw"), method="average", **kwargs)
    rank = _backtransform_ranks(rank)
    norm_dist = stats.XrContinuousRV(scipy.stats.norm)
    return norm_dist.ppf(rank, apply_kwargs=kwargs)


def _split_chains(da, **kwargs):
    """Split and stack chains."""
    half = len(da.draw) // 2
    kwargs = kwargs.copy()
    if kwargs.get("dask", None) == "parallelized":
        # here we force dask="allowed" because einops doesn't play well with parallelized mode
        kwargs["dask"] = "allowed"
    if kwargs.get("dask", None) == "allowed":
        pass  # pylint: disable=unused-import

    return rearrange(da, "(d1 d2)=draw -> (chain d1)=c2 d2", d2=half, **kwargs).rename(
        d2="draw", c2="chain"
    )


def _z_fold(da, **kwargs):
    """Fold and z-scale values."""
    da = abs(da - da.median(("chain", "draw")))
    da = _z_scale(da, **kwargs)
    return da


def _rhat(da):
    """Compute the rhat for an n-D DataArray."""
    num_samples = len(da.draw)

    # Calculate chain mean
    chain_mean = da.mean("draw")
    # Calculate chain variance
    chain_var = da.var("draw", ddof=1)
    # Calculate between-chain variance
    between_chain_variance = num_samples * chain_mean.var("chain", ddof=1)
    # Calculate within-chain variance
    within_chain_variance = chain_var.mean("chain")
    # Estimate of marginal posterior variance
    rhat_value = np.sqrt(
        (between_chain_variance / within_chain_variance + num_samples - 1) / (num_samples)
    )
    return rhat_value


def _rhat_rank(da, **kwargs):
    """Compute the rank normalized rhat for an n-D DataArray.

    Computation follows https://arxiv.org/abs/1903.08008
    """
    split_da = _split_chains(da, **kwargs)
    rhat_bulk = _rhat(_z_scale(split_da, **kwargs))

    rhat_tail = _rhat(_z_fold(split_da, **kwargs))

    rhat_rank = xr.where(rhat_bulk > rhat_tail, rhat_bulk, rhat_tail)
    return rhat_rank


def rhat(ds, group="posterior", method="rank", **kwargs):
    """Calculate rhat diagnostic."""
    func_map = {"identity": _rhat, "rank": _rhat_rank}
    if method not in func_map:
        raise ValueError("method not recognized")
    rhat_func = func_map[method]
    if isinstance(ds, xr.DataTree):
        ds = ds[group].dataset
        return ds.map(rhat_func, **kwargs)
    if isinstance(ds, xr.Dataset):
        return ds.map(rhat_func, **kwargs)
    if isinstance(ds, xr.DataArray):
        return rhat_func(ds, **kwargs)
    raise TypeError("Input type not recognized.")


# might move to einstats
# I tried https://github.com/xgcm/xrft but it only wraps rfftn and was more a headache than
# help, rfft and irfft exist both in numpy and dask, so the wrappers below will
# support dask="allowed" without problem
def rfft(da, dim=None, n=None, prefix="freq_", **kwargs):
    return xr.apply_ufunc(
        np.fft.rfft,
        da,
        input_core_dims=[[dim]],
        output_core_dims=[[f"{prefix}{dim}"]],
        kwargs={"n": n},
        **kwargs,
    )


def irfft(da, dim=None, n=None, prefix="freq_", **kwargs):
    out_dim = dim.replace(prefix, "")
    return xr.apply_ufunc(
        np.fft.irfft,
        da,
        input_core_dims=[[dim]],
        output_core_dims=[[out_dim]],
        kwargs={"n": n},
        **kwargs,
    )


def autocov(da, dim="draw", **kwargs):
    """Compute autocovariance estimates for every lag for the input array.

    Parameters
    ----------
    ary : xr.DataArray
        A DataArray containing MCMC samples. It must have the ``draw`` dimension

    Returns
    -------
    DataArray same size as the input array
    """
    draw_coord = da[dim]
    n = len(draw_coord)
    m = next_fast_len(2 * n)

    fft_da = rfft(da - da.mean(dim), n=m, dim=dim, **kwargs)
    fft_da *= np.conjugate(fft_da)

    cov = irfft(fft_da, n=m, dim=f"freq_{dim}", **kwargs).isel(draw=slice(None, n))
    cov /= n

    return cov.assign_coords({dim: draw_coord})


def autocorr(da, dim="draw", **kwargs):
    da = autocov(da, dim=dim, **kwargs)
    return da / da.isel({dim: 0})


@numba.guvectorize(
    [
        "void(float64[:,:], float64, float64[:])",
    ],
    "(n,m),()->()",
    cache=True,
    target="cpu",
    nopython=True,
)
def geyer(acov, chain_mean_term, tau_hat):
    tau_hat[:] = 0
    n_draw = acov.shape[1]
    mean_var = np.mean(acov[:, 0]) * n_draw / (n_draw - 1.0)
    var_plus = mean_var * (n_draw - 1.0) / n_draw + chain_mean_term

    rho_hat_t = np.zeros(n_draw)
    rho_hat_even = 1.0
    rho_hat_t[0] = rho_hat_even
    rho_hat_odd = 1.0 - (mean_var - np.mean(acov[:, 1])) / var_plus
    rho_hat_t[1] = rho_hat_odd

    # Geyer's initial positive sequence
    t = 1
    while t < (n_draw - 3) and (rho_hat_even + rho_hat_odd) > 0.0:
        rho_hat_even = 1.0 - (mean_var - np.mean(acov[:, t + 1])) / var_plus
        rho_hat_odd = 1.0 - (mean_var - np.mean(acov[:, t + 2])) / var_plus
        if (rho_hat_even + rho_hat_odd) >= 0:
            rho_hat_t[t + 1] = rho_hat_even
            rho_hat_t[t + 2] = rho_hat_odd
        t += 2

    max_t = t - 2
    # improve estimation
    if rho_hat_even > 0:
        rho_hat_t[max_t + 1] = rho_hat_even
    # Geyer's initial monotone sequence
    t = 1
    while t <= max_t - 2:
        if (rho_hat_t[t + 1] + rho_hat_t[t + 2]) > (rho_hat_t[t - 1] + rho_hat_t[t]):
            rho_hat_t[t + 1] = (rho_hat_t[t - 1] + rho_hat_t[t]) / 2.0
            rho_hat_t[t + 2] = rho_hat_t[t + 1]
        t += 2

    tau_hat += (
        -1.0 + 2.0 * np.sum(rho_hat_t[: max_t + 1]) + np.sum(rho_hat_t[max_t + 1 : max_t + 2])
    )


def _ess(da, relative=False, **kwargs):
    n_chain = len(da["chain"])
    n_draw = len(da["draw"])
    maxmin_keep = da.max(("chain", "draw")) - da.min(("chain", "draw")) > np.finfo(float).resolution
    if np.any(~maxmin_keep):
        if not np.any(maxmin_keep):
            return xr.zeros_like(maxmin_keep, dtype=float) + n_chain * n_draw
        da = da.where(maxmin_keep, drop=True)

    acov = autocov(da, **kwargs)
    chain_mean = da.mean("draw")
    chain_mean_term = chain_mean.var(dim="chain", ddof=1) if n_chain > 1 else 0

    kwargs = kwargs.copy()
    if kwargs.get("dask", None) == "allowed" and kwargs.get("dask_override", False):
        kwargs["dask"] = "parallelized"
    tau_hat = xr.apply_ufunc(
        geyer,
        acov,
        chain_mean_term,
        input_core_dims=[["chain", "draw"], []],
        output_core_dims=[[]],
        **kwargs,
    )

    ess_value = n_chain * n_draw
    tau_hat = tau_hat.where(tau_hat > 1 / np.log10(ess_value), 1 / np.log10(ess_value))
    ess_value = (1 if relative else ess_value) / tau_hat

    if np.any(~maxmin_keep):
        ess_aux = xr.zeros_like(maxmin_keep.where(~maxmin_keep, drop=True)) + n_chain * n_draw
        return xr.merge((ess_value, ess_aux), join="outer")
    return ess_value


def _ess_mean(da, relative=False, **kwargs):
    return _ess(_split_chains(da, **kwargs), relative=relative, **kwargs)


def _ess_bulk(da, relative=False, **kwargs):
    da = _z_scale(_split_chains(da, **kwargs), **kwargs)
    return _ess(da, relative=relative, **kwargs)


def ess(ds, group="posterior", method="bulk", **kwargs):
    """Calculate ess statistic/diagnostic."""
    func_map = {"mean": _ess_mean, "bulk": _ess_bulk}
    if method not in func_map:
        raise ValueError("method not recognized")
    ess_func = func_map[method]
    if isinstance(ds, xr.DataTree):
        ds = ds[group].dataset
        return ds.map(ess_func, **kwargs)
    if isinstance(ds, xr.Dataset):
        return ds.map(ess_func, **kwargs)
    if isinstance(ds, xr.DataArray):
        return ess_func(ds, **kwargs)
    raise TypeError("Input type not recognized.")
