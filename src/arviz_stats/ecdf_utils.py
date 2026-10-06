"""Utility functions for computing the PIT ECDF and the simultaneous confidence bands."""

import numpy as np
import xarray as xr


def _pit_for_hist(value, left_edges, right_edges, histogram, *, eps, rng):
    """Help compute PIT for hist on scalar `value` and 1d data."""
    bin_edges = np.append(left_edges, right_edges[-1])
    dx = np.diff(bin_edges)

    total = np.sum(histogram * dx)
    if total > 0:
        histogram = histogram / total

    xmin = bin_edges[:-1]
    xmax = bin_edges[1:]
    cdf = np.concatenate([[0.0], np.cumsum(histogram * dx)])

    if value <= xmin[0]:
        return rng.uniform(0, eps)
    if value >= xmax[-1]:
        return rng.uniform(1 - eps, 1)

    idx = np.searchsorted(xmin, value, side="right") - 1
    return cdf[idx] + (value - xmin[idx]) * histogram[idx]


def compute_pit_for_histogram(dt_group, hist_dt, sample_dims):
    """Compute the PIT values for histogram distributions."""
    rng = np.random.default_rng(3146)
    pit_results = {}
    for var_name in dt_group.data_vars:
        hist_data = hist_dt[var_name]
        left_edges = hist_data.sel(plot_axis="left_edges")
        right_edges = hist_data.sel(plot_axis="right_edges")
        histogram = hist_data.sel(plot_axis="histogram")

        if len(sample_dims) == 1 and sample_dims[0] != "sample":
            sample = dt_group[var_name].rename(sample=sample_dims[0])
        else:
            sample = dt_group[var_name].stack(sample=sample_dims)
        eps = 0.5 / sample.sizes["sample"]

        pit_da = xr.apply_ufunc(
            _pit_for_hist,
            sample,
            left_edges,
            right_edges,
            histogram,
            input_core_dims=[
                [],
                ["hist_dim_" + var_name],
                ["hist_dim_" + var_name],
                ["hist_dim_" + var_name],
            ],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
            kwargs={"eps": eps, "rng": rng},
        )

        pit_results[var_name] = pit_da

    return xr.Dataset(pit_results)


def _interp_cdf(value, grid, cdf, *, eps, rng):
    """Help interpolate cdf on scalar `value` and 1d data."""
    pit = float(np.interp(value, grid, cdf, left=0.0, right=1.0))
    if pit == 0.0:
        return rng.uniform(0, eps)
    if pit == 1.0:
        return rng.uniform(1 - eps, 1)
    return pit


def compute_pit_for_kde(dt_group, kde_dt, sample_dims):
    """Compute the PIT values for KDE distributions."""
    rng = np.random.default_rng(3146)
    pit_results = {}
    for var_name in dt_group.data_vars:
        kde_data = kde_dt[var_name]
        grid = kde_data.sel(plot_axis="x")
        density = kde_data.sel(plot_axis="y")
        dx = grid.diff("kde_dim").isel(kde_dim=0)
        cdf = (density.cumsum("kde_dim") * dx).clip(0.0, 1.0)

        if len(sample_dims) == 1 and sample_dims[0] != "sample":
            sample = dt_group[var_name].rename(sample=sample_dims[0])
        else:
            sample = dt_group[var_name].stack(sample=sample_dims)
        eps = 0.5 / sample.sizes["sample"]

        pit = xr.apply_ufunc(
            _interp_cdf,
            sample,
            grid,
            cdf,
            input_core_dims=[[], ["kde_dim"], ["kde_dim"]],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
            kwargs={"eps": eps, "rng": rng},
        )
        pit_results[var_name] = pit

    return xr.Dataset(pit_results)


def _pit_f_for_qds(values, quantile_positions, radius, nqds, *, rng):
    """Help compute PIT for quantile dotplots on 1d data."""
    pit = np.zeros_like(values, dtype=float)

    for i, xi in enumerate(values):
        left_edges = quantile_positions - radius
        right_edges = quantile_positions + radius

        mask = left_edges > xi
        if np.any(mask):
            q_max = np.argmax(mask)
        else:
            q_max = nqds

        if q_max > 0:
            target_position = quantile_positions[q_max - 1]
            same_position = quantile_positions == target_position
            candidates_mask = same_position & (right_edges >= xi)
            if np.any(candidates_mask):
                q_min = np.argmax(candidates_mask)
            else:
                q_min = q_max - 1 if q_max > 0 else 0
            pit[i] = rng.uniform(q_min, q_max) / nqds
        else:
            pit[i] = rng.uniform(0, 0.5) / nqds

    return pit


def compute_pit_for_qds(dt_group, qds_dt, sample_dims):
    """Compute the PIT values for quantile dot plot distributions."""
    rng = np.random.default_rng(3146)
    pit_results = {}
    for var_name in dt_group.data_vars:
        qds_data = qds_dt[var_name]

        quantile_positions = qds_data.sel(plot_axis="x").values

        radius = qds_data.coords[f"radius_{var_name}"].values
        nqds = qds_data.sizes["qd_dim"]

        if len(sample_dims) == 1 and sample_dims[0] != "sample":
            sample = dt_group[var_name].rename(sample=sample_dims[0])
        else:
            sample = dt_group[var_name].stack(sample=sample_dims)

        pit_da = xr.apply_ufunc(
            _pit_f_for_qds,
            sample,
            quantile_positions,
            radius,
            nqds,
            input_core_dims=[["sample"], ["qd_dim"], [], []],
            output_core_dims=[["sample"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
            kwargs={"rng": rng},
        )

        pit_results[var_name] = pit_da

    return xr.Dataset(pit_results)
