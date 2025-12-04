"""Class with dataarray functions.

"dataarray" functions take :class:`xarray.DataArray` as inputs.
"""

import warnings
from collections.abc import Sequence

import numpy as np
from arviz_base import rcParams
from xarray import DataArray, apply_ufunc, broadcast, concat
from xarray_einstats.stats import _apply_nonreduce_func

from arviz_stats.base.array import array_stats
from arviz_stats.validate import (
    validate_ci_prob,
    validate_dims,
    validate_dims_chain_draw_axis,
    validate_prob,
)


class BaseDataArray:
    """Class with numpy+scipy only functions that take DataArray inputs."""

    def __init__(self, array_class=None):
        self.array_class = array_stats if array_class is None else array_class

    def eti(self, da, prob=None, dim=None, method="linear", **kwargs):
        """Compute eti on DataArray input."""
        dims = validate_dims(dim)
        prob = validate_ci_prob(prob)
        eti_coord = DataArray(
            ["lower", "upper"], dims=["ci_bound"], attrs={"ci_kind": "eti", "ci_prob": prob}
        )

        return apply_ufunc(
            self.array_class.eti,
            da,
            prob,
            input_core_dims=[dims, []],
            output_core_dims=[["ci_bound"]],
            kwargs={"axis": np.arange(-len(dims), 0, 1), "method": method, **kwargs},
        ).assign_coords({"ci_bound": eti_coord})

    def hdi(self, da, prob=None, dim=None, method="nearest", **kwargs):
        """Compute hdi on DataArray input."""
        dims = validate_dims(dim)
        prob = validate_ci_prob(prob)

        mode_dim = "mode" if da.name is None else f"{da.name}_mode"
        hdi_coord = DataArray(
            ["lower", "upper"], dims=["ci_bound"], attrs={"ci_kind": "hdi", "ci_prob": prob}
        )
        return apply_ufunc(
            self.array_class.hdi,
            da,
            prob,
            input_core_dims=[dims, []],
            output_core_dims=[
                [mode_dim, "ci_bound"] if method.startswith("multimodal") else ["ci_bound"]
            ],
            kwargs={
                "axis": np.arange(-len(dims), 0, 1),
                "method": method,
                **kwargs,
            },
        ).assign_coords({"ci_bound": hdi_coord})

    def ess(self, da, sample_dims=None, method="bulk", relative=False, prob=None):
        """Compute ess on DataArray input."""
        dims, chain_axis, draw_axis = validate_dims_chain_draw_axis(sample_dims)
        if method in ("tail", "local") and isinstance(prob, Sequence):
            prob = (validate_prob(prob[0], allow_0=True), validate_prob(prob[1]))
        elif method in ("tail", "quantile"):
            prob = validate_ci_prob(prob)
        return apply_ufunc(
            self.array_class.ess,
            da,
            input_core_dims=[dims],
            output_core_dims=[[]],
            kwargs={
                "method": method,
                "relative": relative,
                "prob": prob,
                "chain_axis": chain_axis,
                "draw_axis": draw_axis,
            },
        )

    def compute_ranks(self, da, dim=None, relative=False):
        """Compute ranks on DataArray input."""
        dims = validate_dims(dim)
        return _apply_nonreduce_func(
            self.array_class.compute_ranks,
            da,
            dims,
            {},
            func_kwargs={"relative": relative},
        )

    def rhat(self, da, sample_dims=None, method="bulk"):
        """Compute rhat on DataArray input."""
        dims = validate_dims(sample_dims)
        if len(dims) != 2:
            raise ValueError("dims must be of length 2 for rhat computation")
        return apply_ufunc(
            self.array_class.rhat,
            da,
            input_core_dims=[dims],
            output_core_dims=[[]],
            kwargs={"method": method, "chain_axis": -2, "draw_axis": -1},
        )

    def rhat_nested(self, da, superchain_ids, sample_dims=None, method="rank"):
        """Compute nested rhat on DataArray input."""
        dims = validate_dims(sample_dims)
        if len(dims) != 2:
            raise ValueError("dims must be of length 2 for rhat computation")
        return apply_ufunc(
            self.array_class.rhat_nested,
            da,
            input_core_dims=[dims],
            output_core_dims=[[]],
            kwargs={
                "superchain_ids": superchain_ids,
                "method": method,
                "chain_axis": -2,
                "draw_axis": -1,
            },
        )

    def mcse(self, da, sample_dims=None, method="mean", prob=None):
        """Compute mcse on DataArray input."""
        dims, chain_axis, draw_axis = validate_dims_chain_draw_axis(sample_dims)
        return apply_ufunc(
            self.array_class.mcse,
            da,
            input_core_dims=[dims],
            output_core_dims=[[]],
            kwargs={
                "method": method,
                "prob": prob,
                "chain_axis": chain_axis,
                "draw_axis": draw_axis,
            },
        )

    def get_bins(self, da, dim=None, bins="arviz"):
        """Compute bins or align provided ones with DataArray input."""
        dims = validate_dims(dim)
        return apply_ufunc(
            self.array_class.get_bins,
            da,
            input_core_dims=[dims],
            output_core_dims=[["edges_dim" if da.name is None else f"edges_dim_{da.name}"]],
            kwargs={
                "bins": bins,
                "axis": np.arange(-len(dims), 0, 1),
            },
        )

    # pylint: disable=redefined-builtin
    def histogram(self, da, dim=None, bins=None, range=None, weights=None, density=None):
        """Compute histogram on DataArray input."""
        dims = validate_dims(dim)
        edges_dim = "edges_dim" if da.name is None else f"edges_dim_{da.name}"
        hist_dim = "hist_dim" if da.name is None else f"hist_dim_{da.name}"
        input_core_dims = [dims]
        if isinstance(bins, DataArray):
            if "plot_axis" in bins.dims:
                bins_dims = [dim for dim in bins.dims if dim not in dims + ["plot_axis"]]
                assert len(bins_dims) == 1
                hist_dim = bins_dims[0]
                bins = (
                    concat(
                        (
                            bins.sel(plot_axis="left_edges", drop=True),
                            bins.sel(plot_axis="right_edges", drop=True).isel({hist_dim: [-1]}),
                        ),
                        dim=hist_dim,
                    )
                    .rename({hist_dim: edges_dim})
                    .drop_vars(edges_dim)
                )
            elif edges_dim not in bins.dims:
                raise ValueError(
                    "Invalid 'bins' DataArray, it should contain either 'plot_axis' or "
                    f"'{edges_dim}' dimension"
                )
            input_core_dims.append([edges_dim])
        else:
            input_core_dims.append([])
        if isinstance(range, DataArray):
            range_dims = [dim for dim in range.dims if dim not in dims]
            assert len(range_dims) == 1
            input_core_dims.append(range_dims)
        else:
            input_core_dims.append([])
        if isinstance(weights, DataArray):
            assert weights.dims == da.dims
            input_core_dims.append(dims)
        else:
            input_core_dims.append([])

        hist, edges = apply_ufunc(
            self.array_class.histogram,
            da,
            bins,
            range,
            weights,
            kwargs={
                "density": density,
                "axis": np.arange(-len(dims), 0, 1),
            },
            input_core_dims=input_core_dims,
            output_core_dims=[[hist_dim], [edges_dim]],
        )
        plot_axis = DataArray(["histogram", "left_edges", "right_edges"], dims="plot_axis")
        out = concat(
            (
                hist,
                edges.isel({edges_dim: slice(None, -1)}).rename({edges_dim: hist_dim}),
                edges.isel({edges_dim: slice(1, None)}).rename({edges_dim: hist_dim}),
            ),
            dim=plot_axis,
        )
        return out

    def kde(self, da, dim=None, circular=False, grid_len=512, **kwargs):
        """Compute kde on DataArray input."""
        dims = validate_dims(dim)
        grid, pdf, bw = apply_ufunc(
            self.array_class.kde,
            da,
            kwargs={
                "circular": circular,
                "grid_len": grid_len,
                "axis": np.arange(-len(dims), 0, 1),
                **kwargs,
            },
            input_core_dims=[dims],
            output_core_dims=[["kde_dim"], ["kde_dim"], []],
        )
        plot_axis = DataArray(["x", "y"], dims="plot_axis")
        out = concat((grid, pdf), dim=plot_axis)
        return out.assign_coords({"bw" if da.name is None else f"bw_{da.name}": bw})

    def qds(self, da, nquantiles=100, binwidth=None, dotsize=1, stackratio=1, dim=None, **kwargs):
        """Compute quantile dots on DataArray input."""
        dims = validate_dims(dim)
        x, y, radius = apply_ufunc(
            self.array_class.qds,
            da,
            kwargs={
                "nquantiles": nquantiles,
                "binwidth": binwidth,
                "dotsize": dotsize,
                "stackratio": stackratio,
                "axis": np.arange(-len(dims), 0, 1),
                **kwargs,
            },
            input_core_dims=[dims],
            output_core_dims=[["qd_dim"], ["qd_dim"], []],
        )
        plot_axis = DataArray(["x", "y"], dims="plot_axis")
        out = concat((x, y), dim=plot_axis)
        return out.assign_coords({"radius" if da.name is None else f"radius_{da.name}": radius})

    def ecdf(self, da, npoints=200, pit=False, dim=None, **kwargs):
        """Compute empirical cumulative distribution function on DataArray input."""
        dims = validate_dims(dim)
        x, y = apply_ufunc(
            self.array_class.ecdf,
            da,
            kwargs={
                "npoints": npoints,
                "pit": pit,
                "axis": np.arange(-len(dims), 0, 1),
                **kwargs,
            },
            input_core_dims=[dims],
            output_core_dims=[["ecdf_dim"], ["ecdf_dim"]],
        )
        plot_axis = DataArray(["x", "y"], dims="plot_axis")
        return concat((x, y), dim=plot_axis)

    def thin_factor(self, da, target_ess=None, reduce_func="mean"):
        """Get thinning factor over draw dimension to preserve ESS in samples or target a given ESS.

        Parameters
        ----------
        da : DataArray
        target_ess : int, optional
            By default, the ESS target will be preserving the ESS of all available samples.
            If an integer value is passed, it must be lower than the average ESS of the input
            samples.
        mode : {"mean", "min"}, default "mean"
        """
        n_samples = da.sizes["chain"] * da.sizes["draw"]
        ess = np.minimum(
            self.ess(da, method="bulk", sample_dims=["chain", "draw"]),
            self.ess(da, method="tail", sample_dims=["chain", "draw"]),
        )
        if reduce_func == "mean":
            ess_ave = ess.mean()
        elif reduce_func == "min":
            ess_ave = ess.min()
        else:
            raise ValueError(
                f"`reduce_func` {reduce_func} not recognized. Valid values are 'mean' or 'min'"
            )
        if target_ess is None:
            target_ess = ess_ave
        if target_ess > ess_ave:
            var_indicator = f"for {ess_ave.name} " if ess_ave.name is not None else ""
            warnings.warn(
                f"ESS not high enough to reach requested {target_ess} ESS. "
                "Returning 1 so no thinning is applied and "
                f"current ESS average {var_indicator} at {ess_ave.item()} is preserved."
            )
            return 1
        # defensive fallback in the unlikely case that ESS exceeds n_samples
        # this should only happen with uncorrelated chains
        if reduce_func == "min":
            factor = int(np.floor(n_samples / target_ess))
        else:
            factor = int(np.ceil(n_samples / target_ess))
        return max(1, factor)

    def thin(self, da, factor="auto", sample_dims=None):
        """Perform thinning on DataArray input."""
        if factor == "auto" and sample_dims is not None:
            warnings.warn("dims are ignored if factor is auto")
        if factor == "auto":
            factor = self.thin_factor(da)
            sample_dims = "draw"

        elif isinstance(factor, (float | int)):
            if sample_dims is None:
                sample_dims = rcParams["data.sample_dims"]
            if not isinstance(sample_dims, str):
                if len(sample_dims) >= 2:
                    raise ValueError("dims must be of length 1")
                if len(sample_dims) == 1:
                    sample_dims = sample_dims[0]

            factor = int(factor)
            if factor == 1:
                return da
            if factor < 1:
                raise ValueError("factor must be greater than 1")

        return da.sel({sample_dims: slice(None, None, factor)})

    def pareto_min_ss(self, da, sample_dims=None):
        """Compute the minimum effective sample size for all variables in the dataset."""
        dims, chain_axis, draw_axis = validate_dims_chain_draw_axis(sample_dims)
        return apply_ufunc(
            self.array_class.pareto_min_ss,
            da,
            input_core_dims=[dims],
            output_core_dims=[[]],
            kwargs={"chain_axis": chain_axis, "draw_axis": draw_axis},
        )

    def psislw(self, da, r_eff=1, dim=None):
        """Compute log weights for Pareto-smoothed importance sampling (PSIS) method."""
        dims = validate_dims(dim)
        return apply_ufunc(
            self.array_class.psislw,
            da,
            r_eff,
            input_core_dims=[dims, []],
            output_core_dims=[dims, []],
            kwargs={"axis": np.arange(-len(dims), 0, 1)},
        )

    def pareto_khat(self, da, sample_dims=None, r_eff=None, tail="both", log_weights=False):
        """Compute Pareto k-hat diagnostic on DataArray input."""
        dims, chain_axis, draw_axis = validate_dims_chain_draw_axis(sample_dims)
        return apply_ufunc(
            self.array_class.pareto_khat,
            da,
            input_core_dims=[dims],
            output_core_dims=[[]],
            kwargs={
                "chain_axis": chain_axis,
                "draw_axis": draw_axis,
                "r_eff": r_eff,
                "tail": tail,
                "log_weights": log_weights,
            },
            dask="parallelized",
            output_dtypes=[float],
        ).rename("pareto_k")

    def loo(
        self, da, sample_dims=None, reff=1.0, log_weights=None, pareto_k=None, log_jacobian=None
    ):
        """Compute PSIS-LOO-CV.

        Parameters
        ----------
        da : DataArray
            Log-likelihood values with shape (chain, draw, *obs_dims)
        sample_dims : list of str, optional
            Sample dimensions. Defaults to ["chain", "draw"]
        reff : float, default 1.0
            Relative effective sample size
        log_weights : DataArray, optional
            Pre-computed PSIS log weights (same shape as da)
        pareto_k : DataArray, optional
            Pre-computed Pareto k values (shape: obs_dims only)
        log_jacobian : DataArray, optional
            Log-Jacobian adjustment (shape: obs_dims only)

        Returns
        -------
        tuple of (elpd_i, pareto_k, p_loo_i) : DataArrays
            Pointwise LOO values for each observation
        """
        dims, chain_axis, draw_axis = validate_dims_chain_draw_axis(sample_dims)
        kwargs = {
            "chain_axis": chain_axis,
            "draw_axis": draw_axis,
            "reff": reff,
        }

        if log_weights is not None and pareto_k is not None:

            def loo_with_weights(ary, lw, pk, lj=None, **kw):
                return self.array_class.loo(ary, log_weights=lw, pareto_k=pk, log_jacobian=lj, **kw)

            inputs = [da, log_weights, pareto_k]
            input_dims = [dims, dims, []]
            if log_jacobian is not None:
                inputs.append(log_jacobian)
                input_dims.append([])

            return apply_ufunc(
                loo_with_weights,
                *inputs,
                input_core_dims=input_dims,
                output_core_dims=[[], [], []],
                vectorize=True,
                join="left",
                kwargs=kwargs,
            )

        if log_jacobian is not None:

            def loo_with_jacobian(ary, lj, **kw):
                return self.array_class.loo(ary, log_jacobian=lj, **kw)

            return apply_ufunc(
                loo_with_jacobian,
                da,
                log_jacobian,
                input_core_dims=[dims, []],
                output_core_dims=[[], [], []],
                vectorize=True,
                join="left",
                kwargs=kwargs,
            )

        return apply_ufunc(
            self.array_class.loo,
            da,
            input_core_dims=[dims],
            output_core_dims=[[], [], []],
            kwargs=kwargs,
        )

    def loo_approximate_posterior(self, da, log_p, log_q, sample_dims=None, log_jacobian=None):
        """Compute PSIS-LOO-CV with approximate posterior correction.

        Parameters
        ----------
        da : DataArray
            Log-likelihood values
        log_p : DataArray
            Target log-density values (chain, draw)
        log_q : DataArray
            Proposal log-density values (chain, draw)
        sample_dims : list of str, optional
            Sample dimensions. Defaults to ["chain", "draw"]
        log_jacobian : DataArray, optional
            Log-Jacobian adjustment for variable transformations

        Returns
        -------
        tuple of (elpd_i, pareto_k, p_loo_i) : DataArrays
            Pointwise LOO values for each observation
        """
        dims, chain_axis, draw_axis = validate_dims_chain_draw_axis(sample_dims)
        kwargs = {
            "chain_axis": chain_axis,
            "draw_axis": draw_axis,
        }

        if log_jacobian is not None:

            def loo_approx_with_jacobian(ary, lp, lq, lj, **kw):
                return self.array_class.loo_approximate_posterior(
                    ary, lp, lq, log_jacobian=lj, **kw
                )

            return apply_ufunc(
                loo_approx_with_jacobian,
                da,
                log_p,
                log_q,
                log_jacobian,
                input_core_dims=[dims, dims, dims, []],
                output_core_dims=[[], [], []],
                vectorize=True,
                join="left",
                kwargs=kwargs,
            )

        return apply_ufunc(
            self.array_class.loo_approximate_posterior,
            da,
            log_p,
            log_q,
            input_core_dims=[dims, dims, dims],
            output_core_dims=[[], [], []],
            vectorize=True,
            kwargs=kwargs,
        )

    def loo_mixture(self, da, sample_dims=None, log_jacobian=None):
        """Compute mixture importance sampling LOO (Mix-IS-LOO).

        Parameters
        ----------
        da : DataArray
            Log-likelihood values
        sample_dims : list of str, optional
            Sample dimensions. Defaults to ["chain", "draw"]
        log_jacobian : DataArray, optional
            Log-Jacobian adjustment for variable transformations

        Returns
        -------
        elpd_i : DataArray
            Pointwise expected log predictive density
        p_loo_i : DataArray
            Pointwise effective number of parameters
        mix_log_weights : DataArray
            Mixture log weights
        """
        dims, _, _ = validate_dims_chain_draw_axis(sample_dims)
        obs_dims = [d for d in da.dims if d not in dims]

        obs_axes = tuple(da.dims.index(d) for d in obs_dims)
        sample_axes = (da.dims.index(dims[0]), da.dims.index(dims[1]))

        jac_values = log_jacobian.values if log_jacobian is not None else None

        elpd_i_vals, p_loo_i_vals, mix_lw_vals = self.array_class.loo_mixture(
            da.values,
            obs_axes=obs_axes,
            chain_axis=sample_axes[0],
            draw_axis=sample_axes[1],
            log_jacobian=jac_values,
        )

        obs_coords = {d: da.coords[d] for d in obs_dims}
        elpd_i = DataArray(elpd_i_vals, dims=obs_dims, coords=obs_coords)
        p_loo_i = DataArray(p_loo_i_vals, dims=obs_dims, coords=obs_coords)
        mix_log_weights = DataArray(mix_lw_vals, dims=da.dims, coords=da.coords)

        return elpd_i, p_loo_i, mix_log_weights

    def loo_score(self, da, y_obs, log_weights, kind="crps", sample_dims=None):
        """Compute CRPS or SCRPS with PSIS-LOO-CV weights.

        Parameters
        ----------
        da : DataArray
            Posterior predictive samples
        y_obs : DataArray or scalar
            Observed values
        log_weights : DataArray
            PSIS-LOO log weights
        kind : str, default "crps"
            "crps" or "scrps"
        sample_dims : list of str, optional
            Sample dimensions. Defaults to ["chain", "draw"]

        Returns
        -------
        scores : DataArray
            Score values (negative CRPS or SCRPS for maximization)
        """
        dims, chain_axis, draw_axis = validate_dims_chain_draw_axis(sample_dims)
        return apply_ufunc(
            self.array_class.loo_score,
            da,
            y_obs,
            log_weights,
            input_core_dims=[dims, [], dims],
            output_core_dims=[[]],
            kwargs={
                "kind": kind,
                "chain_axis": chain_axis,
                "draw_axis": draw_axis,
            },
        )

    def loo_summary(self, da, p_loo_i):
        """Aggregate pointwise LOO values.

        Parameters
        ----------
        da : DataArray
            Pointwise expected log predictive density values (elpd_i)
        p_loo_i : DataArray
            Pointwise effective number of parameters

        Returns
        -------
        elpd : float
            Total expected log predictive density
        se : float
            Standard error of elpd
        p_loo : float
            Total effective number of parameters
        lppd : float
            Log pointwise predictive density
        """
        return self.array_class.loo_summary(da.values, p_loo_i.values)

    def power_scale_lw(self, da, alpha=0, dim=None):
        """Compute log weights for power-scaling component by alpha."""
        dims = validate_dims(dim)
        return apply_ufunc(
            self.array_class.power_scale_lw,
            da,
            alpha,
            input_core_dims=[dims, []],
            output_core_dims=[dims],
            kwargs={"axis": np.arange(-len(dims), 0, 1)},
        )

    def power_scale_sense(self, da, lower_w, upper_w, lower_alpha, upper_alpha, sample_dims=None):
        """Compute power-scaling sensitivity."""
        dims, chain_axis, draw_axis = validate_dims_chain_draw_axis(sample_dims)
        return apply_ufunc(
            self.array_class.power_scale_sense,
            *broadcast(da, lower_w, upper_w),
            lower_alpha,
            upper_alpha,
            input_core_dims=[dims, dims, dims, [], []],
            output_core_dims=[[]],
            kwargs={"chain_axis": chain_axis, "draw_axis": draw_axis},
        )

    def autocorr(self, da, dim=None):
        """Compute autocorrelation on DataArray input."""
        dims = validate_dims(dim)
        return apply_ufunc(
            self.array_class.autocorr,
            da,
            input_core_dims=[dims],
            output_core_dims=[dims],
        )

    def mean(self, da, round_to=None, skipna=False, dim=None):
        """Compute mean on DataArray input."""
        dims = validate_dims(dim)

        return apply_ufunc(
            self.array_class.mean,
            da,
            input_core_dims=[dims],
            output_core_dims=[[]],
            kwargs={"round_to": round_to, "skipna": skipna, "axis": np.arange(-len(dims), 0, 1)},
        )

    def median(self, da, round_to=None, skipna=False, dim=None):
        """Compute median on DataArray input."""
        dims = validate_dims(dim)

        return apply_ufunc(
            self.array_class.median,
            da,
            input_core_dims=[dims],
            output_core_dims=[[]],
            kwargs={"round_to": round_to, "skipna": skipna, "axis": np.arange(-len(dims), 0, 1)},
        )

    def mode(self, da, round_to=None, skipna=False, dim=None):
        """Compute mode on DataArray input."""
        dims = validate_dims(dim)

        return apply_ufunc(
            self.array_class.mode,
            da,
            input_core_dims=[dims],
            output_core_dims=[[]],
            kwargs={"round_to": round_to, "skipna": skipna, "axis": np.arange(-len(dims), 0, 1)},
        )


dataarray_stats = BaseDataArray(array_class=array_stats)
