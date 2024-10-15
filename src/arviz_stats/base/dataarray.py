"""Class with dataarray functions.

"dataarray" functions take :class:`xarray.DataArray` as inputs.
"""

import warnings

import numpy as np
from arviz_base import rcParams
from xarray import DataArray, apply_ufunc, concat
from xarray_einstats.stats import _apply_nonreduce_func

from arviz_stats.base.array import array_stats
from arviz_stats.validate import validate_ci_prob, validate_dims, validate_dims_chain_draw_axis


class BaseDataArray:
    """Class with numpy+scipy only functions that take DataArray inputs."""

    def __init__(self, array_class=None):
        self.array_class = array_stats if array_class is None else array_class

    def eti(self, da, prob=None, dims=None, method="linear"):
        """Compute eti on DataArray input."""
        dims = validate_dims(dims)
        prob = validate_ci_prob(prob)

        return apply_ufunc(
            self.array_class.eti,
            da,
            prob,
            input_core_dims=[dims, []],
            output_core_dims=[["quantile"]],
            kwargs={"axis": np.arange(-len(dims), 0, 1), "method": method},
        )

    def hdi(
        self, da, prob=None, dims=None, method="nearest", circular=False, max_modes=10, skipna=False
    ):
        """Compute hdi on DataArray input."""
        dims = validate_dims(dims)
        prob = validate_ci_prob(prob)

        mode_dim = "mode" if da.name is None else f"{da.name}_mode"
        hdi_coord = DataArray(["lower", "higher"], dims=["hdi"], attrs={"hdi_prob": prob})
        return apply_ufunc(
            self.array_class.hdi,
            da,
            prob,
            input_core_dims=[dims, []],
            output_core_dims=[[mode_dim, "hdi"] if method == "multimodal" else ["hdi"]],
            kwargs={
                "method": method,
                "circular": circular,
                "skipna": skipna,
                "max_modes": max_modes,
                "axes": np.arange(-len(dims), 0, 1),
            },
        ).assign_coords({"hdi": hdi_coord})

    def ess(self, da, dims=None, method="bulk", relative=False, prob=None):
        """Compute ess on DataArray input."""
        dims, chain_axis, draw_axis = validate_dims_chain_draw_axis(dims)
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

    def compute_ranks(self, da, dims=None, relative=False):
        """Compute ranks on DataArray input."""
        dims = validate_dims(dims)
        return _apply_nonreduce_func(
            self.array_class.compute_ranks,
            da,
            dims,
            {},
            func_kwargs={"relative": relative},
        )

    def rhat(self, da, dims=None, method="bulk"):
        """Compute rhat on DataArray input."""
        dims = validate_dims(dims)
        if len(dims) != 2:
            raise ValueError("dims must be of length 2 for rhat computation")
        return apply_ufunc(
            self.array_class.rhat,
            da,
            input_core_dims=[dims],
            output_core_dims=[[]],
            kwargs={"method": method, "chain_axis": -2, "draw_axis": -1},
        )

    def mcse(self, da, dims=None, method="mean", prob=None):
        """Compute mcse on DataArray input."""
        dims, chain_axis, draw_axis = validate_dims_chain_draw_axis(dims)
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

    # pylint: disable=redefined-builtin
    def histogram(self, da, dims=None, bins=None, range=None, weights=None, density=None):
        """Compute histogram on DataArray input."""
        dims = validate_dims(dims)
        edges_dim = "edges_dim"
        hist_dim = "hist_dim"
        input_core_dims = [dims]
        if isinstance(bins, DataArray):
            bins_dims = [dim for dim in bins.dims if dim not in dims + ["plot_axis"]]
            assert len(bins_dims) == 1
            if "plot_axis" in bins.dims:
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
            else:
                edges_dim = bins_dims[0]
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
                "axes": np.arange(-len(dims), 0, 1),
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

    def kde(self, da, dims=None, circular=False, grid_len=512, **kwargs):
        """Compute kde on DataArray input."""
        dims = validate_dims(dims)
        grid, pdf, bw = apply_ufunc(
            self.array_class.kde,
            da,
            kwargs={
                "circular": circular,
                "grid_len": grid_len,
                "axes": np.arange(-len(dims), 0, 1),
                **kwargs,
            },
            input_core_dims=[dims],
            output_core_dims=[["kde_dim"], ["kde_dim"], []],
        )
        plot_axis = DataArray(["x", "y"], dims="plot_axis")
        out = concat((grid, pdf), dim=plot_axis)
        return out.assign_coords({"bw" if da.name is None else f"bw_{da.name}": bw})

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
            self.ess(da, method="bulk", dims=["chain", "draw"]),
            self.ess(da, method="tail", dims=["chain", "draw"]),
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
                f"current ESS average {var_indicator}at {ess_ave.item()} is preserved."
            )
            return 1
        if reduce_func == "min":
            return int(np.floor(n_samples / target_ess))
        return int(np.ceil(n_samples / target_ess))

    def thin(self, da, factor="auto", dims=None):
        """Perform thinning on DataArray input."""
        if factor == "auto" and dims is not None:
            warnings.warn("dims are ignored if factor is auto")

        if factor == "auto":
            factor = self.thin_factor(da)
            dims = "draw"

        elif isinstance(factor, (float | int)):
            if dims is None:
                dims = rcParams["data.sample_dims"]
            if not isinstance(dims, str):
                if len(dims) >= 2:
                    raise ValueError("dims must be of length 1")
                if len(dims) == 1:
                    dims = dims[0]

            factor = int(factor)
            if factor == 1:
                return da
            if factor < 1:
                raise ValueError("factor must be greater than 1")

        return da.sel({dims: slice(None, None, factor)})

    def pareto_min_ss(self, da, dims=None):
        """Compute the minimum effective sample size for all variables in the dataset."""
        dims, chain_axis, draw_axis = validate_dims_chain_draw_axis(dims)
        return apply_ufunc(
            self.array_class.pareto_min_ss,
            da,
            input_core_dims=[dims],
            output_core_dims=[[]],
            kwargs={"chain_axis": chain_axis, "draw_axis": draw_axis},
        )


dataarray_stats = BaseDataArray(array_class=array_stats)
