"""Class with dataarray functions.

"dataarray" functions take :class:`xarray.DataArray` as inputs.
"""

import warnings

import numpy as np
from arviz_base import rcParams
from xarray import DataArray, apply_ufunc, concat

from arviz_stats.base.array import array_stats


class BaseDataArray:
    """Class with numpy+scipy only functions that take DataArray inputs."""

    def __init__(self, array_class=None):
        self.array_class = array_stats if array_class is None else array_class

    def eti(self, da, prob=None, dims=None, method="linear"):
        """Compute eti on DataArray input."""
        if dims is None:
            dims = rcParams["data.sample_dims"]
        if prob is None:
            prob = rcParams["stats.ci_prob"]
        elif not 1 >= prob > 0:
            raise ValueError("The value of hdi_prob should be in the interval (0, 1]")

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
        if dims is None:
            dims = rcParams["data.sample_dims"]
        if isinstance(dims, str):
            dims = [dims]
        if prob is None:
            prob = rcParams["stats.ci_prob"]

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
        if dims is None:
            dims = rcParams["data.sample_dims"]
        if len(dims) != 2:
            raise ValueError("dims must be of length 2")
        return apply_ufunc(
            self.array_class.ess,
            da,
            input_core_dims=[dims],
            output_core_dims=[[]],
            kwargs={"method": method, "relative": relative, "prob": prob},
        )

    def compute_ranks(self, da, dims=None, relative=False):
        """Compute ranks on DataArray input."""
        if dims is None:
            dims = rcParams["data.sample_dims"]
        if isinstance(dims, str):
            dims = [dims]
        return apply_ufunc(
            self.array_class.compute_ranks,
            da,
            input_core_dims=[dims],
            output_core_dims=[dims],
            kwargs={"relative": relative},
        )

    def rhat(self, da, dims=None, method="bulk"):
        """Compute rhat on DataArray input."""
        if dims is None:
            dims = rcParams["data.sample_dims"]
        if len(dims) != 2:
            raise ValueError("dims must be of length 2")
        return apply_ufunc(
            self.array_class.rhat,
            da,
            input_core_dims=[dims],
            output_core_dims=[[]],
            kwargs={"method": method},
        )

    def mcse(self, da, dims=None, method="mean", prob=None):
        """Compute mcse on DataArray input."""
        if dims is None:
            dims = rcParams["data.sample_dims"]
        if len(dims) != 2:
            raise ValueError("dims must be of length 2")
        return apply_ufunc(
            self.array_class.mcse,
            da,
            input_core_dims=[dims],
            output_core_dims=[[]],
            kwargs={"method": method, "prob": prob},
        )

    # pylint: disable=redefined-builtin
    def histogram(self, da, dims=None, bins=None, range=None, weights=None, density=None):
        """Compute histogram on DataArray input."""
        if dims is None:
            dims = rcParams["data.sample_dims"]
        if isinstance(dims, str):
            dims = [dims]
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
        if dims is None:
            dims = rcParams["data.sample_dims"]
        if isinstance(dims, str):
            dims = [dims]
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

    def thin(self, da, factor="auto", dims=None):
        """Perform thinning on DataArray input."""
        if factor == "auto" and dims is not None:
            warnings.warn("dims are ignored if factor is auto")

        if factor == "auto":
            n_samples = da.sizes["chain"] * da.sizes["draw"]
            ess_ave = np.minimum(
                self.ess(da, method="bulk", dims=["chain", "draw"]),
                self.ess(da, method="tail", dims=["chain", "draw"]),
            ).mean()
            factor = int(np.ceil(n_samples / ess_ave))
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


dataarray_stats = BaseDataArray(array_class=array_stats)
