"""Class with dataarray functions.

"dataarray" functions take :class:`xarray.DataArray` as inputs.
"""

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

    def kde(self, da, dims=None, circular=False, grid_len=512, **kwargs):
        """Compute kde on DataArray input."""
        if dims is None:
            dims = rcParams["data.sample_dims"]
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


dataarray_stats = BaseDataArray(array_class=array_stats)
