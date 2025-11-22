"""ArviZ stats general utility functions."""

import warnings
from collections.abc import Hashable
from dataclasses import dataclass
from importlib import import_module

import numpy as np
import xarray as xr
from arviz_base import convert_to_dataset, rcParams
from xarray import DataArray

from arviz_stats.validate import validate_dims

__all__ = ["ELPDData", "get_function", "get_log_likelihood"]


def get_function(func_name):
    """Get a function from arviz_stats.

    Attempts to import the provided function from dataarray class in the module indicated
    in the rcParam ``stats.module``, and if it fails, it imports it from ``arviz_stats.base``.

    Parameters
    ----------
    func_name : str
        Name of the function to be imported and returned

    Returns
    -------
    callable
    """
    module_name = rcParams["stats.module"]
    if isinstance(module_name, str):
        preferred_module = import_module(f"arviz_stats.{module_name}")
    else:
        preferred_module = module_name
    if hasattr(preferred_module, "dataarray_stats"):
        preferred_module = preferred_module.dataarray_stats
    if not hasattr(preferred_module, func_name):
        raise KeyError(f"Requested function '{func_name}' is not available in '{preferred_module}'")
    return getattr(preferred_module, func_name)


def get_array_function(func_name):
    """Get a function from arviz_stats' array layer.

    Attempts to import the provided function from array class in the module indicated
    in the rcParam ``stats.module``, and if it fails, it imports it from ``arviz_stats.base``.

    Parameters
    ----------
    func_name : str
        Name of the function to be imported and returned

    Returns
    -------
    callable
    """
    module_name = rcParams["stats.module"]
    if isinstance(module_name, str):
        preferred_module = import_module(f"arviz_stats.{module_name}")
    else:
        preferred_module = module_name
    if hasattr(preferred_module, "array_stats"):
        preferred_module = preferred_module.array_stats
    if not hasattr(preferred_module, func_name):
        raise KeyError(f"Requested function '{func_name}' is not available in '{preferred_module}'")
    return getattr(preferred_module, func_name)


def get_log_likelihood(idata, var_name=None):
    """Retrieve the log likelihood dataarray of a given variable."""
    if (
        not hasattr(idata, "log_likelihood")
        and hasattr(idata, "sample_stats")
        and hasattr(idata.sample_stats, "log_likelihood")
    ):
        warnings.warn(
            "Storing the log_likelihood in sample_stats groups has been deprecated",
            DeprecationWarning,
        )
        return idata.sample_stats.log_likelihood
    if not hasattr(idata, "log_likelihood"):
        raise TypeError("log likelihood not found in inference data object")
    if var_name is None:
        var_names = list(idata.log_likelihood.data_vars)
        if len(var_names) > 1:
            raise TypeError(
                f"Found several log likelihood arrays {var_names}, var_name cannot be None"
            )
        return idata.log_likelihood[var_names[0]]
    try:
        log_likelihood = idata.log_likelihood[var_name]
    except KeyError as err:
        raise TypeError(f"No log likelihood data named {var_name} found") from err
    return log_likelihood


# get_log_likelihood and get_log_prior functions should be somewhere else
def get_log_likelihood_dataset(idata, var_names=None):
    """Retrieve the log likelihood dataarray of a given variable."""
    if (
        not hasattr(idata, "log_likelihood")
        and hasattr(idata, "sample_stats")
        and hasattr(idata.sample_stats, "log_likelihood")
    ):
        warnings.warn(
            "Storing the log_likelihood in sample_stats groups has been deprecated",
            DeprecationWarning,
        )
        log_lik_ds = idata.sample_stats.ds[["log_likelihood"]]
    if not hasattr(idata, "log_likelihood"):
        raise TypeError("log likelihood not found in inference data object")
    log_lik_ds = idata.log_likelihood.ds
    if var_names is None:
        return log_lik_ds
    if isinstance(var_names, Hashable):
        return log_lik_ds[[var_names]]
    return log_lik_ds[var_names]


def get_log_likelihood_dataarray(data, var_name=None):
    log_lik_ds = get_log_likelihood_dataset(data)
    if var_name is None:
        var_names = list(log_lik_ds.data_vars)
        if len(var_names) > 1:
            raise TypeError(
                f"Found several log likelihood arrays {var_names}, var_name cannot be None"
            )
        return log_lik_ds[var_names[0]]

    try:
        log_likelihood = log_lik_ds[var_name]
    except KeyError as err:
        raise TypeError(f"No log likelihood data named {var_name} found") from err
    return log_likelihood


def get_log_prior(idata, var_names=None):
    """Retrieve the log prior dataarray of a given variable."""
    if not hasattr(idata, "log_prior"):
        raise TypeError("log prior not found in inference data object")
    if var_names is None:
        return idata.log_prior.ds
    if isinstance(var_names, Hashable):
        return idata.log_prior.ds[[var_names]]
    return idata.log_prior.ds[var_names]


BASE_FMT = """Computed from {{n_samples}} posterior samples and \
{{n_points}} observations log-likelihood matrix.

{{0:{0}}} Estimate       SE
{{scale}}_{{kind}} {{ic_value:8.2f}}  {{ic_se:7.2f}}
p_{{kind:{1}}} {{p_value:8.2f}}        -"""
POINTWISE_LOO_FMT = """------

Pareto k diagnostic values:
                         {{0:>{0}}} {{1:>6}}
(-Inf, {{8:.2f}}]   (good)     {{2:{0}d}} {{5:6.1f}}%
   ({{8:.2f}}, 1]   (bad)      {{3:{0}d}} {{6:6.1f}}%
    (1, Inf)   (very bad) {{4:{0}d}} {{7:6.1f}}%
"""
SCALE_DICT = {"deviance": "deviance", "log": "elpd", "negative_log": "-elpd"}


@dataclass
class ELPDData:  # pylint: disable=too-many-ancestors, too-many-instance-attributes
    """Class to contain the data from elpd information criterion like waic or loo."""

    kind: str
    elpd: float
    se: float
    p: float
    n_samples: int
    n_data_points: int
    scale: str
    warning: bool
    good_k: float
    elpd_i: DataArray = None
    pareto_k: DataArray = None
    approx_posterior: bool = False
    subsampling_se: float = None
    subsample_size: int = None
    log_p: object = None
    log_q: object = None
    thin_factor: object = None
    log_weights: DataArray = None
    n_folds: int = None
    loo_subsample_observations: np.ndarray = None
    elpd_loo_approx: DataArray = None
    log_jacobian: DataArray = None
    influence_pareto_k: DataArray = None
    n_eff_i: DataArray = None

    def __str__(self):
        """Print elpd data in a user friendly way."""
        kind = self.kind
        scale_str = SCALE_DICT[self["scale"]]

        # loo_kfold
        if kind == "loo_kfold" and self.n_folds is not None:
            display_kind = "kfold"
            padding = len(scale_str) + len(display_kind) + 1
            base = f"Computed from {self.n_folds}-fold cross validation.\n\n"
            base += f"{{0:{padding}}} Estimate       SE\n"
            base += f"{scale_str}_{display_kind} {{ic_value:8.2f}}  {{ic_se:7.2f}}\n"
            base += f"p_{display_kind:{padding-2}} {{p_value:8.2f}}        -"
            base = base.format(
                "",
                ic_value=self.elpd,
                ic_se=self.se,
                p_value=self.p,
            )

            return base

        padding = len(scale_str) + len(kind) + 1

        # loo_subsample
        if self.subsample_size:
            base = (
                f"Computed from {self.n_samples} by {self.subsample_size} "
                f"subsampled log-likelihood\n"
            )
            base += f"values from {self.n_data_points} total observations.\n\n"
            base += "         Estimate   SE subsampling SE\n"
            base += (
                f"{scale_str}_{kind}  {self.elpd:8.1f} {self.se:4.1f} "
                f"           {self.subsampling_se:0.1f}\n"
            )
            base += f"p_{kind}         {self.p:4.1f}\n"
            if self.approx_posterior:
                header, table = base.split("\n\n", 1)
                base = header + " Posterior approximation correction used.\n\n" + table
        else:
            base = BASE_FMT.format(padding, padding - 2)
            base = base.format(
                "",
                kind=kind,
                scale=scale_str,
                n_samples=self.n_samples,
                n_points=self.n_data_points,
                ic_value=self.elpd,
                ic_se=self.se,
                p_value=self.p,
            )

            if self.approx_posterior:
                header, table = base.split("\n\n", 1)
                base = header + "\nPosterior approximation correction used.\n\n" + table

        if self.warning:
            base += "\n\nThere has been a warning during the calculation. Please check the results."

        # loo
        if kind == "loo" and self.pareto_k is not None:
            bins = np.asarray([-np.inf, self.good_k, 1, np.inf])
            counts, *_ = np.histogram(self.pareto_k, bins=bins, density=False)
            extended = POINTWISE_LOO_FMT.format(max(4, len(str(np.max(counts)))))
            extended = extended.format(
                "Count",
                "Pct.",
                *[*counts, *(counts / np.sum(counts) * 100)],
                self.good_k,
            )
            base = "\n".join([base, extended])

        return base

    def __repr__(self):
        """Alias to ``__str__``."""
        return self.__str__()

    def __getitem__(self, key):
        """Define getitem magic method."""
        return getattr(self, key)

    def __setitem__(self, key, item):
        """Define setitem magic method."""
        setattr(self, key, item)


def _warn_non_unique_coords(xr_obj, dims_to_reduce):
    """Warn if coordinates in `xr_obj` for `dims_to_reduce` have duplicated values."""
    non_unique_coords = [
        dim
        for dim in xr_obj.dims
        if len(np.unique(xr_obj.coords[dim])) != xr_obj.sizes[dim] and dim not in dims_to_reduce
    ]
    if non_unique_coords:
        warnings.warn(
            f"Dimensions {non_unique_coords} don't have unique coordinates which might result "
            "in unexpected results. Either use 'groupby' beforehand or define unique coordiates "
            "to remove this warning"
        )


def _apply_multi_input_function(
    name,
    data,
    dims,
    dims_arg,
    *,
    group,
    var_names,
    filter_vars,
    coords,
    **kwargs,
):
    all_kwargs = {dims_arg: dims} | kwargs
    if isinstance(data, list | tuple | np.ndarray):
        data = np.array(data)
        return get_array_function(name)(data, **kwargs)

    dims = validate_dims(dims)
    if isinstance(data, xr.core.groupby.DataArrayGroupBy | xr.core.groupby.DatasetGroupBy):
        # Make sure the grouped dimension is added as one of the dimensions to be reduced
        dims = list(set(dims).union(data.group1d.dims))

        if isinstance(data, xr.core.groupby.DataArrayGroupBy):

            def reduce_func(xr_obj):
                if coords is not None:
                    xr_obj = xr_obj.sel(coords)
                return getattr(xr_obj.azstats, name)(**{dims_arg: dims}, **kwargs)
        else:

            def reduce_func(xr_obj):
                xr_obj = xr_obj.azstats.filter_vars(
                    var_names=var_names, filter_vars=filter_vars
                ).dataset
                if coords is not None:
                    xr_obj = xr_obj.sel(coords)
                return getattr(xr_obj.azstats, name)(**{dims_arg: dims}, **kwargs)

        return data.map(reduce_func)

    if isinstance(data, xr.DataArray):
        if coords is not None:
            data = data.sel(coords)
        _warn_non_unique_coords(data, dims)
        return getattr(data.azstats, name)(**all_kwargs)

    if isinstance(data, xr.DataTree):
        data = data.azstats.filter_vars(
            group=group, var_names=var_names, filter_vars=filter_vars
        ).datatree
        if coords is not None:
            data = data.sel(coords)
        _warn_non_unique_coords(data, dims)
        return getattr(data.azstats, name)(**all_kwargs)

    data = convert_to_dataset(data, group=group)

    data = data.azstats.filter_vars(var_names=var_names, filter_vars=filter_vars).dataset
    if coords is not None:
        data = data.sel(coords)

    _warn_non_unique_coords(data, dims)
    return getattr(data.azstats, name)(**all_kwargs)
