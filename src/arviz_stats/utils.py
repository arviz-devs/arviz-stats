"""ArviZ stats general utility functions."""

import warnings
from collections.abc import Hashable
from dataclasses import dataclass
from importlib import import_module
from typing import ClassVar

import numpy as np
import xarray as xr
from arviz_base import convert_to_dataset, rcParams
from xarray import DataArray

from arviz_stats.validate import validate_dims

__all__ = [
    "ELPDData",
    "ELPDDataLFO",
    "ELPDDataLOO",
    "ELPDDataLOOKFold",
    "ELPDDataLOOSubsample",
    "get_function",
    "get_log_likelihood",
]


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


BASE_HEADER_FMT = (
    "Computed from {n_samples} posterior samples and {n_points} observations log-likelihood matrix."
)
BASE_TABLE_FMT = """{{0:{0}}} Estimate       SE
{{scale}}_{{kind}} {{ic_value:8.2f}}  {{ic_se:7.2f}}
p_{{kind:{1}}} {{p_value:8.2f}}        -"""
WARNING_FMT = "\n\nThere has been a warning during the calculation. Please check the results."
POINTWISE_LOO_FMT = """------

Pareto k diagnostic values:
                         {{0:>{0}}} {{1:>6}}
(-Inf, {{8:.2f}}]   (good)     {{2:{0}d}} {{5:6.1f}}%
   ({{8:.2f}}, 1]   (bad)      {{3:{0}d}} {{6:6.1f}}%
    (1, Inf)   (very bad) {{4:{0}d}} {{7:6.1f}}%
"""
SCALE_DICT = {"deviance": "deviance", "log": "elpd", "negative_log": "-elpd"}


@dataclass(kw_only=True)
class ELPDData:  # pylint: disable=too-many-instance-attributes
    """Base container for expected log pointwise predictive density (ELPD) results.

    Every cross-validation function returns a subclass of this class.
    :class:`ELPDDataLOO` holds PSIS-LOO-CV results, :class:`ELPDDataLOOSubsample`
    holds subsampled PSIS-LOO-CV results, :class:`ELPDDataLOOKFold` holds k-fold
    cross-validation results and :class:`ELPDDataLFO` holds leave-future-out
    cross-validation results. The ``kind`` attribute names the estimator and the class
    says which additional attributes are available.

    Attributes
    ----------
    kind : str
        Name of the estimator, one of ``"loo"``, ``"loo_kfold"`` or ``"lfo_cv"``.
    elpd : float
        Expected log pointwise predictive density.
    se : float
        Standard error of ``elpd``.
    p : float
        Effective number of parameters.
    n_samples : int
        Number of posterior samples.
    n_data_points : int
        Number of observations.
    scale : str
        Scale of the estimate, ``"log"``, ``"negative_log"`` or ``"deviance"``.
    warning : bool
        True when a warning was raised during the computation.
    good_k : float
        Threshold on the Pareto k diagnostic above which estimates are unreliable.
    elpd_i : DataArray, optional
        Pointwise ELPD values, only when ``pointwise=True`` was requested.
    pareto_k : DataArray, optional
        Pointwise Pareto k diagnostics, only when ``pointwise=True`` was requested.
    """

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

    _display_kind: ClassVar[str | None] = None

    def _header(self):
        return BASE_HEADER_FMT.format(n_samples=self.n_samples, n_points=self.n_data_points)

    def _table(self):
        scale_str = SCALE_DICT[self.scale]
        display_kind = self._display_kind or self.kind
        padding = len(scale_str) + len(display_kind) + 1
        table = BASE_TABLE_FMT.format(padding, padding - 2)
        return table.format(
            "",
            kind=display_kind,
            scale=scale_str,
            ic_value=self.elpd,
            ic_se=self.se,
            p_value=self.p,
        )

    def _footer(self):
        return WARNING_FMT if self.warning else ""

    def __str__(self):
        """Print elpd data in a user friendly way."""
        return self._header() + "\n\n" + self._table() + self._footer()

    def __repr__(self):
        """Alias to ``__str__``."""
        return self.__str__()

    def __getitem__(self, key):
        """Define getitem magic method."""
        return getattr(self, key)

    def __setitem__(self, key, item):
        """Define setitem magic method."""
        setattr(self, key, item)


@dataclass(kw_only=True)
class ELPDDataLOO(ELPDData):
    """PSIS-LOO-CV results, returned by :func:`loo` and related functions.

    Also returned by :func:`loo_i`, :func:`loo_approximate_posterior`,
    :func:`loo_moment_match` and :func:`reloo`. Inherits the
    attributes of :class:`ELPDData`.

    Attributes
    ----------
    approx_posterior : bool
        True when the approximate posterior correction was applied.
    log_weights : DataArray, optional
        Smoothed log importance weights.
    log_jacobian : DataArray, optional
        Log-Jacobian adjustment for variable transformations.
    p_loo_i : DataArray, optional
        Pointwise effective number of parameters, set by ``reloo`` and
        ``loo_moment_match``.
    influence_pareto_k : DataArray, optional
        Pareto k values before moment matching, set by ``loo_moment_match``.
    n_eff_i : DataArray, optional
        Effective sample size per observation, set by ``loo_moment_match``.
    """

    kind: str = "loo"
    approx_posterior: bool = False
    log_weights: DataArray = None
    log_jacobian: DataArray = None
    p_loo_i: DataArray = None
    influence_pareto_k: DataArray = None
    n_eff_i: DataArray = None

    def _header(self):
        header = super()._header()
        if self.approx_posterior:
            header += "\nPosterior approximation correction used."
        return header

    def __str__(self):
        """Print elpd data followed by the Pareto k diagnostic table."""
        base = super().__str__()
        if self.pareto_k is None or self.good_k is None:
            return base
        bins = np.asarray([-np.inf, self.good_k, 1, np.inf])
        counts, *_ = np.histogram(self.pareto_k, bins=bins, density=False)
        extended = POINTWISE_LOO_FMT.format(max(4, len(str(np.max(counts)))))
        extended = extended.format(
            "Count",
            "Pct.",
            *[*counts, *(counts / np.sum(counts) * 100)],
            self.good_k,
        )
        return "\n".join([base, extended])


@dataclass(kw_only=True)
class ELPDDataLOOSubsample(ELPDDataLOO):
    """Subsampled PSIS-LOO-CV results, returned by :func:`loo_subsample`.

    Also returned by :func:`update_subsample`. Inherits the attributes of
    :class:`ELPDDataLOO`. Here ``log_weights`` holds a :class:`~xarray.Dataset` with one
    variable named after the log likelihood variable.

    Attributes
    ----------
    subsample_size : int
        Number of observations in the subsample.
    subsampling_se : float
        Standard error due to subsampling uncertainty only.
    loo_subsample_observations : ndarray
        Indices of the subsampled observations.
    elpd_loo_approx : DataArray
        Approximate pointwise ELPD for every observation.
    log_p : DataArray or ndarray, optional
        Log density of the target posterior.
    log_q : DataArray or ndarray, optional
        Log density of the proposal posterior.
    thin_factor : int or str, optional
        Thinning factor applied to the posterior draws, an integer or ``"auto"``.
    """

    subsample_size: int
    subsampling_se: float
    loo_subsample_observations: np.ndarray
    elpd_loo_approx: DataArray
    log_p: DataArray | np.ndarray = None
    log_q: DataArray | np.ndarray = None
    thin_factor: int | str = None

    def _header(self):
        header = (
            f"Computed from {self.n_samples} by {self.subsample_size} "
            f"subsampled log-likelihood\nvalues from {self.n_data_points} total observations."
        )
        if self.approx_posterior:
            header += " Posterior approximation correction used."
        return header

    def _table(self):
        scale_str = SCALE_DICT[self.scale]
        return (
            "         Estimate   SE subsampling SE\n"
            f"{scale_str}_{self.kind}  {self.elpd:8.1f} {self.se:4.1f} "
            f"           {self.subsampling_se:0.1f}\n"
            f"p_{self.kind}         {self.p:4.1f}\n"
        )


@dataclass(kw_only=True)
class ELPDDataLOOKFold(ELPDData):
    """K-fold cross-validation results, returned by :func:`loo_kfold`.

    Inherits the attributes of :class:`ELPDData`. ``good_k`` and ``pareto_k`` are always
    None because k-fold cross-validation does not use importance sampling.

    Attributes
    ----------
    n_folds : int
        Number of folds.
    p_kfold_i : DataArray, optional
        Pointwise effective number of parameters, only when ``pointwise=True``.
    fold_fits : dict, optional
        Fitted models for each fold, only when ``save_fits=True``.
    """

    kind: str = "loo_kfold"
    n_folds: int
    p_kfold_i: DataArray = None
    fold_fits: dict = None

    _display_kind: ClassVar[str] = "kfold"

    def _header(self):
        return f"Computed from {self.n_folds}-fold cross validation."


@dataclass(kw_only=True)
class ELPDDataLFO(ELPDData):
    """Leave-future-out cross-validation results, returned by :func:`lfo_cv`.

    Inherits the attributes of :class:`ELPDData`. ``n_data_points`` counts forecast origins.

    Attributes
    ----------
    forecast_horizon : int
        Number of steps ahead that were predicted.
    min_observations : int
        Minimum number of training observations.
    refits : ndarray
        Time indices at which the model was refit.
    n_refits : int
        Number of refits performed.
    p_lfo_i : DataArray, optional
        Pointwise effective number of parameters, only when ``pointwise=True``.
    """

    kind: str = "lfo_cv"
    forecast_horizon: int
    min_observations: int
    refits: np.ndarray
    n_refits: int
    p_lfo_i: DataArray = None

    _display_kind: ClassVar[str] = "lfo"

    def _header(self):
        origin_word = "origin" if self.n_data_points == 1 else "origins"
        header = (
            f"Computed from {self.n_data_points} forecast {origin_word} with "
            f"{self.forecast_horizon}-step-ahead predictions.\n"
            f"Minimum training observations: {self.min_observations}.\n"
        )
        if self.good_k is None:
            return header + "The model was fit at every forecast origin."
        refit_word = "refit" if self.n_refits == 1 else "refits"
        return header + (
            f"PSIS triggered {self.n_refits} additional exact {refit_word} "
            f"(k threshold: {self.good_k:.2f})."
        )


def _warn_non_unique_coords(xr_obj, dims_to_reduce):
    """Warn if coordinates in `xr_obj` for `dims_to_reduce` have duplicated values."""
    non_unique_coords = [
        dim
        for dim in xr_obj.dims
        if (dim not in dims_to_reduce)
        and (dim in xr_obj.coords)
        and (len(np.unique(xr_obj.coords[dim])) != xr_obj.sizes[dim])
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
        return getattr(data.azstats, name)(**all_kwargs, group=group)

    data = convert_to_dataset(data, group=group)

    data = data.azstats.filter_vars(var_names=var_names, filter_vars=filter_vars).dataset
    if coords is not None:
        data = data.sel(coords)

    _warn_non_unique_coords(data, dims)
    return getattr(data.azstats, name)(**all_kwargs)
