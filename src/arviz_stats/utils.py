"""ArviZ stats general utility functions."""

import warnings
from collections.abc import Hashable
from dataclasses import dataclass
from importlib import import_module

import numpy as np
from arviz_base import rcParams
from xarray import DataArray

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

    def __str__(self):
        """Print elpd data in a user friendly way."""
        kind = self.kind
        scale_str = SCALE_DICT[self["scale"]]
        padding = len(scale_str) + len(kind) + 1
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

        if kind == "loo" and self.pareto_k is not None:
            bins = bins = np.asarray([-np.inf, self.good_k, 1, np.inf])
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


def round_num(value, precision):
    """Round a number to a given precision.

    Parameters
    ----------
    value : float, numpy array or DataArray
        The value to round. If array assumes it contains a single value.
    precision : int or str
        If an integer, specifies decimal places. If a string ending in 'g',
        specifies significant digits. Use "None" for no rounding.
    """
    if isinstance(value, np.ndarray | DataArray):
        value = value.item()

    if precision is not None:
        if isinstance(precision, int):
            return round(value, precision)

        if isinstance(precision, str) and precision.endswith("g"):
            sig_digits = int(precision[:-1])
            return round(value, sig_digits - int(np.floor(np.log10(abs(value)))) - 1)

    return value
