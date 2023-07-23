"""ArviZ stats general utility functions."""
import warnings
from copy import copy as _copy
from copy import deepcopy as _deepcopy
from dataclasses import dataclass
from importlib import import_module

import numpy as np
from arviz_base import rcParams
from xarray import DataArray

__all__ = ["ELPDData", "get_function", "get_log_likelihood"]


def get_function(func_name):
    """Get a function from arviz_stats.

    Attempts to import the provided function from the module indicated in the rcParam
    ``stats.module``, and if it fails, it imports it from ``arviz_stats.base``.

    Parameters
    ----------
    func_name : str
        Name of the function to be imported and returned

    Returns
    -------
    callable
    """
    preferred_module = import_module(rcParams["stats.module"])
    if hasattr(preferred_module, func_name):
        return getattr(preferred_module, func_name)
    base_module = import_module("arviz_stats.base")
    return getattr(base_module, func_name)


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


BASE_FMT = """Computed from {{n_samples}} posterior samples and \
{{n_points}} observations log-likelihood matrix.

{{0:{0}}} Estimate       SE
{{scale}}_{{kind}} {{ic_value:8.2f}}  {{ic_se:7.2f}}
p_{{kind:{1}}} {{p_value:8.2f}}        -"""
POINTWISE_LOO_FMT = """------

Pareto k diagnostic values:
                         {{0:>{0}}} {{1:>6}}
(-Inf, 0.5]   (good)     {{2:{0}d}} {{6:6.1f}}%
 (0.5, 0.7]   (ok)       {{3:{0}d}} {{7:6.1f}}%
   (0.7, 1]   (bad)      {{4:{0}d}} {{8:6.1f}}%
   (1, Inf)   (very bad) {{5:{0}d}} {{9:6.1f}}%
"""
SCALE_DICT = {"deviance": "deviance", "log": "elpd", "negative_log": "-elpd"}


@dataclass
class ELPDData:  # pylint: disable=too-many-ancestors
    """Class to contain the data from elpd information criterion like waic or loo."""

    kind: str
    elpd: float
    se: float
    p: float
    n_samples: int
    n_data_points: int
    scale: str
    warning: bool
    elpd_i: DataArray = None
    pareto_k: DataArray = None

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

        if self.warning:
            base += "\n\nThere has been a warning during the calculation. Please check the results."

        if kind == "loo" and self.pareto_k is not None:
            bins = np.asarray([-np.Inf, 0.5, 0.7, 1, np.Inf])
            counts, *_ = np.histogram(self.pareto_k, bins=bins, density=False)
            extended = POINTWISE_LOO_FMT.format(max(4, len(str(np.max(counts)))))
            extended = extended.format(
                "Count", "Pct.", *[*counts, *(counts / np.sum(counts) * 100)]
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
