# File generated with docstub

from copy import deepcopy

import numpy as np
import xarray as xr
from arviz_base import rcParams
from xarray import DataArray
from xarray_einstats.stats import logsumexp

from arviz_stats.loo.loo import loo
from arviz_stats.loo.loo_helper import _prepare_loo_inputs
from arviz_stats.loo.wrapper import SamplingWrapper
from arviz_stats.utils import ELPDData

from . import SamplingWrapper

__all__ = ["reloo"]

def reloo(
    wrapper: SamplingWrapper,
    loo_orig: ELPDData | None = ...,
    var_name: str | None = ...,
    log_weights: DataArray | ELPDData | None = ...,
    k_threshold: float = ...,
    pointwise: bool | None = ...,
) -> ELPDData: ...
