# File generated with docstub

from xarray import DataArray

from arviz_stats.loo.wrapper import SamplingWrapper
from arviz_stats.utils import ELPDData

from .wrapper import SamplingWrapper

__all__ = ["reloo"]

def reloo(
    wrapper: SamplingWrapper,
    loo_orig: ELPDData | None = ...,
    var_name: str | None = ...,
    log_weights: DataArray | ELPDData | None = ...,
    k_threshold: float = ...,
    pointwise: bool | None = ...,
) -> ELPDData: ...
