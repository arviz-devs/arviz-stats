"""Class with dataarray numbified functions."""

from arviz_stats.base.dataarray import BaseDataArray
from arviz_stats.numba.array import array_stats


class NumbaDataArray(BaseDataArray):
    """DataArray compatible functions that use numba."""


dataarray_stats = NumbaDataArray(array_class=array_stats)
