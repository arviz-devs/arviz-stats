# pylint: disable=wildcard-import
"""ArviZ computational functions in NumPy.

Functions implemented in this folder should only depend on NumPy and SciPy.
"""
from arviz_stats.base.array import array_stats

try:
    from arviz_stats.base.dataarray import dataarray_stats
except ModuleNotFoundError:
    pass
