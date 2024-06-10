# pylint: disable=wildcard-import
"""ArviZ computational functions in numpy+numba.

Functions implemented in this folder can depend on NumPy, SciPy, Numba and should
support Dask via xarray.
"""

from arviz_stats.numba.array import array_stats

try:
    from arviz_stats.numba.dataarray import dataarray_stats
except ModuleNotFoundError:
    pass
