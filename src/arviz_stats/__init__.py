# pylint: disable=wildcard-import
"""Statistical tests and diagnostics for ArviZ."""
try:
    import numba

    from .numba import *
except ModuleNotFoundError:
    from .base import *

from .utils import *
