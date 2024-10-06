# pylint: disable=wildcard-import
"""Statistical computation and diagnostics for ArviZ."""

try:
    from arviz_stats.utils import *
    from arviz_stats.accessors import *
    from arviz_stats.psens import psens, psens_summary

except ModuleNotFoundError:
    pass
