# pylint: disable=redefined-outer-name, unused-import, no-self-use
# ruff: noqa: F811, F401
import numpy as np
import pytest

from ..helpers import centered_eight, importorskip

xr = importorskip("xarray")

from arviz_stats.numba.intervals import _quantile, eti, quantile


class TestQuantileGuvectorized:
    def test_quantile_basic(self):
        ary = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        q = np.array([0.5])
        result = np.zeros(1)
        _quantile(ary, q, result)
        assert result.shape == (1,)

    def test_quantile_multiple(self):
        ary = np.arange(100, dtype=float)
        q = np.array([0.25, 0.5, 0.75])
        result = np.zeros(3)
        _quantile(ary, q, result)
        assert result.shape == (3,)


class TestQuantileFunction:
    def test_quantile_dataarray(self, centered_eight):
        da = centered_eight.posterior["mu"]
        q = np.array([0.25, 0.5, 0.75])
        result = quantile(da, q, dim="draw")
        assert result is not None

    def test_quantile_multiple_dims(self, centered_eight):
        da = centered_eight.posterior["theta"]
        q = np.array([0.5])
        result = quantile(da, q, dim=["chain", "draw"])
        assert result is not None


class TestETI:
    def test_eti_basic(self, centered_eight):
        da = centered_eight.posterior["mu"]
        result = eti(da, prob=0.94, dims=["chain", "draw"])
        assert result is not None

    def test_eti_invalid_prob(self, centered_eight):
        da = centered_eight.posterior["mu"]
        with pytest.raises(ValueError, match="should be in the interval"):
            eti(da, prob=1.5, dims=["chain", "draw"])

    def test_eti_default_prob(self, centered_eight):
        da = centered_eight.posterior["mu"]
        result = eti(da, dims=["chain", "draw"])
        assert result is not None
