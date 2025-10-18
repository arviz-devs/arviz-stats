# pylint: disable=redefined-outer-name, unused-import, no-self-use
# ruff: noqa: F811, F401
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal

from ..helpers import importorskip

azb = importorskip("arviz_base")
xr = importorskip("xarray")
xr_einstats = importorskip("xarray_einstats")
xr_stats = xr_einstats.stats

from arviz_stats.numba.diagnostics import (
    _backtransform_ranks,
    _ess,
    _ess_bulk,
    _ess_mean,
    _rhat,
    _rhat_rank,
    _split_chains,
    _z_fold,
    _z_scale,
    autocorr,
    autocov,
    ess,
    geyer,
    irfft,
    rfft,
    rhat,
)

GOOD_RHAT = 1.1


class TestBacktransformRanks:
    def test_backtransform_ranks_basic(self, centered_eight):
        da = centered_eight.posterior["mu"]
        rank = xr_stats.rankdata(da, dims=("chain", "draw"), method="average")
        result = _backtransform_ranks(rank)
        assert result.shape == da.shape
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_backtransform_ranks_range(self):
        rng = np.random.default_rng(42)
        data = rng.uniform(1, 100, size=(4, 100))
        da = xr.DataArray(data, dims=["chain", "draw"])
        rank = xr_stats.rankdata(da, dims=("chain", "draw"), method="average")
        result = _backtransform_ranks(rank)
        assert result.min() >= 0
        assert result.max() <= 1


class TestZScale:
    def test_z_scale_basic(self, centered_eight):
        da = centered_eight.posterior["mu"]
        result = _z_scale(da)
        assert result.shape == da.shape
        assert result.dims == da.dims

    def test_z_scale_preserves_coords(self, centered_eight):
        da = centered_eight.posterior["mu"]
        result = _z_scale(da)
        assert "chain" in result.dims
        assert "draw" in result.dims


class TestSplitChains:
    def test_split_chains_shape(self, centered_eight):
        da = centered_eight.posterior["mu"]
        result = _split_chains(da)
        n_chains, n_draws = da.shape
        assert result.shape == (n_chains * 2, n_draws // 2)

    def test_split_chains_dims(self, centered_eight):
        da = centered_eight.posterior["mu"]
        result = _split_chains(da)
        assert "chain" in result.dims
        assert "draw" in result.dims


class TestZFold:
    def test_z_fold_basic(self, centered_eight):
        da = centered_eight.posterior["mu"]
        result = _z_fold(da)
        assert result.shape == da.shape
        assert result.dims == da.dims

    def test_z_fold_preserves_coords(self, centered_eight):
        da = centered_eight.posterior["mu"]
        result = _z_fold(da)
        assert "chain" in result.dims
        assert "draw" in result.dims

    def test_z_fold_with_symmetric_data(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, size=(4, 100))
        da = xr.DataArray(data, dims=["chain", "draw"])
        result = _z_fold(da)
        assert result.shape == (4, 100)


class TestRhat:
    def test_rhat_basic(self, centered_eight):
        da = centered_eight.posterior["mu"]
        result = _rhat(da)
        assert isinstance(result, xr.DataArray)
        assert result.shape == ()

    def test_rhat_value_range(self, centered_eight):
        da = centered_eight.posterior["mu"]
        result = _rhat(da)
        assert 0.95 < result.values < 1.15

    def test_rhat_bad_convergence(self):
        data = np.vstack([np.random.normal(0, 1, (2, 100)), np.random.normal(10, 1, (2, 100))])
        da = xr.DataArray(data, dims=["chain", "draw"])
        result = _rhat(da)
        assert result.values > 1.2


class TestRhatRank:
    def test_rhat_rank_basic(self, centered_eight):
        da = centered_eight.posterior["mu"]
        result = _rhat_rank(da)
        assert isinstance(result, xr.DataArray)

    def test_rhat_rank_vs_rhat(self, centered_eight):
        da = centered_eight.posterior["mu"]
        rhat_val = _rhat(da)
        rhat_rank_val = _rhat_rank(da)
        assert 0.9 < rhat_val.values < 1.2
        assert 0.9 < rhat_rank_val.values < 1.2


class TestRhatFunction:
    def test_rhat_dataarray(self, centered_eight):
        da = centered_eight.posterior["mu"]
        result = rhat(da, method="rank")
        assert isinstance(result, xr.DataArray)

    def test_rhat_dataset(self, centered_eight):
        ds = centered_eight.posterior.dataset
        result = rhat(ds, method="rank")
        assert isinstance(result, xr.Dataset)
        assert "mu" in result
        assert "tau" in result

    def test_rhat_datatree(self, centered_eight):
        result = rhat(centered_eight, group="posterior", method="rank")
        assert isinstance(result, xr.Dataset)
        assert "mu" in result
        assert "tau" in result

    def test_rhat_identity_method(self, centered_eight):
        da = centered_eight.posterior["mu"]
        result = rhat(da, method="identity")
        assert isinstance(result, xr.DataArray)

    def test_rhat_invalid_method(self, centered_eight):
        da = centered_eight.posterior["mu"]
        with pytest.raises(ValueError, match="method not recognized"):
            rhat(da, method="invalid")

    def test_rhat_invalid_type(self):
        with pytest.raises(TypeError, match="Input type not recognized"):
            rhat([1, 2, 3])


class TestFFT:
    def test_rfft_basic(self, centered_eight):
        da = centered_eight.posterior["mu"]
        result = rfft(da, dim="draw")
        assert "freq_draw" in result.dims

    def test_irfft_basic(self, centered_eight):
        da = centered_eight.posterior["mu"]
        fft_result = rfft(da, dim="draw")
        n = len(da.draw)
        result = irfft(fft_result, dim="freq_draw", n=n)
        assert "draw" in result.dims
        assert len(result.draw) == n

    def test_rfft_irfft_roundtrip(self, centered_eight):
        da = centered_eight.posterior["mu"]
        n = len(da.draw)
        fft_result = rfft(da, dim="draw", n=n)
        reconstructed = irfft(fft_result, dim="freq_draw", n=n)
        assert_array_almost_equal(da.values, reconstructed.values, decimal=10)


class TestAutocov:
    def test_autocov_basic(self, centered_eight):
        da = centered_eight.posterior["mu"]
        result = autocov(da)
        assert result.shape == da.shape
        assert "draw" in result.dims

    def test_autocov_coords_preserved(self, centered_eight):
        da = centered_eight.posterior["mu"]
        result = autocov(da)
        assert np.array_equal(result.draw.values, da.draw.values)


class TestAutocorr:
    def test_autocorr_basic(self, centered_eight):
        da = centered_eight.posterior["mu"]
        result = autocorr(da)
        assert result.shape == da.shape

    def test_autocorr_lag_zero_is_one(self, centered_eight):
        da = centered_eight.posterior["mu"]
        result = autocorr(da)
        assert_allclose(result.isel(draw=0).values, 1.0, atol=1e-10)

    def test_autocorr_range(self, centered_eight):
        da = centered_eight.posterior["mu"]
        result = autocorr(da)
        assert np.all(result.values >= -1.1)
        assert np.all(result.values <= 1.1)


class TestGeyer:
    def test_geyer_basic(self):
        rng = np.random.default_rng(42)
        acov = rng.normal(size=(4, 100))
        chain_mean_term = 0.1
        tau_hat = np.zeros(1)
        geyer(acov, chain_mean_term, tau_hat)
        assert tau_hat[0] > 0

    @pytest.mark.parametrize("n_chains", [2, 4, 8])
    def test_geyer_different_chain_counts(self, n_chains):
        rng = np.random.default_rng(42)
        acov = rng.normal(size=(n_chains, 100))
        chain_mean_term = 0.1
        tau_hat = np.zeros(1)
        geyer(acov, chain_mean_term, tau_hat)
        assert not np.isnan(tau_hat[0])


class TestESS:
    def test_ess_basic(self, centered_eight):
        da = centered_eight.posterior["mu"]
        result = _ess(da)
        assert isinstance(result, xr.DataArray)
        assert result > 0

    def test_ess_relative(self, centered_eight):
        da = centered_eight.posterior["mu"]
        ess_abs = _ess(da, relative=False)
        ess_rel = _ess(da, relative=True)
        n_samples = da.size
        assert_allclose(ess_rel.values, ess_abs.values / n_samples, rtol=0.01)

    def test_ess_constant_data(self):
        data = np.ones((4, 100))
        da = xr.DataArray(data, dims=["chain", "draw"])
        result = _ess(da)
        assert_allclose(result.values, 400, rtol=0.01)


class TestESSMean:
    def test_ess_mean_basic(self, centered_eight):
        da = centered_eight.posterior["mu"]
        result = _ess_mean(da)
        assert isinstance(result, xr.DataArray)
        assert result > 0


class TestESSBulk:
    def test_ess_bulk_basic(self, centered_eight):
        da = centered_eight.posterior["mu"]
        result = _ess_bulk(da)
        assert isinstance(result, xr.DataArray)
        assert result > 0


class TestESSFunction:
    def test_ess_dataarray(self, centered_eight):
        da = centered_eight.posterior["mu"]
        result = ess(da, method="bulk")
        assert isinstance(result, xr.DataArray)

    def test_ess_dataset(self, centered_eight):
        ds = centered_eight.posterior.dataset
        result = ess(ds, method="bulk")
        assert isinstance(result, xr.Dataset)
        assert "mu" in result
        assert "tau" in result

    def test_ess_datatree(self, centered_eight):
        result = ess(centered_eight, group="posterior", method="bulk")
        assert isinstance(result, xr.Dataset)
        assert "mu" in result
        assert "tau" in result

    def test_ess_mean_method(self, centered_eight):
        da = centered_eight.posterior["mu"]
        result = ess(da, method="mean")
        assert isinstance(result, xr.DataArray)
        assert result > 0

    def test_ess_invalid_method(self, centered_eight):
        da = centered_eight.posterior["mu"]
        with pytest.raises(ValueError, match="method not recognized"):
            ess(da, method="invalid")

    def test_ess_invalid_type(self):
        with pytest.raises(TypeError, match="Input type not recognized"):
            ess([1, 2, 3])


class TestIntegration:
    def test_rhat_convergence_threshold(self, centered_eight):
        da = centered_eight.posterior["mu"]
        result = rhat(da, method="rank")
        assert result.values < GOOD_RHAT

    def test_ess_reasonable_value(self, centered_eight):
        da = centered_eight.posterior["mu"]
        result = ess(da, method="bulk")
        total_samples = da.size
        assert 0 < result.values < total_samples

    def test_rhat_ess_consistency(self, centered_eight):
        da = centered_eight.posterior["mu"]
        rhat_val = rhat(da, method="rank")
        ess_val = ess(da, method="bulk")
        assert rhat_val.values < 1.1
        assert ess_val.values > 100
