"""Tests for array interface base functions with xarray comparison / optional dependencies."""

# pylint: disable=redefined-outer-name, no-self-use, protected-access
import pytest

from ..helpers import importorskip

np = importorskip("numpy")
azb = importorskip("arviz_base")
einstats = importorskip("xarray_einstats")
xr = importorskip("xarray")

from numpy.testing import assert_allclose

from arviz_stats import loo, loo_approximate_posterior, loo_influence, loo_pit, loo_score
from arviz_stats.base.array import BaseArray
from arviz_stats.loo.loo_helper import _get_r_eff, _prepare_loo_inputs
from arviz_stats.utils import get_log_likelihood_dataset


@pytest.fixture
def array_stats():
    return BaseArray()


class TestLOOXarray:
    def test_loo_matches_xarray(self, array_stats, centered_eight):
        log_lik = get_log_likelihood_dataset(centered_eight, var_names="obs")["obs"]
        loo_inputs = _prepare_loo_inputs(centered_eight, var_name="obs")
        reff = _get_r_eff(centered_eight, loo_inputs.n_samples)

        loo_xr = loo(centered_eight, pointwise=True, var_name="obs")
        elpd_i_array, pareto_k_array, p_loo_i_array = array_stats.loo(
            log_lik.values, chain_axis=0, draw_axis=1, r_eff=reff
        )

        lppd_xr = einstats.stats.logsumexp(
            log_lik, b=1 / loo_inputs.n_samples, dims=["chain", "draw"]
        )
        p_loo_i_xr = lppd_xr - loo_xr.elpd_i

        assert_allclose(elpd_i_array, loo_xr.elpd_i.values, rtol=1e-10)
        assert_allclose(pareto_k_array, loo_xr.pareto_k.values, rtol=1e-10)
        assert_allclose(p_loo_i_array, p_loo_i_xr.values, rtol=1e-10)


class TestLOOApproximatePosteriorXarray:
    def test_loo_approximate_posterior_matches_xarray(self, array_stats, centered_eight, rng):
        log_lik = get_log_likelihood_dataset(centered_eight, var_names="obs")["obs"]

        log_p = xr.DataArray(
            rng.normal(0, 1, size=(log_lik.chain.size, log_lik.draw.size)),
            dims=["chain", "draw"],
            coords={"chain": log_lik.chain, "draw": log_lik.draw},
        )
        log_q = xr.DataArray(
            rng.normal(-1, 1, size=(log_lik.chain.size, log_lik.draw.size)),
            dims=["chain", "draw"],
            coords={"chain": log_lik.chain, "draw": log_lik.draw},
        )

        loo_approx_xr = loo_approximate_posterior(
            centered_eight, log_p=log_p, log_q=log_q, var_name="obs", pointwise=True
        )

        n_obs = log_lik.shape[2]
        n_samples = log_lik.chain.size * log_lik.draw.size
        log_p_broadcast = np.repeat(log_p.values[..., np.newaxis], n_obs, axis=2)
        log_q_broadcast = np.repeat(log_q.values[..., np.newaxis], n_obs, axis=2)

        elpd_i_array, pareto_k_array, p_loo_i_array = array_stats.loo_approximate_posterior(
            log_lik.values, log_p_broadcast, log_q_broadcast, chain_axis=0, draw_axis=1
        )

        lppd_xr = einstats.stats.logsumexp(log_lik, b=1 / n_samples, dims=["chain", "draw"])
        p_loo_i_xr = lppd_xr - loo_approx_xr.elpd_i

        assert_allclose(elpd_i_array, loo_approx_xr.elpd_i.values, rtol=1e-10)
        assert_allclose(pareto_k_array, loo_approx_xr.pareto_k.values, rtol=1e-10)
        assert_allclose(p_loo_i_array, p_loo_i_xr.values, rtol=1e-10)


class TestLOOScoreXarray:
    @pytest.mark.parametrize("kind", ["crps", "scrps"])
    def test_loo_score_basic(self, array_stats, centered_eight, kind):
        log_lik = get_log_likelihood_dataset(centered_eight, var_names="obs")["obs"]
        n_samples = log_lik.chain.size * log_lik.draw.size
        reff = _get_r_eff(centered_eight, n_samples)

        log_weights_ds, _ = log_lik.azstats.psislw(r_eff=reff)
        log_weights_xr = log_weights_ds.transpose("chain", "draw", "school")

        y_pred = centered_eight.posterior_predictive["obs"].values[:, :, :1]
        y_obs = centered_eight.observed_data["obs"].values[:1]
        log_weights = log_weights_xr.values[:, :, :1]

        scores = array_stats.loo_score(
            y_pred,
            y_obs,
            log_weights,
            kind=kind,
            chain_axis=0,
            draw_axis=1,
        )

        assert scores.shape == (1,)
        assert np.isfinite(scores[0])

    def test_loo_score_multiple_obs(self, array_stats, centered_eight):
        log_lik = get_log_likelihood_dataset(centered_eight, var_names="obs")["obs"]
        n_samples = log_lik.chain.size * log_lik.draw.size
        reff = _get_r_eff(centered_eight, n_samples)

        log_weights_ds, _ = log_lik.azstats.psislw(r_eff=reff)
        log_weights_xr = log_weights_ds.transpose("chain", "draw", "school")

        y_pred = centered_eight.posterior_predictive["obs"].values
        y_obs = centered_eight.observed_data["obs"].values
        log_weights = log_weights_xr.values

        scores = array_stats.loo_score(
            y_pred,
            y_obs,
            log_weights,
            kind="crps",
            chain_axis=0,
            draw_axis=1,
        )

        assert scores.shape == (8,)
        assert np.all(np.isfinite(scores))

    def test_loo_score_chain_axis_none(self, array_stats, centered_eight):
        log_lik = get_log_likelihood_dataset(centered_eight, var_names="obs")["obs"]
        n_samples = log_lik.chain.size * log_lik.draw.size
        reff = _get_r_eff(centered_eight, n_samples)

        log_weights_ds, _ = log_lik.azstats.psislw(r_eff=reff)
        log_weights_xr = log_weights_ds.transpose("chain", "draw", "school")

        y_pred = centered_eight.posterior_predictive["obs"].values
        y_obs = centered_eight.observed_data["obs"].values
        log_weights = log_weights_xr.values

        y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
        log_weights_flat = log_weights.reshape(-1, log_weights.shape[-1])

        scores = array_stats.loo_score(
            y_pred_flat,
            y_obs,
            log_weights_flat,
            kind="crps",
            chain_axis=None,
            draw_axis=0,
        )

        assert scores.shape == (8,)
        assert np.all(np.isfinite(scores))

    def test_loo_score_diff_axes(self, array_stats, centered_eight):
        log_lik = get_log_likelihood_dataset(centered_eight, var_names="obs")["obs"]
        n_samples = log_lik.chain.size * log_lik.draw.size
        reff = _get_r_eff(centered_eight, n_samples)

        log_weights_ds, _ = log_lik.azstats.psislw(r_eff=reff)
        log_weights_xr = log_weights_ds.transpose("chain", "draw", "school")

        y_pred = centered_eight.posterior_predictive["obs"].values
        y_obs = centered_eight.observed_data["obs"].values
        log_weights = log_weights_xr.values

        y_pred_reorder = np.transpose(y_pred, (2, 0, 1))
        log_weights_reorder = np.transpose(log_weights, (2, 0, 1))

        scores = array_stats.loo_score(
            y_pred_reorder,
            y_obs,
            log_weights_reorder,
            kind="crps",
            chain_axis=1,
            draw_axis=2,
        )

        assert scores.shape == (8,)
        assert np.all(np.isfinite(scores))

    @pytest.mark.parametrize("kind", ["crps", "scrps"])
    def test_loo_score_matches_xarray(self, array_stats, centered_eight, kind):
        log_lik = get_log_likelihood_dataset(centered_eight, var_names="obs")["obs"]
        n_samples = log_lik.chain.size * log_lik.draw.size
        reff = _get_r_eff(centered_eight, n_samples)

        log_weights_ds, _ = log_lik.azstats.psislw(r_eff=reff)
        log_weights_xr = log_weights_ds.transpose("chain", "draw", "school")

        loo_score_xr = loo_score(centered_eight, kind=kind, pointwise=True)

        y_pred = centered_eight.posterior_predictive["obs"].values
        y_obs = centered_eight.observed_data["obs"].values
        log_weights = log_weights_xr.values

        scores_array = array_stats.loo_score(
            y_pred,
            y_obs,
            log_weights,
            kind=kind,
            chain_axis=0,
            draw_axis=1,
        )

        assert_allclose(scores_array, loo_score_xr.pointwise.values, rtol=1e-10)


class TestLooPitXarray:
    def test_loo_pit_basic(self, array_stats, centered_eight):
        log_lik = get_log_likelihood_dataset(centered_eight, var_names="obs")["obs"]
        n_samples = log_lik.chain.size * log_lik.draw.size
        reff = _get_r_eff(centered_eight, n_samples)

        log_weights_ds, _ = log_lik.azstats.psislw(r_eff=reff)
        log_weights_xr = log_weights_ds.transpose("chain", "draw", "school")

        y_pred = centered_eight.posterior_predictive["obs"].values[:, :, :1]
        y_obs = centered_eight.observed_data["obs"].values[:1]
        log_weights = log_weights_xr.values[:, :, :1]

        pit_values = array_stats.loo_pit(
            y_pred,
            y_obs,
            log_weights,
            chain_axis=0,
            draw_axis=1,
        )

        assert pit_values.shape == (1,)
        assert np.all(pit_values >= 0)
        assert np.all(pit_values <= 1)

    def test_loo_pit_multiple_obs(self, array_stats, centered_eight):
        log_lik = get_log_likelihood_dataset(centered_eight, var_names="obs")["obs"]
        n_samples = log_lik.chain.size * log_lik.draw.size
        reff = _get_r_eff(centered_eight, n_samples)

        log_weights_ds, _ = log_lik.azstats.psislw(r_eff=reff)
        log_weights_xr = log_weights_ds.transpose("chain", "draw", "school")

        y_pred = centered_eight.posterior_predictive["obs"].values
        y_obs = centered_eight.observed_data["obs"].values
        log_weights = log_weights_xr.values

        pit_values = array_stats.loo_pit(
            y_pred,
            y_obs,
            log_weights,
            chain_axis=0,
            draw_axis=1,
        )

        assert pit_values.shape == (8,)
        assert np.all(pit_values >= 0)
        assert np.all(pit_values <= 1)

    def test_loo_pit_chain_axis_none(self, array_stats, centered_eight):
        log_lik = get_log_likelihood_dataset(centered_eight, var_names="obs")["obs"]
        n_samples = log_lik.chain.size * log_lik.draw.size
        reff = _get_r_eff(centered_eight, n_samples)

        log_weights_ds, _ = log_lik.azstats.psislw(r_eff=reff)
        log_weights_xr = log_weights_ds.transpose("chain", "draw", "school")

        y_pred = centered_eight.posterior_predictive["obs"].values
        y_obs = centered_eight.observed_data["obs"].values
        log_weights = log_weights_xr.values

        y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
        log_weights_flat = log_weights.reshape(-1, log_weights.shape[-1])

        pit_values = array_stats.loo_pit(
            y_pred_flat,
            y_obs,
            log_weights_flat,
            chain_axis=None,
            draw_axis=0,
        )

        assert pit_values.shape == (8,)
        assert np.all(pit_values >= 0)
        assert np.all(pit_values <= 1)

    def test_loo_pit_diff_axes(self, array_stats, centered_eight):
        log_lik = get_log_likelihood_dataset(centered_eight, var_names="obs")["obs"]
        n_samples = log_lik.chain.size * log_lik.draw.size
        reff = _get_r_eff(centered_eight, n_samples)

        log_weights_ds, _ = log_lik.azstats.psislw(r_eff=reff)
        log_weights_xr = log_weights_ds.transpose("chain", "draw", "school")

        y_pred = centered_eight.posterior_predictive["obs"].values
        y_obs = centered_eight.observed_data["obs"].values
        log_weights = log_weights_xr.values

        y_pred_reorder = np.transpose(y_pred, (2, 0, 1))
        log_weights_reorder = np.transpose(log_weights, (2, 0, 1))

        pit_values = array_stats.loo_pit(
            y_pred_reorder,
            y_obs,
            log_weights_reorder,
            chain_axis=1,
            draw_axis=2,
        )

        assert pit_values.shape == (8,)
        assert np.all(pit_values >= 0)
        assert np.all(pit_values <= 1)

    def test_loo_pit_matches_xarray(self, array_stats, centered_eight):
        log_lik = get_log_likelihood_dataset(centered_eight, var_names="obs")["obs"]
        n_samples = log_lik.chain.size * log_lik.draw.size
        reff = _get_r_eff(centered_eight, n_samples)

        log_weights_ds, _ = log_lik.azstats.psislw(r_eff=reff)
        log_weights_xr = log_weights_ds.transpose("chain", "draw", "school")

        loo_pit_xr = loo_pit(centered_eight)

        y_pred = centered_eight.posterior_predictive["obs"].values
        y_obs = centered_eight.observed_data["obs"].values
        log_weights = log_weights_xr.values

        pit_values_array = array_stats.loo_pit(
            y_pred,
            y_obs,
            log_weights,
            chain_axis=0,
            draw_axis=1,
        )

        assert_allclose(pit_values_array, loo_pit_xr["obs"].values, rtol=1e-10)

    def test_loo_pit_single_observation(self, array_stats, centered_eight):
        log_lik = get_log_likelihood_dataset(centered_eight, var_names="obs")["obs"]
        n_samples = log_lik.chain.size * log_lik.draw.size
        reff = _get_r_eff(centered_eight, n_samples)

        log_weights_ds, _ = log_lik.azstats.psislw(r_eff=reff)
        log_weights_xr = log_weights_ds.transpose("chain", "draw", "school")

        y_pred = centered_eight.posterior_predictive["obs"].values[:, :, :1]
        y_obs = centered_eight.observed_data["obs"].values[:1]
        log_weights = log_weights_xr.values[:, :, :1]

        pit_values = array_stats.loo_pit(y_pred, y_obs, log_weights, chain_axis=0, draw_axis=1)

        assert pit_values.shape == (1,)
        assert 0 <= pit_values[0] <= 1


class TestLooExpectationXarray:
    def test_loo_expectation(self, array_stats, centered_eight):
        log_lik = get_log_likelihood_dataset(centered_eight, var_names="obs")["obs"]
        n_samples = log_lik.chain.size * log_lik.draw.size
        reff = _get_r_eff(centered_eight, n_samples)

        log_weights_ds, _ = log_lik.azstats.psislw(r_eff=reff)
        log_weights_xr = log_weights_ds.transpose("chain", "draw", "school")

        y_pred = centered_eight.posterior_predictive["obs"].values[:, :, :1]
        log_weights = log_weights_xr.values[:, :, :1]

        expectation = array_stats.loo_expectation(
            y_pred,
            log_weights,
            kind="mean",
            chain_axis=0,
            draw_axis=1,
        )

        assert expectation.shape == (1,)
        assert np.all(np.isfinite(expectation))

    def test_loo_expectation_multiple_obs(self, array_stats, centered_eight):
        log_lik = get_log_likelihood_dataset(centered_eight, var_names="obs")["obs"]
        n_samples = log_lik.chain.size * log_lik.draw.size
        reff = _get_r_eff(centered_eight, n_samples)

        log_weights_ds, _ = log_lik.azstats.psislw(r_eff=reff)
        log_weights_xr = log_weights_ds.transpose("chain", "draw", "school")

        y_pred = centered_eight.posterior_predictive["obs"].values
        log_weights = log_weights_xr.values

        expectation = array_stats.loo_expectation(
            y_pred,
            log_weights,
            kind="mean",
            chain_axis=0,
            draw_axis=1,
        )

        assert expectation.shape == (8,)
        assert np.all(np.isfinite(expectation))

    @pytest.mark.parametrize("kind", ["mean", "median", "var", "sd"])
    def test_loo_expectation_kinds(self, array_stats, centered_eight, kind):
        log_lik = get_log_likelihood_dataset(centered_eight, var_names="obs")["obs"]
        n_samples = log_lik.chain.size * log_lik.draw.size
        reff = _get_r_eff(centered_eight, n_samples)

        log_weights_ds, _ = log_lik.azstats.psislw(r_eff=reff)
        log_weights_xr = log_weights_ds.transpose("chain", "draw", "school")

        y_pred = centered_eight.posterior_predictive["obs"].values
        log_weights = log_weights_xr.values

        expectation = array_stats.loo_expectation(
            y_pred,
            log_weights,
            kind=kind,
            chain_axis=0,
            draw_axis=1,
        )

        assert expectation.shape == (8,)
        assert np.all(np.isfinite(expectation))
        if kind in ("var", "sd"):
            assert np.all(expectation >= 0)

    def test_loo_expectation_chain_axis_none(self, array_stats, centered_eight):
        log_lik = get_log_likelihood_dataset(centered_eight, var_names="obs")["obs"]
        n_samples = log_lik.chain.size * log_lik.draw.size
        reff = _get_r_eff(centered_eight, n_samples)

        log_weights_ds, _ = log_lik.azstats.psislw(r_eff=reff)
        log_weights_xr = log_weights_ds.transpose("chain", "draw", "school")

        y_pred = centered_eight.posterior_predictive["obs"].values
        log_weights = log_weights_xr.values

        y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
        log_weights_flat = log_weights.reshape(-1, log_weights.shape[-1])

        expectation = array_stats.loo_expectation(
            y_pred_flat,
            log_weights_flat,
            kind="mean",
            chain_axis=None,
            draw_axis=0,
        )

        assert expectation.shape == (8,)
        assert np.all(np.isfinite(expectation))

    def test_loo_expectation_diff_axes(self, array_stats, centered_eight):
        log_lik = get_log_likelihood_dataset(centered_eight, var_names="obs")["obs"]
        n_samples = log_lik.chain.size * log_lik.draw.size
        reff = _get_r_eff(centered_eight, n_samples)

        log_weights_ds, _ = log_lik.azstats.psislw(r_eff=reff)
        log_weights_xr = log_weights_ds.transpose("chain", "draw", "school")

        y_pred = centered_eight.posterior_predictive["obs"].values
        log_weights = log_weights_xr.values

        y_pred_reorder = np.transpose(y_pred, (2, 0, 1))
        log_weights_reorder = np.transpose(log_weights, (2, 0, 1))

        expectation = array_stats.loo_expectation(
            y_pred_reorder,
            log_weights_reorder,
            kind="mean",
            chain_axis=1,
            draw_axis=2,
        )

        assert expectation.shape == (8,)
        assert np.all(np.isfinite(expectation))


class TestLooQuantileXarray:
    def test_loo_quantile_basic(self, array_stats, centered_eight):
        log_lik = get_log_likelihood_dataset(centered_eight, var_names="obs")["obs"]
        n_samples = log_lik.chain.size * log_lik.draw.size
        reff = _get_r_eff(centered_eight, n_samples)

        log_weights_ds, _ = log_lik.azstats.psislw(r_eff=reff)
        log_weights_xr = log_weights_ds.transpose("chain", "draw", "school")

        y_pred = centered_eight.posterior_predictive["obs"].values[:, :, :1]
        log_weights = log_weights_xr.values[:, :, :1]

        quantile = array_stats.loo_quantile(
            y_pred,
            log_weights,
            prob=0.5,
            chain_axis=0,
            draw_axis=1,
        )

        assert quantile.shape == (1,)
        assert np.all(np.isfinite(quantile))

    def test_loo_quantile_multiple_obs(self, array_stats, centered_eight):
        log_lik = get_log_likelihood_dataset(centered_eight, var_names="obs")["obs"]
        n_samples = log_lik.chain.size * log_lik.draw.size
        reff = _get_r_eff(centered_eight, n_samples)

        log_weights_ds, _ = log_lik.azstats.psislw(r_eff=reff)
        log_weights_xr = log_weights_ds.transpose("chain", "draw", "school")

        y_pred = centered_eight.posterior_predictive["obs"].values
        log_weights = log_weights_xr.values

        quantile = array_stats.loo_quantile(
            y_pred,
            log_weights,
            prob=0.5,
            chain_axis=0,
            draw_axis=1,
        )

        assert quantile.shape == (8,)
        assert np.all(np.isfinite(quantile))

    @pytest.mark.parametrize("prob", [0.1, 0.25, 0.5, 0.75, 0.9])
    def test_loo_quantile_probs(self, array_stats, centered_eight, prob):
        log_lik = get_log_likelihood_dataset(centered_eight, var_names="obs")["obs"]
        n_samples = log_lik.chain.size * log_lik.draw.size
        reff = _get_r_eff(centered_eight, n_samples)

        log_weights_ds, _ = log_lik.azstats.psislw(r_eff=reff)
        log_weights_xr = log_weights_ds.transpose("chain", "draw", "school")

        y_pred = centered_eight.posterior_predictive["obs"].values
        log_weights = log_weights_xr.values

        quantile = array_stats.loo_quantile(
            y_pred,
            log_weights,
            prob=prob,
            chain_axis=0,
            draw_axis=1,
        )

        assert quantile.shape == (8,)
        assert np.all(np.isfinite(quantile))

    def test_loo_quantile_ordering(self, array_stats, centered_eight):
        log_lik = get_log_likelihood_dataset(centered_eight, var_names="obs")["obs"]
        n_samples = log_lik.chain.size * log_lik.draw.size
        reff = _get_r_eff(centered_eight, n_samples)

        log_weights_ds, _ = log_lik.azstats.psislw(r_eff=reff)
        log_weights_xr = log_weights_ds.transpose("chain", "draw", "school")

        y_pred = centered_eight.posterior_predictive["obs"].values
        log_weights = log_weights_xr.values

        q25 = array_stats.loo_quantile(y_pred, log_weights, prob=0.25, chain_axis=0, draw_axis=1)
        q50 = array_stats.loo_quantile(y_pred, log_weights, prob=0.50, chain_axis=0, draw_axis=1)
        q75 = array_stats.loo_quantile(y_pred, log_weights, prob=0.75, chain_axis=0, draw_axis=1)

        assert np.all(q25 <= q50)
        assert np.all(q50 <= q75)

    def test_loo_quantile_chain_axis_none(self, array_stats, centered_eight):
        log_lik = get_log_likelihood_dataset(centered_eight, var_names="obs")["obs"]
        n_samples = log_lik.chain.size * log_lik.draw.size
        reff = _get_r_eff(centered_eight, n_samples)

        log_weights_ds, _ = log_lik.azstats.psislw(r_eff=reff)
        log_weights_xr = log_weights_ds.transpose("chain", "draw", "school")

        y_pred = centered_eight.posterior_predictive["obs"].values
        log_weights = log_weights_xr.values

        y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
        log_weights_flat = log_weights.reshape(-1, log_weights.shape[-1])

        quantile = array_stats.loo_quantile(
            y_pred_flat,
            log_weights_flat,
            prob=0.5,
            chain_axis=None,
            draw_axis=0,
        )

        assert quantile.shape == (8,)
        assert np.all(np.isfinite(quantile))

    def test_loo_quantile_diff_axes(self, array_stats, centered_eight):
        log_lik = get_log_likelihood_dataset(centered_eight, var_names="obs")["obs"]
        n_samples = log_lik.chain.size * log_lik.draw.size
        reff = _get_r_eff(centered_eight, n_samples)

        log_weights_ds, _ = log_lik.azstats.psislw(r_eff=reff)
        log_weights_xr = log_weights_ds.transpose("chain", "draw", "school")

        y_pred = centered_eight.posterior_predictive["obs"].values
        log_weights = log_weights_xr.values

        y_pred_reorder = np.transpose(y_pred, (2, 0, 1))
        log_weights_reorder = np.transpose(log_weights, (2, 0, 1))

        quantile = array_stats.loo_quantile(
            y_pred_reorder,
            log_weights_reorder,
            prob=0.5,
            chain_axis=1,
            draw_axis=2,
        )

        assert quantile.shape == (8,)
        assert np.all(np.isfinite(quantile))


class TestLooInfluenceXarray:
    @pytest.fixture
    def log_weights(self, centered_eight):
        log_lik = get_log_likelihood_dataset(centered_eight, var_names="obs")["obs"]
        reff = _get_r_eff(centered_eight, log_lik.chain.size * log_lik.draw.size)
        log_weights, _ = log_lik.azstats.psislw(r_eff=reff)
        return log_weights.transpose("school", "chain", "draw").values

    @pytest.mark.parametrize(
        "kind, kwargs",
        [
            ("mean", {}),
            ("mean", {"standardize": False}),
            ("median", {}),
            ("sd", {}),
            ("var", {}),
            ("quantile", {"probs": [0.25, 0.75]}),
            ("octiles", {}),
        ],
    )
    def test_loo_influence_matches_xarray(
        self, array_stats, centered_eight, log_weights, kind, kwargs
    ):
        y_pred = centered_eight.posterior_predictive["obs"].transpose("school", "chain", "draw")

        influence = array_stats.loo_influence(y_pred.values, log_weights, kind=kind, **kwargs)

        expected, _ = loo_influence(centered_eight, kind=kind, **kwargs)
        if isinstance(expected, xr.Dataset):
            expected = expected["obs"]
        assert influence.shape == (8,)
        assert_allclose(influence, expected.transpose("school").values)

    def test_loo_influence_posterior(self, array_stats, centered_eight, log_weights):
        mu = centered_eight.posterior["mu"].transpose("chain", "draw").values

        influence = array_stats.loo_influence(mu, log_weights, kind="median")

        expected, _ = loo_influence(
            centered_eight, kind="median", var_names="mu", group="posterior"
        )
        assert influence.shape == (8,)
        assert_allclose(influence, expected["mu"].transpose("school").values)
