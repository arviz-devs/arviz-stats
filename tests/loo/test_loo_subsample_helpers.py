"""Test sub-sampling helper functions for PyMC models."""

# pylint: disable=no-self-use, redefined-outer-name
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import norm

from ..helpers import importorskip

azb = importorskip("arviz_base")
pm = importorskip("pymc")
xr = importorskip("xarray")

from arviz_stats import loo_subsample, update_subsample
from arviz_stats.loo import ll_from_pymc


def _build_normal_setup(*, n_chains=2, n_draws=60, n_obs=12, seed=0):
    rng = np.random.default_rng(seed)
    y_obs = rng.normal(1.0, 1.5, size=n_obs)
    mu_draws = rng.normal(1.0, 0.2, size=(n_chains, n_draws))
    sigma_draws = rng.lognormal(0.2, 0.15, size=(n_chains, n_draws))
    log_lik = norm.logpdf(y_obs[None, None, :], mu_draws[..., None], sigma_draws[..., None])
    idata = azb.from_dict(
        {
            "posterior": {"mu": mu_draws, "sigma": sigma_draws},
            "log_likelihood": {"y": log_lik},
            "observed_data": {"y": y_obs},
        }
    )
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 5)
        sigma = pm.HalfNormal("sigma", 2.0)
        pm.Normal("y", mu, sigma, observed=y_obs)
    return idata, model, y_obs


def _manual_ll_fn(observed, data):
    mu = data.posterior["mu"]
    sigma = data.posterior["sigma"]
    return -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * ((observed - mu) / sigma) ** 2


@pytest.fixture
def normal_setup():
    return _build_normal_setup()


class TestSubsampleLogLikFunctions:
    def test_lpd_dims_and_values(self, normal_setup):
        idata, model, y_obs = normal_setup
        log_lik_fn = ll_from_pymc(idata, model=model, var_name="y")
        observed = idata.observed_data["y"]
        result = log_lik_fn(observed, idata)
        assert isinstance(result, xr.DataArray)
        assert result.dims == ("chain", "draw", "y_dim_0")
        assert result.shape == (2, 60, y_obs.size)
        mu_draws = idata.posterior["mu"].values
        sigma_draws = idata.posterior["sigma"].values
        expected = norm.logpdf(y_obs[None, None, :], mu_draws[..., None], sigma_draws[..., None])
        assert_allclose(result.values, expected, rtol=1e-6)

    def test_plpd_point_estimate_values(self, normal_setup):
        idata, model, y_obs = normal_setup
        log_lik_fn = ll_from_pymc(idata, model=model, var_name="y")
        observed = idata.observed_data["y"]
        means = idata.posterior.dataset.mean(dim=["chain", "draw"])
        data_point = idata.copy()
        data_point.posterior = xr.Dataset(means.data_vars)
        result = log_lik_fn(observed, data_point)
        assert result.dims == ("y_dim_0",)
        expected = norm.logpdf(y_obs, means["mu"].item(), means["sigma"].item())
        assert_allclose(result.values, expected, rtol=1e-6)

    def test_loo_subsample_plpd_model_matches_manual(self, normal_setup):
        idata, model, _ = normal_setup
        observations = np.array([0, 3, 7, 9])
        loo_model = loo_subsample(
            idata,
            observations=observations,
            var_name="y",
            method="plpd",
            model=model,
            pointwise=True,
        )
        loo_manual = loo_subsample(
            idata,
            observations=observations,
            var_name="y",
            method="plpd",
            log_lik_fn=_manual_ll_fn,
            pointwise=True,
        )
        assert_allclose(loo_model.elpd, loo_manual.elpd, rtol=1e-6)
        assert_allclose(loo_model.se, loo_manual.se, rtol=1e-6)
        assert_allclose(
            loo_model.elpd_i.values[observations],
            loo_manual.elpd_i.values[observations],
            rtol=1e-6,
        )

    def test_loo_subsample_lpd_model_recomputes_approximation(self, normal_setup):
        idata, model, y_obs = normal_setup
        ramp = xr.DataArray(np.linspace(0.0, 4.0, y_obs.size), dims="y_dim_0")
        idata_bad = idata.copy()
        idata_bad["log_likelihood"] = idata.log_likelihood.dataset + ramp
        observations = np.array([1, 4, 6, 10])
        kwargs = {"observations": observations, "var_name": "y", "method": "lpd", "pointwise": True}
        loo_model = loo_subsample(idata_bad, model=model, **kwargs)
        loo_manual = loo_subsample(idata_bad, log_lik_fn=_manual_ll_fn, **kwargs)
        loo_default = loo_subsample(idata_bad, **kwargs)
        assert_allclose(
            loo_model.elpd_loo_approx.values, loo_manual.elpd_loo_approx.values, rtol=1e-6
        )
        assert not np.allclose(loo_model.elpd_loo_approx.values, loo_default.elpd_loo_approx.values)

    def test_update_subsample_with_model(self, normal_setup):
        idata, model, _ = normal_setup
        initial = loo_subsample(
            idata,
            observations=np.array([0, 2, 5]),
            var_name="y",
            method="plpd",
            model=model,
            pointwise=True,
        )
        updated = update_subsample(
            initial,
            idata,
            observations=np.array([1, 8]),
            method="plpd",
            model=model,
        )
        assert updated.subsample_size == 5
        assert np.isfinite(updated.elpd)
        assert np.isfinite(updated.se)
        assert np.sum(~np.isnan(updated.pareto_k.values)) == 5

    def test_update_subsample_lpd_model_matches_manual(self, normal_setup):
        idata, model, _ = normal_setup
        initial = loo_subsample(
            idata,
            observations=np.array([0, 2, 5]),
            var_name="y",
            method="lpd",
            model=model,
            pointwise=True,
        )
        updated_model = update_subsample(
            initial, idata, observations=np.array([1, 8]), method="lpd", model=model
        )
        updated_manual = update_subsample(
            initial, idata, observations=np.array([1, 8]), method="lpd", log_lik_fn=_manual_ll_fn
        )
        assert updated_model.subsample_size == 5
        assert_allclose(updated_model.elpd, updated_manual.elpd, rtol=1e-6)
        assert_allclose(updated_model.se, updated_manual.se, rtol=1e-6)

    def test_loo_subsample_model_log_false_raises(self, normal_setup):
        idata, model, _ = normal_setup
        with pytest.raises(ValueError, match="log must be True"):
            loo_subsample(
                idata,
                observations=4,
                var_name="y",
                method="plpd",
                model=model,
                log=False,
            )

    def test_update_subsample_model_log_false_raises(self, normal_setup):
        idata, model, _ = normal_setup
        initial = loo_subsample(
            idata,
            observations=np.array([0, 2, 5]),
            var_name="y",
            method="plpd",
            model=model,
            pointwise=True,
        )
        with pytest.raises(ValueError, match="log must be True"):
            update_subsample(
                initial, idata, observations=np.array([1, 8]), method="plpd", model=model, log=False
            )

    def test_update_subsample_plpd_without_loglik_or_model_raises(self, normal_setup):
        idata, model, _ = normal_setup
        initial = loo_subsample(
            idata,
            observations=np.array([0, 2, 5]),
            var_name="y",
            method="plpd",
            model=model,
            pointwise=True,
        )
        with pytest.raises(ValueError, match="log_lik_fn or model must be provided"):
            update_subsample(initial, idata, observations=np.array([1, 8]), method="plpd")

    def test_missing_posterior_var_raises(self, normal_setup):
        idata, model, _ = normal_setup
        log_lik_fn = ll_from_pymc(idata, model=model, var_name="y")
        observed = idata.observed_data["y"]
        data_missing = idata.copy()
        data_missing.posterior = xr.Dataset({"mu": idata.posterior.dataset["mu"]})
        with pytest.raises(KeyError, match="sigma"):
            log_lik_fn(observed, data_missing)

    def test_missing_posterior_var_with_shadowing_coord_raises(self, normal_setup):
        idata, model, _ = normal_setup
        log_lik_fn = ll_from_pymc(idata, model=model, var_name="y")
        observed = idata.observed_data["y"]
        data_shadow = idata.copy()
        data_shadow.posterior = xr.Dataset({"mu": idata.posterior.dataset["mu"]}).assign_coords(
            sigma=("sigma_dim", [1.0, 2.0])
        )
        with pytest.raises(KeyError, match="sigma"):
            log_lik_fn(observed, data_shadow)

    @pytest.mark.parametrize(
        "rv_name, var_name, match",
        [
            ("y", "not_a_var", "is not present in `idata.observed_data`"),
            ("y_renamed", "y", "does not match any observed random variable"),
        ],
    )
    def test_invalid_var_name_raises(self, normal_setup, rv_name, var_name, match):
        idata, _, y_obs = normal_setup
        with pm.Model() as model:
            mu = pm.Normal("mu", 0, 5)
            pm.Normal(rv_name, mu, 1.0, observed=y_obs)
        with pytest.raises(ValueError, match=match):
            ll_from_pymc(idata, model=model, var_name=var_name)

    def test_discrete_free_rv_raises(self, normal_setup):
        idata, _, _ = normal_setup
        with pm.Model() as model:
            p = pm.Poisson("p", mu=2.0)
            pm.Normal("y", p, 1.0, observed=[1.0, 2.0, 3.0])
        with pytest.raises(NotImplementedError, match="discrete"):
            ll_from_pymc(idata, model=model, var_name="y")

    def test_named_dims_transposed_posterior(self):
        rng = np.random.default_rng(0)
        y_obs = rng.normal(size=(3, 4))
        coords = {"g": ["a", "b", "c"], "t": list(range(4))}
        with pm.Model(coords=coords) as model:
            mu = pm.Normal("mu", 0, 5, dims=("g", "t"))
            pm.Normal("y", mu, 1.0, dims=("g", "t"), observed=y_obs)
        mu_draws = rng.normal(size=(2, 10, 3, 4))
        log_lik = norm.logpdf(y_obs[None, None], mu_draws, 1.0)
        idata = azb.from_dict(
            {
                "posterior": {"mu": mu_draws},
                "log_likelihood": {"y": log_lik},
                "observed_data": {"y": y_obs},
            },
            dims={"mu": ["g", "t"], "y": ["g", "t"]},
            coords=coords,
        )
        log_lik_fn = ll_from_pymc(idata, model=model, var_name="y")
        result = log_lik_fn(idata.observed_data["y"], idata)
        assert result.dims == ("chain", "draw", "g", "t")
        assert_allclose(result.values, log_lik, rtol=1e-6)
        idata_t = idata.copy()
        idata_t["posterior"] = idata.posterior.dataset.transpose("chain", "draw", "t", "g")
        result_t = log_lik_fn(idata_t.observed_data["y"], idata_t)
        assert_allclose(result_t.values, log_lik, rtol=1e-6)

    def test_reduced_likelihood_raises(self):
        rng = np.random.default_rng(0)
        y_obs = np.array([12, 8, 5])
        with pm.Model() as model:
            theta = pm.Dirichlet("theta", a=np.ones(3))
            pm.Multinomial("y", n=int(y_obs.sum()), p=theta, observed=y_obs)
        theta_draws = rng.dirichlet(np.ones(3), size=(2, 40))
        idata = azb.from_dict(
            {
                "posterior": {"theta": theta_draws},
                "observed_data": {"y": y_obs},
            }
        )
        log_lik_fn = ll_from_pymc(idata, model=model, var_name="y")
        with pytest.raises(ValueError, match="does not match the observed data shape"):
            log_lik_fn(idata.observed_data["y"], idata)
