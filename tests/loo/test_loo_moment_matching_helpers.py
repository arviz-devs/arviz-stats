"""Test moment matching helper functions for PyMC models."""

# pylint: disable=no-self-use, redefined-outer-name
import numpy as np
import pytest
from scipy.stats import multinomial, poisson

from ..helpers import importorskip

azb = importorskip("arviz_base")
pm = importorskip("pymc")
xr = importorskip("xarray")

from arviz_stats.loo import mm_from_pymc


def _build_poisson_outlier_idata(*, n_chains=2, n_draws=80, seed=0):
    rng = np.random.default_rng(seed)
    y_obs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.0])
    beta_draws = rng.lognormal(0, 1, size=(n_chains, n_draws))
    log_lik = poisson.logpmf(y_obs[None, None, :], beta_draws[:, :, None])
    idata = azb.from_dict(
        {
            "posterior": {"beta": beta_draws},
            "log_likelihood": {"y": log_lik},
            "observed_data": {"y": y_obs},
        }
    )
    return idata, y_obs


def _build_poisson_outlier_model(y_obs):
    with pm.Model() as model:
        beta = pm.HalfNormal("beta", 1.0)
        pm.Poisson("y", beta, observed=y_obs)
    return model


def _build_dirichlet_multinomial_setup(*, n_chains=2, n_draws=40, seed=0):
    rng = np.random.default_rng(seed)
    y_obs = np.array([12, 8, 5])
    with pm.Model() as model:
        theta = pm.Dirichlet("theta", a=np.ones(3))
        pm.Multinomial("y", n=int(y_obs.sum()), p=theta, observed=y_obs)
    theta_draws = rng.dirichlet(np.ones(3), size=(n_chains, n_draws))
    log_lik = multinomial.logpmf(y_obs, n=int(y_obs.sum()), p=theta_draws)
    idata = azb.from_dict(
        {
            "posterior": {"theta": theta_draws},
            "log_likelihood": {"y": log_lik},
            "observed_data": {"y": y_obs},
        }
    )
    return idata, model


def _build_unconstrained_only_idata(*, n_chains=2, n_draws=80, seed=0):
    rng = np.random.default_rng(seed)
    y_obs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.0])
    beta_log__draws = rng.normal(0, 1, size=(n_chains, n_draws))
    beta = np.exp(beta_log__draws)
    log_lik = poisson.logpmf(y_obs[None, None, :], beta[:, :, None])
    idata = azb.from_dict(
        {
            "posterior": {"beta_log__": beta_log__draws},
            "log_likelihood": {"y": log_lik},
            "observed_data": {"y": y_obs},
        }
    )
    return idata, y_obs


def _compiled_logp(model):
    return model.compile_fn(
        inputs=model.value_vars,
        outs=[model.logp(jacobian=True, sum=True)],
        point_fn=False,
    )


@pytest.fixture
def poisson_setup():
    idata, y_obs = _build_poisson_outlier_idata()
    return idata, _build_poisson_outlier_model(y_obs), y_obs


@pytest.fixture
def dirichlet_setup():
    return _build_dirichlet_multinomial_setup()


class TestMomentMatchFunctions:
    def test_returns_expected_types_and_dims(self, poisson_setup):
        idata, model, _ = poisson_setup
        log_prob_fn, log_lik_i_fn, upars = mm_from_pymc(idata, model=model, var_name="y")
        assert isinstance(upars, xr.DataArray)
        assert upars.dims == ("chain", "draw", "unconstrained_parameter")
        assert upars.sizes["unconstrained_parameter"] == 1
        assert list(upars.coords["unconstrained_parameter"].values) == ["beta_log__"]
        out = log_prob_fn(upars)
        assert isinstance(out, xr.DataArray)
        assert out.dims == ("chain", "draw")
        assert out.shape == (2, 80)
        for i in (0, 5, 10):
            r = log_lik_i_fn(upars, i)
            assert r.dims == ("chain", "draw")
            assert r.shape == (2, 80)

    def test_log_prob_upars_matches_model_logp(self, poisson_setup):
        idata, model, _ = poisson_setup
        log_prob_fn, _, upars = mm_from_pymc(idata, model=model, var_name="y")
        reference = _compiled_logp(model)
        beta_log = upars.values
        for ch, dr in [(0, 0), (0, 7), (1, 12), (1, 79)]:
            unc = float(beta_log[ch, dr, 0])
            actual = reference(beta_log__=np.array(unc))[0]
            pred = float(
                log_prob_fn(upars.isel(chain=slice(ch, ch + 1), draw=slice(dr, dr + 1))).values[
                    0, 0
                ]
            )
            np.testing.assert_allclose(pred, actual)

    def test_log_lik_i_matches_manual_elemwise(self, poisson_setup):
        idata, model, y_obs = poisson_setup
        _, log_lik_i_fn, upars = mm_from_pymc(idata, model=model, var_name="y")
        beta_log_draws = upars.values[:, :, 0]
        beta_draws = np.exp(beta_log_draws)
        n_obs = y_obs.size
        for i in (0, 5, n_obs - 1):
            pred = log_lik_i_fn(upars, i).values
            actual = poisson.logpmf(y_obs[i], beta_draws)
            np.testing.assert_allclose(pred, actual)

    def test_log_lik_i_matches_arviz_log_likelihood_group(self):
        y_obs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.0])
        with pm.Model() as model:
            beta = pm.HalfNormal("beta", 1.0)
            pm.Poisson("y", beta, observed=y_obs)
        idata = pm.sample(
            model=model,
            draws=50,
            tune=50,
            chains=2,
            random_seed=3,
            progressbar=False,
        )
        pm.compute_log_likelihood(idata, model=model)
        _, log_lik_i_fn, upars = mm_from_pymc(idata, model=model, var_name="y")
        obs_dims = [d for d in idata.observed_data["y"].dims if d not in ("chain", "draw")]
        ll = idata.log_likelihood["y"]
        for i in (0, 5, 10):
            pred = log_lik_i_fn(upars, i).values
            sel = dict.fromkeys(obs_dims, i)
            actual = ll.isel(sel).values
            np.testing.assert_allclose(pred, actual, rtol=1e-6)

    def test_unconstrained_only_posterior_used_directly(self, poisson_setup):
        _, model, _ = poisson_setup
        idata, _ = _build_unconstrained_only_idata()
        _, _, upars = mm_from_pymc(idata, model=model, var_name="y")
        bg = idata.posterior["beta_log__"]
        np.testing.assert_allclose(upars.values[..., 0], bg.values)

    def test_constrained_and_unconstrained_sizes_differ(self, dirichlet_setup):
        idata, model = dirichlet_setup
        theta_draws = idata.posterior["theta"].values
        log_prob_fn, log_lik_i_fn, upars = mm_from_pymc(idata, model=model, var_name="y")
        assert upars.dims == ("chain", "draw", "unconstrained_parameter")
        assert upars.sizes["unconstrained_parameter"] == 2
        assert list(upars.coords["unconstrained_parameter"].values) == [
            "theta_simplex__[0]",
            "theta_simplex__[1]",
        ]
        rv = model.free_RVs[0]
        transform = model.rvs_to_transforms[rv]
        theta_back = transform.backward(upars.values.reshape(-1, 2), *rv.owner.inputs).eval()
        np.testing.assert_allclose(theta_back.reshape(*theta_draws.shape), theta_draws, rtol=1e-6)
        out = log_prob_fn(upars)
        assert out.dims == ("chain", "draw")
        reference = _compiled_logp(model)
        for ch, dr in [(0, 0), (0, 17), (1, 39)]:
            actual = reference(theta_simplex__=upars.values[ch, dr])[0]
            np.testing.assert_allclose(out.values[ch, dr], actual, rtol=1e-6)
        np.testing.assert_allclose(
            log_lik_i_fn(upars, 0).values, idata.log_likelihood["y"].values, rtol=1e-6
        )

    def test_dim_order_invariance(self, dirichlet_setup):
        idata, model = dirichlet_setup
        log_prob_fn, log_lik_i_fn, upars_ref = mm_from_pymc(idata, model=model, var_name="y")
        upars_t = upars_ref.transpose("unconstrained_parameter", "draw", "chain")
        np.testing.assert_allclose(log_prob_fn(upars_t).values, log_prob_fn(upars_ref).values)
        np.testing.assert_allclose(
            log_lik_i_fn(upars_t, 0).values, log_lik_i_fn(upars_ref, 0).values
        )
        param_dim = [d for d in idata.posterior["theta"].dims if d not in ("chain", "draw")][0]
        idata["posterior"] = idata.posterior.dataset.transpose(param_dim, "draw", "chain")
        _, _, upars_t2 = mm_from_pymc(idata, model=model, var_name="y")
        np.testing.assert_allclose(upars_t2.values, upars_ref.values)

    @pytest.mark.parametrize(
        "rv_name, var_name, match",
        [
            ("y", "not_a_var", "is not present in `idata.observed_data`"),
            ("y_renamed", "y", "does not match any observed random variable"),
        ],
    )
    def test_invalid_var_name_raises(self, poisson_setup, rv_name, var_name, match):
        idata, _, y_obs = poisson_setup
        with pm.Model() as model:
            beta = pm.HalfNormal("beta", 1.0)
            pm.Poisson(rv_name, beta, observed=y_obs)
        with pytest.raises(ValueError, match=match):
            mm_from_pymc(idata, model=model, var_name=var_name)

    def test_missing_observed_data_group(self, poisson_setup):
        idata, model, _ = poisson_setup
        del idata["observed_data"]
        _, _, upars = mm_from_pymc(idata, model=model, var_name="y")
        assert upars.sizes["unconstrained_parameter"] == 1

    def test_discrete_free_rv_raises(self, poisson_setup):
        idata, _, _ = poisson_setup
        with pm.Model() as model:
            p = pm.Poisson("p", mu=2.0)
            pm.Normal("y", p, 1.0, observed=[1.0, 2.0, 3.0])
        with pytest.raises(NotImplementedError, match="is not supported for models with discrete"):
            mm_from_pymc(idata, model=model, var_name=None)

    def test_potential_raises(self, poisson_setup):
        idata, _, _ = poisson_setup
        with pm.Model() as model:
            beta = pm.HalfNormal("beta", 1.0)
            pm.Potential("constr", pm.math.constant(0.0))
            pm.Poisson("y", beta, observed=[1.0, 2.0, 3.0])
        with pytest.raises(NotImplementedError, match="potentials"):
            mm_from_pymc(idata, model=model, var_name="y")

    def test_multi_observed_rv_requires_var_name(self):
        with pm.Model() as model:
            beta = pm.HalfNormal("beta", 1.0)
            pm.Poisson("y1", beta, observed=[1.0, 2.0, 3.0])
            pm.Poisson("y2", beta, observed=[1.0, 2.0, 3.0])
        rng = np.random.default_rng(0)
        n_chains, n_draws = 2, 30
        beta_draws = rng.lognormal(0, 1, size=(n_chains, n_draws))
        ll = poisson.logpmf(np.array([1.0, 2.0, 3.0])[None, None, :], beta_draws[:, :, None])
        idata = azb.from_dict(
            {
                "posterior": {"beta": beta_draws},
                "log_likelihood": {"y1": ll, "y2": ll},
                "observed_data": {
                    "y1": np.array([1.0, 2.0, 3.0]),
                    "y2": np.array([1.0, 2.0, 3.0]),
                },
            }
        )
        with pytest.raises(ValueError, match="multiple observed random variables"):
            mm_from_pymc(idata, model=model, var_name=None)
        _, _, up = mm_from_pymc(idata, model=model, var_name="y1")
        assert up.sizes["unconstrained_parameter"] == 1
