"""Test pointwise conditional log-likelihoods for non-factorized models."""

import pytest
from numpy.testing import assert_allclose

from ..helpers import importorskip

azb = importorskip("arviz_base")
xr = importorskip("xarray")
np = importorskip("numpy")

from arviz_stats import log_lik_nonfactorized, loo
from arviz_stats.base import array_stats


def test_log_lik_nonfactorized():
    rng = np.random.default_rng(0)
    y = rng.normal(size=5)
    mean = rng.normal(size=(2, 10, 5))
    factor = rng.normal(size=(2, 10, 5, 5))
    cov = factor @ factor.swapaxes(-1, -2) + np.eye(5)
    df = rng.uniform(3, 10, size=(2, 10))
    chol = np.linalg.cholesky(cov)
    data = azb.from_dict(
        {
            "posterior": {
                "mu": mean,
                "cov": cov,
                "chol": chol,
                "chol_upper": chol.swapaxes(-1, -2),
                "nu": df,
            },
            "observed_data": {"y": y},
        },
        dims={
            "mu": ["obs_id"],
            "cov": ["obs_id", "obs_id_bis"],
            "chol": ["obs_id", "obs_id_bis"],
            "chol_upper": ["obs_id", "obs_id_bis"],
            "y": ["obs_id"],
        },
    )

    log_lik = log_lik_nonfactorized(data, mean="mu", cov="cov", df="nu")

    assert log_lik.name == "y"
    assert log_lik.dims == ("chain", "draw", "obs_id")
    expected = array_stats.log_lik_nonfactorized(y, mean, cov=cov, df=df)
    assert_allclose(log_lik.values, expected)
    log_lik_chol = log_lik_nonfactorized(data, mean="mu", chol="chol", df="nu")
    assert_allclose(log_lik_chol.values, expected)
    log_lik_upper = log_lik_nonfactorized(data, mean="mu", chol="chol_upper", lower=False, df="nu")
    assert_allclose(log_lik_upper.values, expected)


def test_log_lik_nonfactorized_dims():
    rng = np.random.default_rng(0)
    y = xr.DataArray(rng.normal(size=(3, 4)), dims=("group", "time"))
    factor = rng.normal(size=(3, 4, 4))
    prec = xr.DataArray(
        factor @ factor.swapaxes(-1, -2) + np.eye(4), dims=("group", "time", "time_bis")
    )
    data = xr.DataTree.from_dict(
        {"constant_data": xr.Dataset({"prec": prec}), "observed_data": xr.Dataset({"y": y})}
    )

    log_lik = log_lik_nonfactorized(data, mean=0, prec="prec", dims=("time", "time_bis"))

    assert log_lik.dims == ("group", "time")
    expected = array_stats.log_lik_nonfactorized(y.values, 0, prec=prec.values)
    assert_allclose(log_lik.values, expected)
    with pytest.raises(ValueError, match="two matrix dimensions"):
        log_lik_nonfactorized(data, mean=0, prec="prec")


def test_log_lik_nonfactorized_mean_dims():
    data = azb.from_dict(
        {
            "posterior": {"mu": np.zeros((2, 10, 5)), "cov": np.ones((2, 10, 5, 5))},
            "observed_data": {"y": np.zeros(5)},
        }
    )
    with pytest.raises(ValueError, match="neither sample dimensions"):
        log_lik_nonfactorized(data, mean="mu", cov="cov")


def test_log_lik_nonfactorized_one_matrix():
    data = azb.from_dict(
        {"posterior": {"mu": np.zeros((2, 10))}, "observed_data": {"y": np.zeros(2)}}
    )
    with pytest.raises(ValueError, match="Exactly one of `cov`, `prec` and `chol`"):
        log_lik_nonfactorized(data, mean="mu")


def test_log_lik_nonfactorized_exact():
    rng = np.random.default_rng(1)
    n_obs = 8
    lag = np.abs(np.subtract.outer(np.arange(n_obs), np.arange(n_obs)))
    cov = 0.6**lag
    y = rng.multivariate_normal(np.ones(n_obs), cov)
    prec = np.linalg.inv(cov)
    post_var = 1 / (1 + prec.sum())
    post_mean = post_var * (prec @ y).sum()
    theta = rng.normal(post_mean, np.sqrt(post_var), size=(4, 500))
    data = azb.from_dict(
        {
            "posterior": {"theta": theta, "cov": np.broadcast_to(cov, (4, 500, n_obs, n_obs))},
            "observed_data": {"y": y},
        }
    )

    data["log_likelihood"] = log_lik_nonfactorized(data, mean="theta", cov="cov").to_dataset()
    loo_data = loo(data, pointwise=True)

    exact = array_stats.log_lik_nonfactorized(y, 0, cov=cov + 1)
    assert_allclose(loo_data.elpd_i, exact, atol=0.02)
