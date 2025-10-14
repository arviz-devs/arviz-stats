# pylint: disable=redefined-outer-name, unused-import, unused-argument
# ruff: noqa: F811, F401
import numpy as np
import pytest
from numpy.testing import assert_allclose

from ..helpers import importorskip

azb = importorskip("arviz_base")
xr = importorskip("xarray")
sp = importorskip("scipy")

from arviz_stats import loo_subsample, update_subsample
from arviz_stats.utils import ELPDData


def log_lik_fn(obs_da, datatree):
    theta = datatree.posterior["theta"]
    sigma = 12.5
    log_lik = -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * ((obs_da - theta) / sigma) ** 2
    return log_lik


def log_lik_fn_subsample(obs_da, datatree):
    theta = datatree.posterior["theta"]
    sigma = datatree.constant_data["sigma"]
    return sp.stats.norm.logpdf(obs_da, loc=theta, scale=sigma)


@pytest.mark.parametrize("pointwise", [True, False])
@pytest.mark.parametrize("method", ["lpd", "plpd"])
def test_loo_subsample(centered_eight_with_sigma, pointwise, method):
    observations = 4
    np.random.seed(42)
    loo_sub = loo_subsample(
        centered_eight_with_sigma,
        observations=observations,
        pointwise=pointwise,
        var_name="obs",
        method=method,
        log_lik_fn=log_lik_fn_subsample,
        param_names=["theta"],
        seed=42,
    )

    assert isinstance(loo_sub, ELPDData)
    assert loo_sub.kind == "loo"
    assert loo_sub.subsample_size == observations
    assert isinstance(loo_sub.elpd, float)
    assert isinstance(loo_sub.se, float) and loo_sub.se >= 0
    assert isinstance(loo_sub.p, float)
    assert isinstance(loo_sub.subsampling_se, float) and loo_sub.subsampling_se >= 0
    assert loo_sub.n_data_points == centered_eight_with_sigma.observed_data.obs.size

    assert -35 < loo_sub.elpd < -29
    assert 0 < loo_sub.p < 5

    if method == "plpd":
        assert loo_sub.elpd < -29

    if pointwise:
        assert hasattr(loo_sub, "elpd_i")
        assert hasattr(loo_sub, "pareto_k")
        assert loo_sub.elpd_i is not None
        assert loo_sub.pareto_k is not None
        assert loo_sub.elpd_i.dims == ("school",)
        assert loo_sub.elpd_i.shape == (loo_sub.n_data_points,)
        assert loo_sub.pareto_k.shape == (loo_sub.n_data_points,)
        assert np.sum(~np.isnan(loo_sub.pareto_k.values)) == observations
        assert np.isnan(loo_sub.elpd_i).sum() == loo_sub.n_data_points - observations
        assert not np.isnan(loo_sub.elpd_i).all()

        valid_k = loo_sub.pareto_k[~np.isnan(loo_sub.pareto_k)]
        assert np.all(valid_k < 1.0)
    else:
        assert not hasattr(loo_sub, "elpd_i") or loo_sub.elpd_i is None
        assert not hasattr(loo_sub, "pareto_k") or loo_sub.pareto_k is None


@pytest.mark.parametrize("method", ["lpd", "plpd"])
def test_loo_subsample_with_custom_loglik(centered_eight_with_sigma, method):
    observations = 4
    result = loo_subsample(
        centered_eight_with_sigma,
        observations=observations,
        var_name="obs",
        method=method,
        log_lik_fn=log_lik_fn_subsample,
        param_names=["theta"],
        seed=42,
        pointwise=True,
    )

    assert isinstance(result, ELPDData)
    assert result.subsample_size == observations
    assert -40 < result.elpd < -25
    assert np.sum(~np.isnan(result.pareto_k.values)) == observations

    if method == "lpd":
        loo_precomputed = loo_subsample(
            centered_eight_with_sigma,
            observations=observations,
            var_name="obs",
            method=method,
            seed=42,
            pointwise=True,
        )
        assert_allclose(result.elpd, loo_precomputed.elpd, rtol=5e-2)


@pytest.mark.parametrize("method", ["lpd", "plpd"])
def test_loo_subsample_loglik_errors(centered_eight_with_sigma, method):
    def log_lik_fn(obs_da, datatree):
        _ = datatree.posterior["missing_param"]
        return obs_da * 0

    with pytest.raises(KeyError, match="Variable not found in posterior"):
        loo_subsample(
            centered_eight_with_sigma,
            observations=4,
            var_name="obs",
            method=method,
            log_lik_fn=log_lik_fn,
            param_names=["theta"],
        )

    def log_lik_fn_scalar(obs_da, datatree):
        return xr.DataArray(0.0)

    with pytest.raises(ValueError, match="log_lik_fn must return an object with dims"):
        loo_subsample(
            centered_eight_with_sigma,
            observations=4,
            var_name="obs",
            method=method,
            log_lik_fn=log_lik_fn_scalar,
            param_names=["theta"],
        )


def test_update_loo_subsample(centered_eight_with_sigma):
    initial_loo = loo_subsample(
        centered_eight_with_sigma,
        observations=3,
        var_name="obs",
        method="lpd",
        log_lik_fn=log_lik_fn_subsample,
        param_names=["theta"],
        seed=42,
    )

    updated_loo = update_subsample(
        initial_loo,
        centered_eight_with_sigma,
        observations=3,
        var_name="obs",
        method="lpd",
        log_lik_fn=log_lik_fn_subsample,
        param_names=["theta"],
        seed=43,
    )

    assert updated_loo.subsample_size == 6
    assert updated_loo.subsampling_se <= initial_loo.subsampling_se
    assert np.sum(~np.isnan(updated_loo.elpd_i.values)) == 6
    assert np.sum(~np.isnan(updated_loo.pareto_k.values)) == 6


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("thin", [None, 2, "auto"])
def test_loo_subsample_thin_parameter(centered_eight_with_sigma, thin):
    result = loo_subsample(
        centered_eight_with_sigma,
        observations=4,
        var_name="obs",
        thin=thin,
        seed=42,
    )

    assert isinstance(result, ELPDData)
    assert result.kind == "loo"
    assert -40 < result.elpd < -25
    assert result.subsample_size == 4

    if thin is not None:
        assert hasattr(result, "thin_factor")
        assert result.thin_factor == thin


def test_loo_subsample_validation_errors(centered_eight_with_sigma):
    with pytest.raises(ValueError, match="Number of observations must be between 1 and"):
        loo_subsample(centered_eight_with_sigma, observations=0, var_name="obs")
    with pytest.raises(TypeError, match="observations must be an integer"):
        loo_subsample(centered_eight_with_sigma, observations=4.2, var_name="obs")


def test_log_weights_storage_subsample(centered_eight_with_sigma):
    loo_sub_pw_true = loo_subsample(
        centered_eight_with_sigma, observations=4, pointwise=True, seed=42
    )
    assert loo_sub_pw_true.log_weights is not None
    assert isinstance(loo_sub_pw_true.log_weights, xr.Dataset)
    assert "obs" in loo_sub_pw_true.log_weights
    assert loo_sub_pw_true.log_weights["obs"].shape == (4, 4, 500)

    loo_sub_pw_false = loo_subsample(
        centered_eight_with_sigma, observations=4, pointwise=False, seed=42
    )
    assert loo_sub_pw_false.log_weights is not None
    assert loo_sub_pw_false.log_weights.shape == (4, 4, 500)

    loo_updated = update_subsample(
        loo_sub_pw_true, centered_eight_with_sigma, observations=2, seed=43
    )
    assert loo_updated.log_weights is not None
    assert isinstance(loo_updated.log_weights, xr.Dataset)
    assert "obs" in loo_updated.log_weights
    assert loo_updated.log_weights["obs"].shape[0] >= 5


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_log_weights_input_formats_subsample(centered_eight_with_sigma):
    from arviz_stats import loo

    loo_result = loo(centered_eight_with_sigma, pointwise=True)
    loo_sub_elpddata = loo_subsample(
        centered_eight_with_sigma,
        observations=4,
        pointwise=True,
        var_name="obs",
        log_weights=loo_result,
    )
    assert loo_sub_elpddata is not None
    assert loo_sub_elpddata.log_weights is not None
    assert loo_sub_elpddata.log_weights["obs"].shape == (4, 4, 500)
