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


def test_loo_subsample_finite_values(centered_eight_with_sigma):
    result = loo_subsample(
        centered_eight_with_sigma, observations=4, pointwise=True, var_name="obs", seed=42
    )
    assert np.isfinite(result.elpd)
    assert np.isfinite(result.se)
    assert np.isfinite(result.p)
    assert np.isfinite(result.subsampling_se)


def test_loo_subsample_se_positive(centered_eight_with_sigma):
    result = loo_subsample(
        centered_eight_with_sigma, observations=4, pointwise=False, var_name="obs", seed=42
    )
    assert result.se > 0
    assert result.subsampling_se >= 0


def test_loo_subsample_observations_sampled(centered_eight_with_sigma):
    observations = 4
    result = loo_subsample(
        centered_eight_with_sigma,
        observations=observations,
        pointwise=True,
        var_name="obs",
        seed=42,
    )
    non_nan_count = np.sum(~np.isnan(result.elpd_i.values))
    assert non_nan_count == observations
    assert np.sum(~np.isnan(result.pareto_k.values)) == observations


def test_update_subsample_no_overlap(centered_eight_with_sigma):
    initial = loo_subsample(
        centered_eight_with_sigma,
        observations=3,
        var_name="obs",
        method="lpd",
        log_lik_fn=log_lik_fn_subsample,
        param_names=["theta"],
        seed=42,
    )
    initial_indices = np.where(~np.isnan(initial.elpd_i.values))[0]

    updated = update_subsample(
        initial,
        centered_eight_with_sigma,
        observations=2,
        var_name="obs",
        method="lpd",
        log_lik_fn=log_lik_fn_subsample,
        param_names=["theta"],
        seed=43,
    )
    updated_indices = np.where(~np.isnan(updated.elpd_i.values))[0]

    assert len(updated_indices) == len(initial_indices) + 2
    assert set(initial_indices).issubset(set(updated_indices))


def test_update_subsample_size_increases(centered_eight_with_sigma):
    initial = loo_subsample(
        centered_eight_with_sigma, observations=2, var_name="obs", seed=42, pointwise=True
    )
    updated = update_subsample(
        initial, centered_eight_with_sigma, observations=3, var_name="obs", seed=43
    )
    assert updated.subsample_size == 5
    assert initial.subsample_size == 2


def test_loo_subsample_methods_differ(centered_eight_with_sigma):
    result_lpd = loo_subsample(
        centered_eight_with_sigma,
        observations=4,
        method="lpd",
        var_name="obs",
        seed=42,
        pointwise=True,
    )
    result_plpd = loo_subsample(
        centered_eight_with_sigma,
        observations=4,
        method="plpd",
        var_name="obs",
        log_lik_fn=log_lik_fn_subsample,
        param_names=["theta"],
        seed=42,
        pointwise=True,
    )
    assert result_lpd.elpd != result_plpd.elpd


def test_loo_subsample_observations_as_array(centered_eight_with_sigma):
    indices = np.array([0, 2, 4, 6])
    result = loo_subsample(
        centered_eight_with_sigma,
        observations=indices,
        pointwise=True,
        var_name="obs",
        seed=42,
    )
    assert result.subsample_size == 4
    assert isinstance(result, ELPDData)
    non_nan_indices = np.where(~np.isnan(result.elpd_i.values))[0]
    assert np.array_equal(non_nan_indices, indices)
    assert np.sum(~np.isnan(result.pareto_k.values)) == 4


@pytest.mark.parametrize("method", ["lpd", "plpd"])
def test_loo_subsample_observations_as_array_methods(centered_eight_with_sigma, method):
    indices = np.array([1, 3, 5])
    result = loo_subsample(
        centered_eight_with_sigma,
        observations=indices,
        method=method,
        var_name="obs",
        log_lik_fn=log_lik_fn_subsample if method == "plpd" else None,
        param_names=["theta"] if method == "plpd" else None,
        pointwise=True,
        seed=42,
    )
    non_nan_indices = np.where(~np.isnan(result.elpd_i.values))[0]
    assert np.array_equal(non_nan_indices, indices)
    assert result.subsample_size == len(indices)


def test_loo_subsample_custom_reff(centered_eight_with_sigma):
    result_default = loo_subsample(
        centered_eight_with_sigma,
        observations=4,
        var_name="obs",
        seed=42,
        pointwise=True,
    )
    result_custom = loo_subsample(
        centered_eight_with_sigma,
        observations=4,
        var_name="obs",
        reff=0.5,
        seed=42,
        pointwise=True,
    )
    assert isinstance(result_custom, ELPDData)
    assert result_custom.elpd != result_default.elpd


def test_loo_subsample_lpd_custom_loglik(centered_eight_with_sigma):
    result_custom = loo_subsample(
        centered_eight_with_sigma,
        observations=4,
        method="lpd",
        var_name="obs",
        log_lik_fn=log_lik_fn_subsample,
        param_names=["theta"],
        seed=42,
        pointwise=True,
    )
    result_precomputed = loo_subsample(
        centered_eight_with_sigma,
        observations=4,
        method="lpd",
        var_name="obs",
        seed=42,
        pointwise=True,
    )
    assert isinstance(result_custom, ELPDData)
    assert_allclose(result_custom.elpd, result_precomputed.elpd, rtol=5e-2)
    assert_allclose(result_custom.p, result_precomputed.p, rtol=5e-2)


def test_loo_subsample_log_false(centered_eight_with_sigma):
    def lik_fn(obs_da, datatree):
        theta = datatree.posterior["theta"]
        sigma = datatree.constant_data["sigma"]
        return sp.stats.norm.pdf(obs_da, loc=theta, scale=sigma)

    result = loo_subsample(
        centered_eight_with_sigma,
        observations=4,
        method="plpd",
        var_name="obs",
        log_lik_fn=lik_fn,
        param_names=["theta"],
        log=False,
        seed=42,
        pointwise=True,
    )
    assert isinstance(result, ELPDData)
    assert np.isfinite(result.elpd)
    assert -40 < result.elpd < -25


def test_loo_subsample_single_observation(centered_eight_with_sigma):
    result = loo_subsample(
        centered_eight_with_sigma,
        observations=1,
        var_name="obs",
        seed=42,
        pointwise=True,
    )
    assert result.subsample_size == 1
    assert np.sum(~np.isnan(result.elpd_i.values)) == 1
    assert np.isfinite(result.elpd)


def test_loo_subsample_all_observations(centered_eight_with_sigma):
    n_obs = centered_eight_with_sigma.observed_data.obs.size
    result = loo_subsample(
        centered_eight_with_sigma,
        observations=n_obs,
        var_name="obs",
        seed=42,
        pointwise=True,
    )
    assert result.subsample_size == n_obs
    assert np.sum(~np.isnan(result.elpd_i.values)) == n_obs
    assert result.subsampling_se == 0.0


def test_loo_subsample_seed_reproducibility(centered_eight_with_sigma):
    result1 = loo_subsample(
        centered_eight_with_sigma,
        observations=4,
        var_name="obs",
        seed=123,
        pointwise=True,
    )
    result2 = loo_subsample(
        centered_eight_with_sigma,
        observations=4,
        var_name="obs",
        seed=123,
        pointwise=True,
    )
    assert_allclose(result1.elpd, result2.elpd)
    assert_allclose(result1.p, result2.p)
    assert np.array_equal(
        np.where(~np.isnan(result1.elpd_i.values))[0],
        np.where(~np.isnan(result2.elpd_i.values))[0],
    )


def test_loo_subsample_different_seeds(centered_eight_with_sigma):
    result1 = loo_subsample(
        centered_eight_with_sigma,
        observations=4,
        var_name="obs",
        seed=42,
        pointwise=True,
    )
    result2 = loo_subsample(
        centered_eight_with_sigma,
        observations=4,
        var_name="obs",
        seed=99,
        pointwise=True,
    )
    indices1 = np.where(~np.isnan(result1.elpd_i.values))[0]
    indices2 = np.where(~np.isnan(result2.elpd_i.values))[0]
    assert not np.array_equal(indices1, indices2)


def test_loo_subsample_invalid_method(centered_eight_with_sigma):
    with pytest.raises(ValueError, match="Method must be either 'lpd' or 'plpd'"):
        loo_subsample(
            centered_eight_with_sigma,
            observations=4,
            var_name="obs",
            method="invalid_method",
        )


def test_loo_subsample_observations_exceeds_total(centered_eight_with_sigma):
    n_obs = centered_eight_with_sigma.observed_data.obs.size
    with pytest.raises(ValueError, match="Number of observations must be between 1 and"):
        loo_subsample(
            centered_eight_with_sigma,
            observations=n_obs + 1,
            var_name="obs",
        )


def test_loo_subsample_plpd_without_loglik(centered_eight_with_sigma):
    with pytest.raises(ValueError, match="log_lik_fn must be provided when method='plpd'"):
        loo_subsample(
            centered_eight_with_sigma,
            observations=4,
            var_name="obs",
            method="plpd",
        )


def test_update_subsample_with_array_observations(centered_eight_with_sigma):
    initial = loo_subsample(
        centered_eight_with_sigma,
        observations=np.array([0, 2, 4]),
        var_name="obs",
        method="lpd",
        log_lik_fn=log_lik_fn_subsample,
        param_names=["theta"],
        seed=42,
    )
    updated = update_subsample(
        initial,
        centered_eight_with_sigma,
        observations=np.array([1, 3]),
        var_name="obs",
        method="lpd",
        log_lik_fn=log_lik_fn_subsample,
        param_names=["theta"],
        seed=43,
    )
    assert updated.subsample_size == 5
    assert np.sum(~np.isnan(updated.elpd_i.values)) == 5


def test_update_subsample_observations_none(centered_eight_with_sigma):
    initial = loo_subsample(
        centered_eight_with_sigma,
        observations=3,
        var_name="obs",
        seed=42,
        pointwise=True,
    )
    updated = update_subsample(
        initial,
        centered_eight_with_sigma,
        observations=None,
        var_name="obs",
    )
    assert updated.elpd == initial.elpd
    assert updated.subsample_size == initial.subsample_size


def test_update_subsample_observations_zero(centered_eight_with_sigma):
    initial = loo_subsample(
        centered_eight_with_sigma,
        observations=3,
        var_name="obs",
        seed=42,
        pointwise=True,
    )
    updated = update_subsample(
        initial,
        centered_eight_with_sigma,
        observations=0,
        var_name="obs",
    )
    assert updated.elpd == initial.elpd
    assert updated.subsample_size == initial.subsample_size


def test_update_subsample_plpd_method(centered_eight_with_sigma):
    initial = loo_subsample(
        centered_eight_with_sigma,
        observations=3,
        var_name="obs",
        method="plpd",
        log_lik_fn=log_lik_fn_subsample,
        param_names=["theta"],
        seed=42,
    )
    updated = update_subsample(
        initial,
        centered_eight_with_sigma,
        observations=2,
        var_name="obs",
        method="plpd",
        log_lik_fn=log_lik_fn_subsample,
        param_names=["theta"],
        seed=43,
    )
    assert updated.subsample_size == 5
    assert isinstance(updated, ELPDData)
    assert np.isfinite(updated.elpd)


def test_update_subsample_without_pointwise_raises(centered_eight_with_sigma):
    initial = loo_subsample(
        centered_eight_with_sigma,
        observations=3,
        var_name="obs",
        pointwise=False,
        seed=42,
    )
    with pytest.raises(ValueError, match="Original loo_subsample result must have pointwise=True"):
        update_subsample(
            initial,
            centered_eight_with_sigma,
            observations=2,
            var_name="obs",
        )


def test_loo_subsample_approximate_posterior(centered_eight, log_densities):
    log_p, log_q = log_densities["dataarray"]
    result = loo_subsample(
        centered_eight,
        observations=4,
        var_name="obs",
        log_p=log_p,
        log_q=log_q,
        seed=42,
        pointwise=True,
    )
    assert isinstance(result, ELPDData)
    assert result.approx_posterior is True
    assert np.isfinite(result.elpd)
    assert result.log_p is not None
    assert result.log_q is not None
    assert result.subsample_size == 4


@pytest.mark.parametrize("method", ["lpd", "plpd"])
def test_loo_subsample_approximate_posterior_methods(centered_eight_with_sigma, method):
    from arviz_stats.loo.loo_subsample import _prepare_loo_inputs

    loo_inputs = _prepare_loo_inputs(centered_eight_with_sigma, "obs", None)
    log_lik = loo_inputs.log_likelihood
    rng = np.random.default_rng(42)
    log_p_vals = rng.normal(size=(log_lik.chain.size, log_lik.draw.size))
    log_q_vals = rng.normal(loc=-1.0, size=(log_lik.chain.size, log_lik.draw.size))

    log_p = xr.DataArray(
        log_p_vals,
        dims=["chain", "draw"],
        coords={"chain": log_lik.chain, "draw": log_lik.draw},
    )
    log_q = xr.DataArray(
        log_q_vals,
        dims=["chain", "draw"],
        coords={"chain": log_lik.chain, "draw": log_lik.draw},
    )

    result = loo_subsample(
        centered_eight_with_sigma,
        observations=4,
        var_name="obs",
        method=method,
        log_p=log_p,
        log_q=log_q,
        log_lik_fn=log_lik_fn_subsample if method == "plpd" else None,
        param_names=["theta"] if method == "plpd" else None,
        seed=42,
        pointwise=True,
    )
    assert result.approx_posterior is True
    assert np.isfinite(result.elpd)
