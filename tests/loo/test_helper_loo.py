"""Test helper functions for PSIS-LOO-CV."""

# pylint: disable=redefined-outer-name, unused-argument
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal

from ..helpers import importorskip

azb = importorskip("arviz_base")
xr = importorskip("xarray")
sp = importorskip("scipy")

from arviz_stats.loo.helper_loo import (
    _align_data_to_obs,
    _check_log_density,
    _check_log_jacobian,
    _compute_loo_approximation,
    _diff_srs_estimator,
    _extract_loo_data,
    _generate_subsample_indices,
    _get_log_likelihood_i,
    _get_r_eff,
    _get_r_eff_i,
    _get_weights_and_k_i,
    _prepare_full_arrays,
    _prepare_loo_inputs,
    _prepare_subsample,
    _prepare_update_subsample,
    _select_obs_by_coords,
    _select_obs_by_indices,
    _shift,
    _shift_and_cov,
    _shift_and_scale,
    _srs_estimator,
    _warn_pareto_k,
    _warn_pointwise_loo,
)
from arviz_stats.loo.loo_moment_match import _split_moment_match
from arviz_stats.loo.loo_subsample import loo_subsample
from arviz_stats.manipulation import thin
from arviz_stats.utils import get_log_likelihood_dataset


def log_lik_fn(obs_da, datatree):
    theta = datatree.posterior["theta"]
    sigma = 12.5
    log_lik = -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * ((obs_da - theta) / sigma) ** 2
    return log_lik


def log_lik_fn_subsample(obs_da, datatree):
    theta = datatree.posterior["theta"]
    sigma = datatree.constant_data["sigma"]
    return sp.stats.norm.logpdf(obs_da, loc=theta, scale=sigma)


def test_get_r_eff(centered_eight):
    n_samples = centered_eight.posterior.chain.size * centered_eight.posterior.draw.size
    r_eff = _get_r_eff(centered_eight, n_samples)

    assert isinstance(r_eff, float)
    assert r_eff > 0
    assert r_eff <= 1.0


def test_get_r_eff_error():
    data = xr.Dataset()
    with pytest.raises(TypeError, match="Must be able to extract a posterior group"):
        _get_r_eff(data, 1000)


def test_diff_srs_estimator():
    elpd_loo_i_sample = xr.DataArray(np.array([-5.0, -6.0, -7.0]))
    lpd_approx_sample = xr.DataArray(np.array([-4.5, -5.5, -6.5]))
    lpd_approx_all = xr.DataArray(np.array([-4.5, -5.5, -6.5, -7.5, -8.5]))
    n_data_points = 5

    elpd_est, subsampling_se, total_se = _diff_srs_estimator(
        elpd_loo_i_sample, lpd_approx_sample, lpd_approx_all, n_data_points
    )

    assert isinstance(elpd_est, float)
    assert isinstance(subsampling_se, float)
    assert isinstance(total_se, float)
    assert subsampling_se >= 0
    assert total_se >= 0

    assert_almost_equal(elpd_est, -35.0, decimal=1)


def test_srs_estimator():
    y_sample = xr.DataArray(np.array([1.0, 2.0, 3.0, 4.0]))
    n_data_points = 10

    y_hat, var_y_hat, hat_var_y = _srs_estimator(y_sample, n_data_points)

    expected_y_hat = n_data_points * y_sample.mean().values

    assert isinstance(var_y_hat, float)
    assert isinstance(hat_var_y, float)
    assert var_y_hat >= 0
    assert hat_var_y >= 0

    assert_almost_equal(y_hat, expected_y_hat, decimal=6)


def test_generate_subsample_indices():
    n_data_points = 10
    observations = 5
    indices, subsample_size = _generate_subsample_indices(n_data_points, observations, seed=42)

    assert subsample_size == observations
    assert indices.shape == (observations,)
    assert indices.min() >= 0
    assert indices.max() < n_data_points

    obs_array = np.array([1, 3, 5, 7])
    indices, subsample_size = _generate_subsample_indices(n_data_points, obs_array, seed=42)

    assert subsample_size == len(obs_array)
    assert_allclose(indices, obs_array)


def test_generate_subsample_indices_errors():
    n_data_points = 10

    with pytest.raises(ValueError, match="Number of observations must be between 1 and"):
        _generate_subsample_indices(n_data_points, 0, seed=42)

    with pytest.raises(ValueError, match="Number of observations must be between 1 and"):
        _generate_subsample_indices(n_data_points, n_data_points + 1, seed=42)

    with pytest.raises(TypeError, match="observations must be an integer"):
        _generate_subsample_indices(n_data_points, 4.5, seed=42)

    with pytest.raises(ValueError, match="Observation indices must be between"):
        _generate_subsample_indices(n_data_points, np.array([-1, 2, 3]), seed=42)

    with pytest.raises(ValueError, match="Observation indices must be between"):
        _generate_subsample_indices(n_data_points, np.array([5, 8, 10]), seed=42)


def test_prepare_loo_inputs(centered_eight):
    result = _prepare_loo_inputs(centered_eight, var_name="obs")

    assert result.var_name == "obs"
    assert result.sample_dims == ["chain", "draw"]
    assert "school" in result.obs_dims
    assert result.n_samples > 0
    assert result.n_data_points > 0


def test_get_weights_and_k_i(centered_eight):
    loo_inputs = _prepare_loo_inputs(centered_eight, var_name="obs")
    log_lik_i = _get_log_likelihood_i(loo_inputs.log_likelihood, 0, loo_inputs.obs_dims)

    log_weights_i, pareto_k_i = _get_weights_and_k_i(
        None,
        None,
        0,
        loo_inputs.obs_dims,
        loo_inputs.sample_dims,
        centered_eight,
        loo_inputs.n_samples,
        None,
        log_lik_i,
        loo_inputs.var_name,
    )

    assert isinstance(log_weights_i, xr.DataArray)
    assert all(dim in log_weights_i.dims for dim in loo_inputs.sample_dims)

    for dim in loo_inputs.sample_dims:
        assert log_weights_i.sizes[dim] == loo_inputs.log_likelihood.sizes[dim]
    for obs_dim in loo_inputs.obs_dims:
        if obs_dim in log_weights_i.dims:
            assert log_weights_i.sizes[obs_dim] == 1

    assert isinstance(pareto_k_i, xr.DataArray)
    assert all(dim in loo_inputs.obs_dims for dim in pareto_k_i.dims)

    for dim in pareto_k_i.dims:
        assert pareto_k_i.sizes[dim] == 1


def test_extract_loo_data(elpd_data):
    elpd_i_extracted, pareto_k_extracted = _extract_loo_data(elpd_data)

    assert isinstance(elpd_i_extracted, xr.DataArray)
    assert isinstance(pareto_k_extracted, xr.DataArray)
    assert elpd_i_extracted.dims == pareto_k_extracted.dims
    assert elpd_i_extracted.shape == pareto_k_extracted.shape

    xr.testing.assert_allclose(elpd_i_extracted, elpd_data.elpd_i)
    xr.testing.assert_allclose(pareto_k_extracted, elpd_data.pareto_k)


def test_check_log_density(log_likelihood_dataset):
    log_likelihood = log_likelihood_dataset["obs"]
    n_samples = log_likelihood.chain.size * log_likelihood.draw.size
    sample_dims = ["chain", "draw"]

    rng = np.random.default_rng(42)
    log_dens_np = rng.normal(size=n_samples)
    result_np = _check_log_density(log_dens_np, "log_p", log_likelihood, n_samples, sample_dims)

    assert isinstance(result_np, xr.DataArray)
    assert all(dim in result_np.dims for dim in sample_dims)

    log_dens_da = xr.DataArray(
        rng.normal(size=(log_likelihood.chain.size, log_likelihood.draw.size)),
        dims=sample_dims,
        coords={dim: log_likelihood[dim] for dim in sample_dims},
    )
    result_da = _check_log_density(log_dens_da, "log_p", log_likelihood, n_samples, sample_dims)

    assert isinstance(result_da, xr.DataArray)
    assert all(dim in result_da.dims for dim in sample_dims)


def test_check_log_density_errors(log_likelihood_dataset):
    log_likelihood = log_likelihood_dataset["obs"]
    n_samples = log_likelihood.chain.size * log_likelihood.draw.size
    sample_dims = ["chain", "draw"]

    with pytest.raises(TypeError, match="log_p must be a numpy ndarray or xarray DataArray"):
        _check_log_density([1, 2, 3], "log_p", log_likelihood, n_samples, sample_dims)

    rng = np.random.default_rng(42)
    with pytest.raises(ValueError, match="Size of log_p .* must match"):
        _check_log_density(
            rng.normal(size=n_samples - 1), "log_p", log_likelihood, n_samples, sample_dims
        )

    bad_dims_da = xr.DataArray(rng.normal(size=n_samples), dims=["sample"])
    with pytest.raises(ValueError, match="log_p must have dimension 'chain'"):
        _check_log_density(bad_dims_da, "log_p", log_likelihood, n_samples, sample_dims)


def test_warn_pareto_k():
    pareto_k_good = np.array([0.1, 0.2, 0.3, 0.4])
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        warn_flag, _ = _warn_pareto_k(pareto_k_good, n_samples=1000)
        assert not warn_flag
        assert len(record) == 0

    pareto_k_bad = np.array([0.1, 0.2, 0.8, 0.9])
    with pytest.warns(
        UserWarning, match="Estimated shape parameter of Pareto distribution is greater than"
    ):
        warn_flag, _ = _warn_pareto_k(pareto_k_bad, n_samples=1000)
        assert warn_flag


def test_warn_pointwise_loo():
    elpd = np.array([1.0, 2.0, 3.0])
    elpd_i_same = np.array([1.0, 2.0, 3.0])

    with pytest.warns(UserWarning, match="The point-wise LOO is the same with the sum LOO"):
        _warn_pointwise_loo(elpd, elpd_i_same)

    elpd = np.array([1.0, 2.0, 3.0])
    elpd_i_diff = np.array([1.1, 2.2, 3.3])

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _warn_pointwise_loo(elpd, elpd_i_diff)
        assert len(record) == 0


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_thin_draws(log_likelihood_dataset):
    log_likelihood = log_likelihood_dataset["obs"]
    original_draws = log_likelihood.draw.size
    original_chains = log_likelihood.chain.size
    original_samples = original_draws * original_chains

    thinned = thin(log_likelihood, factor=original_samples + 100)
    assert thinned.draw.size == 1
    assert thinned.chain.size == original_chains

    thinned = thin(log_likelihood)
    assert thinned.chain.size == original_chains
    assert thinned.draw.size < original_draws
    assert thinned.draw.size > 0

    if len(thinned.draw) > 1:
        observed_factor = int(thinned.draw.values[1] - thinned.draw.values[0])
        for i in range(1, len(thinned.draw) - 1):
            assert thinned.draw.values[i + 1] - thinned.draw.values[i] == observed_factor

    thinned = thin(log_likelihood, factor=5)
    assert thinned.chain.size == original_chains
    assert thinned.draw.size == original_draws // 5

    with pytest.raises(ValueError, match="factor must be greater than 1"):
        thin(log_likelihood, factor=0)


def test_select_obs_by_indices():
    data = xr.DataArray(np.arange(10), dims=["x"], coords={"x": np.arange(10)})

    indices = np.array([1, 3, 5])
    result = _select_obs_by_indices(data, indices, dims=["x"], dim_name="x")

    assert result.shape == (3,)
    assert_allclose(result.values, np.array([1, 3, 5]))


def test_select_obs_by_coords():
    data = xr.DataArray(np.arange(10), dims=["x"], coords={"x": np.arange(10)})

    coord_array = xr.DataArray([5, 7, 9], dims=["x"], coords={"x": [5, 7, 9]})
    result = _select_obs_by_coords(data, coord_array, dims=["x"], dim_name="x")

    assert result.shape == (3,)
    assert_allclose(result.values, np.array([5, 7, 9]))


@pytest.mark.parametrize("method", ["lpd", "plpd"])
def test_prepare_subsample(centered_eight, method):
    log_likelihood_ds = get_log_likelihood_dataset(centered_eight, var_names="obs")
    log_likelihood_da = log_likelihood_ds["obs"]

    var_name = "obs"
    observations = 3
    seed = 42
    param_names = ["theta"]
    log = True
    obs_dims = ["school"]
    sample_dims = ["chain", "draw"]
    n_data_points = log_likelihood_da.school.size
    n_samples = log_likelihood_da.chain.size * log_likelihood_da.draw.size

    result = _prepare_subsample(
        centered_eight,
        log_likelihood_da,
        var_name,
        observations,
        seed,
        method,
        log_lik_fn,
        param_names,
        log,
        obs_dims,
        sample_dims,
        n_data_points,
        n_samples,
    )

    assert result.subsample_size == observations
    assert result.indices.shape == (observations,)
    assert isinstance(result.log_likelihood_sample, xr.DataArray)
    assert isinstance(result.lpd_approx_sample, xr.DataArray)
    assert isinstance(result.lpd_approx_all, xr.DataArray)
    assert "school" in result.log_likelihood_sample.dims
    assert result.log_likelihood_sample.sizes["school"] == result.subsample_size
    assert result.lpd_approx_sample.sizes["school"] == result.subsample_size
    assert result.lpd_approx_all.shape[0] == n_data_points


@pytest.mark.parametrize("method", ["lpd", "plpd"])
def test_prepare_update_subsample(centered_eight, method):
    var_name = "obs"
    seed = 42
    observations = 2
    param_names = ["theta"]
    log = True
    n_data_points = centered_eight.observed_data[var_name].size
    initial_observations = n_data_points - observations

    loo_orig_subsample = loo_subsample(
        data=centered_eight,
        observations=initial_observations,
        var_name=var_name,
        seed=seed - 1,
        method=method,
        log_lik_fn=log_lik_fn,
        param_names=param_names,
        log=log,
    )

    result = _prepare_update_subsample(
        loo_orig=loo_orig_subsample,
        data=centered_eight,
        observations=observations,
        var_name=var_name,
        seed=seed,
        method=method,
        log_lik_fn=log_lik_fn,
        param_names=param_names,
        log=log,
    )

    old_indices = np.where(~np.isnan(loo_orig_subsample.elpd_i.values.flatten()))[0]
    expected_combined_size = len(old_indices) + observations

    assert result.combined_size == expected_combined_size
    assert len(result.new_indices) == observations
    assert isinstance(result.log_likelihood_new, xr.Dataset)
    assert isinstance(result.lpd_approx_all, xr.DataArray)
    assert isinstance(result.old_elpd_i, xr.DataArray)
    assert isinstance(result.old_pareto_k, xr.DataArray)
    assert result.concat_dim == "school"
    assert var_name in result.log_likelihood_new
    assert result.lpd_approx_all.dims == ("school",)
    assert result.lpd_approx_all.shape == (n_data_points,)
    assert not np.any(np.isin(result.new_indices, old_indices))
    assert len(np.intersect1d(result.new_indices, old_indices)) == 0


def test_shift():
    rng = np.random.default_rng(42)
    chain_size, draw_size, param_size = 2, 100, 3
    upars = xr.DataArray(
        rng.normal(size=(chain_size, draw_size, param_size)),
        dims=["chain", "draw", "param"],
        coords={
            "chain": np.arange(chain_size),
            "draw": np.arange(draw_size),
            "param": np.arange(param_size),
        },
    )

    lwi = xr.DataArray(
        rng.normal(size=(chain_size, draw_size)),
        dims=["chain", "draw"],
        coords={"chain": upars.chain, "draw": upars.draw},
    )

    result = _shift(upars, lwi)

    assert hasattr(result, "upars")
    assert hasattr(result, "shift")
    assert isinstance(result.upars, xr.DataArray)
    assert isinstance(result.shift, np.ndarray)

    assert set(result.upars.dims) == set(upars.dims)
    assert result.upars.sizes == upars.sizes
    assert result.shift.shape == (param_size,)

    upars_stacked = upars.stack(__sample__=["chain", "draw"])
    lwi_stacked = lwi.stack(__sample__=["chain", "draw"])
    weights = np.exp(lwi_stacked.values)

    expected_mean = np.sum(
        weights[:, None] * upars_stacked.transpose("__sample__", "param").values, axis=0
    )
    actual_mean = result.upars.stack(__sample__=["chain", "draw"]).mean(dim="__sample__").values

    assert_allclose(actual_mean, expected_mean, rtol=1e-5)


def test_shift_and_scale():
    rng = np.random.default_rng(42)
    chain_size, draw_size, param_size = 2, 100, 3
    upars = xr.DataArray(
        rng.normal(size=(chain_size, draw_size, param_size)),
        dims=["chain", "draw", "param"],
        coords={
            "chain": np.arange(chain_size),
            "draw": np.arange(draw_size),
            "param": np.arange(param_size),
        },
    )

    lwi = xr.DataArray(
        rng.normal(size=(chain_size, draw_size)),
        dims=["chain", "draw"],
        coords={"chain": upars.chain, "draw": upars.draw},
    )

    result = _shift_and_scale(upars, lwi)

    assert hasattr(result, "upars")
    assert hasattr(result, "shift")
    assert hasattr(result, "scaling")
    assert isinstance(result.upars, xr.DataArray)
    assert isinstance(result.shift, np.ndarray)
    assert isinstance(result.scaling, np.ndarray)

    assert set(result.upars.dims) == set(upars.dims)
    assert result.upars.sizes == upars.sizes
    assert result.shift.shape == (param_size,)
    assert result.scaling.shape == (param_size,)

    assert np.all(result.scaling > 0)


def test_shift_and_cov():
    rng = np.random.default_rng(42)
    chain_size, draw_size, param_size = 2, 100, 3
    upars = xr.DataArray(
        rng.normal(size=(chain_size, draw_size, param_size)),
        dims=["chain", "draw", "param"],
        coords={
            "chain": np.arange(chain_size),
            "draw": np.arange(draw_size),
            "param": np.arange(param_size),
        },
    )

    lwi = xr.DataArray(
        rng.normal(size=(chain_size, draw_size)),
        dims=["chain", "draw"],
        coords={"chain": upars.chain, "draw": upars.draw},
    )

    result = _shift_and_cov(upars, lwi)

    assert hasattr(result, "upars")
    assert hasattr(result, "shift")
    assert hasattr(result, "mapping")
    assert isinstance(result.upars, xr.DataArray)
    assert isinstance(result.shift, np.ndarray)
    assert isinstance(result.mapping, np.ndarray)

    assert set(result.upars.dims) == set(upars.dims)
    assert result.upars.sizes == upars.sizes
    assert result.shift.shape == (param_size,)
    assert result.mapping.shape == (param_size, param_size)


@pytest.mark.parametrize("cov", [True, False])
def test_split_moment_match(cov, centered_eight):
    rng = np.random.default_rng(42)
    posterior = centered_eight.posterior
    chain_size = posterior.chain.size
    draw_size = posterior.draw.size
    param_size = 3

    upars = xr.DataArray(
        rng.normal(size=(chain_size, draw_size, param_size)),
        dims=["chain", "draw", "param"],
        coords={
            "chain": posterior.chain,
            "draw": posterior.draw,
            "param": np.arange(param_size),
        },
    )

    total_shift = rng.normal(size=param_size)
    total_scaling = rng.uniform(0.5, 2.0, param_size)
    total_mapping = np.eye(param_size) + 0.1 * rng.normal(size=(param_size, param_size))

    total_mapping = total_mapping @ total_mapping.T

    i = 0
    reff = 0.8

    def log_prob_upars_fn(upars_in):
        return xr.DataArray(
            -0.5 * (upars_in**2).sum(dim="param"),
            dims=["chain", "draw"],
            coords={"chain": upars_in.chain, "draw": upars_in.draw},
        )

    def log_lik_i_upars_fn(upars_in, _idx):
        return xr.DataArray(
            -0.5 * upars_in.isel(param=0) ** 2,
            dims=["chain", "draw"],
            coords={"chain": upars_in.chain, "draw": upars_in.draw},
        )

    result = _split_moment_match(
        upars=upars,
        cov=cov,
        total_shift=total_shift,
        total_scaling=total_scaling,
        total_mapping=total_mapping,
        i=i,
        reff=reff,
        log_prob_upars_fn=log_prob_upars_fn,
        log_lik_i_upars_fn=log_lik_i_upars_fn,
    )

    assert hasattr(result, "lwi")
    assert hasattr(result, "lwfi")
    assert hasattr(result, "log_liki")
    assert hasattr(result, "reff")

    assert result.lwi.dims == ("chain", "draw")
    assert result.lwfi.dims == ("chain", "draw")
    assert result.log_liki.dims == ("chain", "draw")

    assert result.lwi.shape == (chain_size, draw_size)
    assert result.lwfi.shape == (chain_size, draw_size)
    assert result.log_liki.shape == (chain_size, draw_size)


@pytest.mark.parametrize("method", ["lpd", "plpd"])
def test_compute_loo_approximation(centered_eight_with_sigma, method):
    result = _compute_loo_approximation(
        data=centered_eight_with_sigma,
        var_name="obs",
        log_lik_fn=log_lik_fn_subsample,
        param_names=["theta"],
        method=method,
        log=True,
    )

    assert isinstance(result, xr.DataArray)
    assert result.name in ["lpd", "plpd"]
    assert result.dims == ("school",)
    assert result.shape == (8,)
    assert np.all(np.isfinite(result.values))


def test_compute_loo_approximation_auxiliary_data(centered_eight):
    sigma_values = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])
    sigma_da = xr.DataArray(
        sigma_values,
        dims=["school"],
        coords={"school": centered_eight.observed_data.school.values},
    )

    centered_eight_aux = centered_eight.copy()
    centered_eight_aux["constant_data"] = (
        centered_eight_aux["constant_data"].to_dataset().assign(sigma=sigma_da)
    )

    def log_lik_fn_aux(obs_da, datatree):
        theta = datatree.posterior["theta"]
        sigma = datatree.constant_data["sigma"]
        return -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * ((obs_da - theta) / sigma) ** 2

    subset_schools = centered_eight_aux.observed_data["obs"].isel(school=[1, 3, 5])
    aligned_data = _align_data_to_obs(centered_eight_aux, subset_schools)

    assert "constant_data" in aligned_data
    assert "sigma" in aligned_data.constant_data
    assert aligned_data.constant_data["sigma"].shape == (3,)
    assert_allclose(aligned_data.constant_data["sigma"].values, sigma_values[[1, 3, 5]])

    result = _compute_loo_approximation(
        data=centered_eight_aux,
        var_name="obs",
        log_lik_fn=log_lik_fn_aux,
        param_names=["theta"],
        method="lpd",
        log=True,
    )

    assert np.all(np.isfinite(result.values))


def test_compute_loo_approximation_errors(centered_eight_with_sigma):
    def log_lik_fn_bad(obs_da, datatree):
        theta = datatree.posterior["theta"]
        sigma = datatree.constant_data["sigma"]
        result = sp.stats.norm.logpdf(obs_da, loc=theta, scale=sigma)
        return result.expand_dims({"extra": [1, 2]})

    with pytest.raises(ValueError, match="dims|dimensions"):
        _compute_loo_approximation(
            data=centered_eight_with_sigma,
            var_name="obs",
            log_lik_fn=log_lik_fn_bad,
            param_names=["theta"],
            method="lpd",
            log=True,
        )

    def log_lik_fn_missing_param(obs_da, datatree):
        _ = datatree.posterior["missing_param"]
        return obs_da * 0

    with pytest.raises(KeyError, match="Variable not found in posterior"):
        _compute_loo_approximation(
            data=centered_eight_with_sigma,
            var_name="obs",
            log_lik_fn=log_lik_fn_missing_param,
            param_names=["theta"],
            method="lpd",
            log=True,
        )


def test_prepare_full_arrays_single_dim(log_likelihood_dataset):
    log_lik = log_likelihood_dataset["obs"]
    obs_dims = ["school"]
    indices = np.array([1, 3, 5])

    pointwise_values = xr.DataArray(
        [-2.5, -3.0, -2.8], dims=["school"], coords={"school": [1, 3, 5]}
    )
    pareto_k_values = xr.DataArray([0.3, 0.4, 0.2], dims=["school"], coords={"school": [1, 3, 5]})
    ref_array = log_lik.isel(chain=0, draw=0)

    elpd_full, pareto_full = _prepare_full_arrays(
        pointwise_values, pareto_k_values, ref_array, indices, obs_dims
    )

    assert elpd_full.shape == ref_array.shape
    assert pareto_full.shape == ref_array.shape
    assert_allclose(elpd_full.isel(school=indices).values, pointwise_values.values)
    assert np.sum(~np.isnan(elpd_full.values)) == len(indices)


@pytest.mark.parametrize("obs_dims", [[], None])
def test_prepare_full_arrays_no_obs_dims(obs_dims):
    pointwise_values = xr.DataArray([-2.5])
    pareto_k_values = xr.DataArray([0.3])
    ref_array = xr.DataArray([1.0])
    indices = np.array([0])

    elpd_full, pareto_full = _prepare_full_arrays(
        pointwise_values, pareto_k_values, ref_array, indices, obs_dims or []
    )

    assert elpd_full is pointwise_values
    assert pareto_full is pareto_k_values


def test_check_log_jacobian_valid(log_likelihood_dataset):
    log_lik = log_likelihood_dataset["obs"]
    obs_dims = ["school"]

    log_jacobian = xr.DataArray(np.zeros(8), dims=["school"], coords={"school": log_lik.school})

    result = _check_log_jacobian(log_jacobian, obs_dims)
    assert result is not None
    assert isinstance(result, xr.DataArray)
    assert result.dims == ("school",)


def test_check_log_jacobian_none():
    result = _check_log_jacobian(None, ["school"])
    assert result is None


@pytest.mark.parametrize(
    "error_case,error_type,error_match",
    [
        ("not_dataarray", TypeError, "log_jacobian must be an xarray.DataArray"),
        ("has_sample_dims", ValueError, "log_jacobian must not have sample dimensions"),
        ("wrong_dims", ValueError, "log_jacobian dimensions must exactly match"),
        ("has_nans", ValueError, "log_jacobian must contain only finite values"),
        ("has_infs", ValueError, "log_jacobian must contain only finite values"),
    ],
)
def test_check_log_jacobian_errors(log_likelihood_dataset, error_case, error_type, error_match):
    log_lik = log_likelihood_dataset["obs"]
    obs_dims = ["school"]

    if error_case == "not_dataarray":
        with pytest.raises(error_type, match=error_match):
            _check_log_jacobian([1, 2, 3], obs_dims)

    elif error_case == "has_sample_dims":
        bad_jac = xr.DataArray(
            np.zeros((log_lik.chain.size, log_lik.draw.size, log_lik.school.size)),
            dims=["chain", "draw", "school"],
            coords={"chain": log_lik.chain, "draw": log_lik.draw, "school": log_lik.school},
        )
        with pytest.raises(error_type, match=error_match):
            _check_log_jacobian(bad_jac, obs_dims)

    elif error_case == "wrong_dims":
        bad_jac = xr.DataArray(np.zeros(5), dims=["wrong_dim"])
        with pytest.raises(error_type, match=error_match):
            _check_log_jacobian(bad_jac, obs_dims)

    elif error_case == "has_nans":
        bad_jac = xr.DataArray(
            np.array([0.0, np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            dims=["school"],
            coords={"school": log_lik.school},
        )
        with pytest.raises(error_type, match=error_match):
            _check_log_jacobian(bad_jac, obs_dims)

    elif error_case == "has_infs":
        bad_jac = xr.DataArray(
            np.array([0.0, np.inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            dims=["school"],
            coords={"school": log_lik.school},
        )
        with pytest.raises(error_type, match=error_match):
            _check_log_jacobian(bad_jac, obs_dims)


def test_get_r_eff_i(log_likelihood_dataset):
    log_lik = log_likelihood_dataset["obs"]
    obs_dims = ["school"]

    r_eff_array = xr.DataArray(
        np.array([0.8, 0.85, 0.9, 0.75, 0.82, 0.88, 0.91, 0.79]),
        dims=["school"],
        coords={"school": log_lik.school},
    )

    r_eff_val = _get_r_eff_i(r_eff_array, 3, obs_dims)

    assert isinstance(r_eff_val, float)
    assert_allclose(r_eff_val, 0.75)


@pytest.mark.parametrize(
    "error_case,error_type,error_match",
    [
        ("not_dataarray", TypeError, "r_eff must be an xarray.DataArray"),
        ("no_obs_dims", ValueError, "r_eff must have observation dimensions"),
        ("missing_dims", ValueError, "r_eff must include observation dimensions"),
    ],
)
def test_get_r_eff_i_errors(error_case, error_type, error_match):
    obs_dims = ["school"]

    if error_case == "not_dataarray":
        with pytest.raises(error_type, match=error_match):
            _get_r_eff_i(0.8, 0, obs_dims)

    elif error_case == "no_obs_dims":
        r_eff_array = xr.DataArray([0.8], dims=["x"])
        with pytest.raises(error_type, match=error_match):
            _get_r_eff_i(r_eff_array, 0, [])

    elif error_case == "missing_dims":
        r_eff_array = xr.DataArray(np.ones(5), dims=["x"])
        with pytest.raises(error_type, match=error_match):
            _get_r_eff_i(r_eff_array, 0, obs_dims)


def test_select_obs_by_indices_multidim():
    data = xr.DataArray(
        np.arange(12).reshape(3, 4),
        dims=["x", "y"],
        coords={"x": [0, 1, 2], "y": [0, 1, 2, 3]},
    )
    indices = np.array([1, 5, 9])
    result = _select_obs_by_indices(data, indices, dims=["x", "y"], dim_name="__obs__")

    assert result.shape[0] == 3
    assert_allclose(result.values.ravel(), [1, 5, 9])


def test_diff_srs_estimator_edge_case_full_sample():
    elpd_sample = xr.DataArray([-5.0, -6.0])
    lpd_approx_sample = xr.DataArray([-5.0, -6.0])
    lpd_approx_all = xr.DataArray([-4.0, -5.0])
    n_data_points = 2

    elpd_est, subsampling_se, total_se = _diff_srs_estimator(
        elpd_sample, lpd_approx_sample, lpd_approx_all, n_data_points
    )

    assert isinstance(elpd_est, float)
    assert isinstance(subsampling_se, float)
    assert isinstance(total_se, float)
    assert subsampling_se == 0.0
