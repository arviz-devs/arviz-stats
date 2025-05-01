# pylint: disable=redefined-outer-name

import warnings

import numpy as np
import pytest
import xarray as xr
from arviz_base import load_arviz_data
from numpy.testing import assert_allclose, assert_almost_equal

from arviz_stats.helper_loo import (
    _check_log_density,
    _diff_srs_estimator,
    _extract_loo_data,
    _generate_subsample_indices,
    _get_r_eff,
    _plpd_approx,
    _prepare_loo_inputs,
    _prepare_subsample,
    _select_obs_by_coords,
    _select_obs_by_indices,
    _srs_estimator,
    _warn_pareto_k,
    _warn_pointwise_loo,
)
from arviz_stats.utils import ELPDData, get_log_likelihood_dataset


@pytest.fixture(name="centered_eight", scope="session")
def fixture_centered_eight():
    return load_arviz_data("centered_eight")


@pytest.fixture(scope="module")
def log_likelihood_dataset(centered_eight):
    return get_log_likelihood_dataset(centered_eight, var_names="obs")


@pytest.fixture(scope="module")
def log_lik_fn():
    def _log_likelihood_eight_schools(data, theta):
        sigma = 12.5
        log_lik = -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * ((data - theta) / sigma) ** 2
        return log_lik

    return _log_likelihood_eight_schools


@pytest.fixture(scope="module")
def elpd_data(centered_eight):
    log_likelihood = get_log_likelihood_dataset(centered_eight, var_names="obs")["obs"]
    n_samples = log_likelihood.chain.size * log_likelihood.draw.size
    n_data_points = log_likelihood.school.size

    elpd_values = np.random.normal(size=n_data_points)
    pareto_k_values = np.random.uniform(0, 0.7, size=n_data_points)

    elpd_i = xr.DataArray(elpd_values, dims=["school"], coords={"school": log_likelihood.school})

    pareto_k = xr.DataArray(
        pareto_k_values, dims=["school"], coords={"school": log_likelihood.school}
    )

    mock_elpd = ELPDData(
        elpd=float(elpd_values.sum()),
        se=1.0,
        p=2.0,
        good_k=0.7,
        n_samples=n_samples,
        n_data_points=n_data_points,
        warning=False,
        elpd_i=elpd_i,
        pareto_k=pareto_k,
        scale="log",
        kind="loo",
    )
    return mock_elpd


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
    subsample_size = 3

    elpd_est, subsampling_se, total_se = _diff_srs_estimator(
        elpd_loo_i_sample, lpd_approx_sample, lpd_approx_all, n_data_points, subsample_size
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
    subsample_size = 4

    y_hat, var_y_hat, hat_var_y = _srs_estimator(y_sample, n_data_points, subsample_size)

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

    log_dens_np = np.random.randn(n_samples)
    result_np = _check_log_density(log_dens_np, "log_p", log_likelihood, n_samples, sample_dims)

    assert isinstance(result_np, xr.DataArray)
    assert all(dim in result_np.dims for dim in sample_dims)

    log_dens_da = xr.DataArray(
        np.random.randn(log_likelihood.chain.size, log_likelihood.draw.size),
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

    with pytest.raises(ValueError, match="Size of log_p .* must match"):
        _check_log_density(
            np.random.randn(n_samples - 1), "log_p", log_likelihood, n_samples, sample_dims
        )

    bad_dims_da = xr.DataArray(np.random.randn(n_samples), dims=["sample"])
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


def test_plpd_approx(centered_eight, log_lik_fn):
    result = _plpd_approx(
        data=centered_eight, var_name="obs", log_lik_fn=log_lik_fn, param_names=["theta"], log=True
    )

    assert isinstance(result, xr.DataArray)
    assert result.dims == centered_eight.observed_data.obs.dims
    assert result.shape == centered_eight.observed_data.obs.shape
    assert result.name == "plpd"


def test_plpd_approx_errors(centered_eight):
    with pytest.raises(TypeError, match="log_lik_fn must be a callable function"):
        _plpd_approx(centered_eight, var_name="obs", log_lik_fn="not_callable")

    def invalid_fn():
        return 0.0

    with pytest.raises(ValueError, match="Parameters .* not found in posterior group"):
        _plpd_approx(
            centered_eight, var_name="obs", log_lik_fn=invalid_fn, param_names=["missing_param"]
        )

    def bad_return_fn():
        return "not_numeric"

    with pytest.raises(RuntimeError):
        _plpd_approx(
            centered_eight, var_name="obs", log_lik_fn=bad_return_fn, param_names=["theta"]
        )


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


def test_prepare_subsample(centered_eight, log_lik_fn):
    log_likelihood_ds = get_log_likelihood_dataset(centered_eight, var_names="obs")
    log_likelihood_da = log_likelihood_ds["obs"]

    var_name = "obs"
    observations = 3
    seed = 42
    method = "lpd"
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

    method = "plpd"
    result_plpd = _prepare_subsample(
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

    assert result_plpd.subsample_size == observations
    assert isinstance(result_plpd.lpd_approx_all, xr.DataArray)
    assert result_plpd.lpd_approx_all.dims == ("school",)
