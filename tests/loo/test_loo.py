"""Test PSIS-LOO-CV."""

# pylint: disable=redefined-outer-name, unused-argument
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal

from ..helpers import importorskip

azb = importorskip("arviz_base")
xr = importorskip("xarray")
sp = importorskip("scipy")

from arviz_stats import loo, loo_i


@pytest.mark.parametrize("pointwise", [True, False])
def test_loo(centered_eight, pointwise):
    loo_data = loo(centered_eight, pointwise=pointwise)
    assert_almost_equal(loo_data.elpd, -30.78, decimal=1)
    assert_almost_equal(loo_data.p, 0.94, decimal=1)
    assert_almost_equal(loo_data.se, 1.34, decimal=1)
    assert_almost_equal(loo_data.good_k, 0.69, decimal=1)
    assert_almost_equal(loo_data.n_data_points, 8)
    assert_almost_equal(loo_data.n_samples, 2000)
    assert loo_data.warning is False
    assert loo_data.kind == "loo"
    assert loo_data.scale == "log"
    assert loo_data.log_weights is not None
    assert loo_data.log_weights.shape == (8, 4, 500)

    if pointwise:
        assert_almost_equal(
            loo_data.pareto_k, [0.43, 0.39, 0.49, 0.47, 0.44, 0.55, 0.31, 0.52], decimal=1
        )
        assert loo_data.elpd_i.shape == (8,)


def test_log_weights_storage(centered_eight):
    loo_pw_true = loo(centered_eight, pointwise=True)
    assert loo_pw_true.log_weights is not None
    assert loo_pw_true.log_weights.shape == (8, 4, 500)

    loo_pw_false = loo(centered_eight, pointwise=False)
    assert loo_pw_false.log_weights is not None
    assert loo_pw_false.log_weights.shape == (8, 4, 500)

    loo_with_weights = loo(
        centered_eight,
        pointwise=True,
        log_weights=loo_pw_true.log_weights,
        pareto_k=loo_pw_true.pareto_k,
    )
    assert loo_with_weights.log_weights is not None
    assert_array_equal(loo_with_weights.log_weights.values, loo_pw_true.log_weights.values)


def test_log_weights_reuse(centered_eight):
    loo_result = loo(centered_eight, pointwise=True)
    assert loo_result.log_weights is not None
    assert loo_result.log_weights.shape == (8, 4, 500)
    assert loo_result.log_weights.values.shape == (8, 4, 500)


def test_loo_i(centered_eight):
    loo_full = loo(centered_eight, pointwise=True)

    result_0 = loo_i(0, centered_eight)
    assert isinstance(result_0, type(loo_full))
    assert result_0.kind == "loo"
    assert result_0.n_data_points == 1
    assert result_0.n_samples == 2000
    assert_almost_equal(result_0.elpd, loo_full.elpd_i[0].item(), decimal=10)
    assert_almost_equal(result_0.pareto_k.item(), loo_full.pareto_k[0].item(), decimal=10)

    result_7 = loo_i(7, centered_eight)
    assert_almost_equal(result_7.elpd, loo_full.elpd_i[7].item(), decimal=10)
    assert_almost_equal(result_7.pareto_k.item(), loo_full.pareto_k[7].item(), decimal=10)

    schools = centered_eight.observed_data["obs"].coords["school"].values
    label0 = schools[0].item() if hasattr(schools[0], "item") else schools[0]

    result_label = loo_i({"school": label0}, centered_eight)
    assert_almost_equal(result_label.elpd, loo_full.elpd_i[0].item(), decimal=10)
    assert_almost_equal(result_label.pareto_k.item(), loo_full.pareto_k[0].item(), decimal=10)

    result_scalar = loo_i(label0, centered_eight)
    assert_almost_equal(result_scalar.elpd, loo_full.elpd_i[0].item(), decimal=10)
    assert_almost_equal(result_scalar.pareto_k.item(), loo_full.pareto_k[0].item(), decimal=10)

    with pytest.raises(IndexError):
        loo_i(-1, centered_eight)

    with pytest.raises(IndexError):
        loo_i(len(schools), centered_eight)

    with pytest.raises(KeyError):
        loo_i(3.5, centered_eight)

    with pytest.raises(ValueError, match=r"Provide selections for all observation dims:.*"):
        loo_i({"student": 1}, centered_eight)

    with pytest.raises(ValueError, match=r"Provide selections for all observation dims:.*"):
        loo_i({"school": label0, "student": 123}, centered_eight)

    with pytest.raises(KeyError):
        loo_i((), centered_eight)


def test_loo_i_with_log_lik_fn(centered_eight):
    sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])
    sigma_da = xr.DataArray(
        sigma, dims=["school"], coords={"school": centered_eight.observed_data.school.values}
    )
    centered_eight["constant_data"] = (
        centered_eight["constant_data"].to_dataset().assign(sigma=sigma_da)
    )

    def custom_log_lik(observed_i, data):
        school_coord = observed_i.coords["school"].values.item()
        y_value = observed_i.values.item()
        theta_samples = data.posterior["theta"].sel(school=school_coord)
        obs_sd = data.constant_data["sigma"].sel(school=school_coord).values

        return sp.stats.norm.logpdf(y_value, theta_samples.values, obs_sd)

    result_standard = loo_i(0, centered_eight, var_name="obs")
    result_custom = loo_i(0, centered_eight, var_name="obs", log_lik_fn=custom_log_lik)
    assert_almost_equal(result_standard.elpd, result_custom.elpd, decimal=10)
    assert_almost_equal(result_standard.pareto_k, result_custom.pareto_k, decimal=10)

    result_standard = loo_i({"school": "Choate"}, centered_eight, var_name="obs")
    result_custom = loo_i(
        {"school": "Choate"}, centered_eight, var_name="obs", log_lik_fn=custom_log_lik
    )
    assert_almost_equal(result_standard.elpd, result_custom.elpd, decimal=10)
    assert_almost_equal(result_standard.pareto_k, result_custom.pareto_k, decimal=10)

    def bad_log_lik_wrong_type(observed_i, data):
        return np.array([1, 2, 3])

    with pytest.raises(ValueError, match="dims and shape should match"):
        loo_i(0, centered_eight, var_name="obs", log_lik_fn=bad_log_lik_wrong_type)

    def bad_log_lik_wrong_dims(observed_i, data):
        return xr.DataArray(np.ones((10, 20)), dims=["wrong1", "wrong2"])

    with pytest.raises(ValueError, match="log_lik_fn must return DataArray with dimensions"):
        loo_i(0, centered_eight, var_name="obs", log_lik_fn=bad_log_lik_wrong_dims)

    with pytest.raises(TypeError, match="log_lik_fn must be a callable"):
        loo_i(0, centered_eight, var_name="obs", log_lik_fn="not a function")


def test_loo_i_numpy(centered_eight):
    xre = importorskip("xarray_einstats")

    data = centered_eight.copy(deep=True)
    del data["log_likelihood"]

    def numpy_log_lik(observed_i, datatree):
        school = observed_i.coords["school"].values.item()
        theta = datatree.posterior["theta"].sel(school=school)
        y_val = observed_i.values.item()
        return sp.stats.norm.logpdf(y_val, theta.values, scale=12.5)

    result = loo_i({"school": "Choate"}, data, var_name="obs", log_lik_fn=numpy_log_lik)

    assert isinstance(result, type(loo(centered_eight, pointwise=True)))
    assert result.log_weights.dims == ("chain", "draw")

    observed = data.observed_data["obs"].sel({"school": "Choate"})
    theta = data.posterior["theta"].sel({"school": "Choate"})
    log_lik_array = numpy_log_lik(observed, data)
    log_lik_da = xr.DataArray(
        log_lik_array,
        dims=("chain", "draw"),
        coords={"chain": theta.coords["chain"], "draw": theta.coords["draw"]},
    )

    expected_elpd = xre.stats.logsumexp(
        result.log_weights + log_lik_da, dims=["chain", "draw"]
    ).item()

    assert_almost_equal(result.elpd, expected_elpd, decimal=10)


def test_loo_jacobian(centered_eight):
    loo_no_jacobian = loo(centered_eight, pointwise=True)

    y_obs = centered_eight.observed_data["obs"].values
    y_positive = y_obs + np.abs(y_obs.min()) + 1

    log_jacobian_values = -np.log(2) - 0.5 * np.log(y_positive)
    log_jacobian = xr.DataArray(
        log_jacobian_values,
        dims=["school"],
        coords={"school": centered_eight.observed_data["obs"].coords["school"]},
    )
    loo_with_jacobian = loo(centered_eight, pointwise=True, log_jacobian=log_jacobian)

    elpd_i_no_jac = loo_no_jacobian.elpd_i.values
    elpd_i_with_jac = loo_with_jacobian.elpd_i.values
    expected_elpd_i = elpd_i_no_jac + log_jacobian_values
    assert_allclose(elpd_i_with_jac, expected_elpd_i, rtol=1e-10)

    expected_elpd = loo_no_jacobian.elpd + np.sum(log_jacobian_values)
    assert_almost_equal(loo_with_jacobian.elpd, expected_elpd, decimal=10)

    n_obs = len(y_obs)
    elpd_se_adjusted = (n_obs * np.var(elpd_i_with_jac)) ** 0.5

    assert_almost_equal(loo_with_jacobian.se, elpd_se_adjusted, decimal=10)
    assert_almost_equal(loo_with_jacobian.p, loo_no_jacobian.p, decimal=10)

    jacobian_da_reordered = xr.DataArray(
        log_jacobian_values[::-1],
        dims=["school"],
        coords={"school": centered_eight.log_likelihood["obs"].coords["school"][::-1]},
    )
    loo_with_jacobian_reordered = loo(
        centered_eight, pointwise=True, log_jacobian=jacobian_da_reordered
    )
    assert_allclose(loo_with_jacobian_reordered.elpd_i.values, expected_elpd_i, rtol=1e-10)

    assert loo_with_jacobian.elpd != loo_no_jacobian.elpd

    loo_no_pointwise = loo(centered_eight, pointwise=False, log_jacobian=log_jacobian)
    assert_almost_equal(loo_no_pointwise.elpd, expected_elpd, decimal=10)

    wrong_type_jacobian = np.ones(8)
    with pytest.raises(TypeError, match="log_jacobian must be an xarray.DataArray"):
        loo(centered_eight, pointwise=True, log_jacobian=wrong_type_jacobian)

    wrong_dims_jacobian = xr.DataArray(
        np.ones(5),
        dims=["wrong_dim"],
        coords={"wrong_dim": np.arange(5)},
    )
    with pytest.raises(ValueError, match="Missing dimensions"):
        loo(centered_eight, pointwise=True, log_jacobian=wrong_dims_jacobian)

    nan_jacobian = xr.DataArray(
        np.array([np.nan] + [0.1] * 7),
        dims=["school"],
        coords={"school": centered_eight.observed_data["obs"].coords["school"]},
    )
    with pytest.raises(ValueError, match="must contain only finite values"):
        loo(centered_eight, pointwise=True, log_jacobian=nan_jacobian)


def test_loo_log_weights_pareto_k_mismatch(centered_eight):
    loo_result = loo(centered_eight, pointwise=True)

    with pytest.raises(ValueError, match="Both log_weights and pareto_k must be provided together"):
        loo(centered_eight, log_weights=loo_result.log_weights, pareto_k=None)

    with pytest.raises(ValueError, match="Both log_weights and pareto_k must be provided together"):
        loo(centered_eight, log_weights=None, pareto_k=loo_result.pareto_k)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_loo_single_chain():
    rng = np.random.default_rng(42)

    single_chain_data = azb.from_dict(
        {
            "posterior": {"theta": rng.normal(size=(1, 100, 8))},
            "log_likelihood": {"obs": rng.normal(size=(1, 100, 8))},
            "observed_data": {"obs": rng.normal(size=8)},
        }
    )

    result = loo(single_chain_data, pointwise=True)

    assert result.n_samples == 100
    assert result.elpd is not None
    assert result.pareto_k is not None
    assert result.p is not None


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("reff_value", [0.5, 0.8, 1.0])
def test_loo_custom_reff(centered_eight, reff_value):
    result = loo(centered_eight, reff=reff_value, pointwise=True)

    assert result.elpd is not None
    assert result.p is not None
    assert result.kind == "loo"
    assert result.pareto_k is not None


def test_loo_i_with_precomputed_weights(centered_eight):
    loo_full = loo(centered_eight, pointwise=True)

    i = 3
    result_with_weights = loo_i(
        i, centered_eight, log_weights=loo_full.log_weights, pareto_k=loo_full.pareto_k
    )

    result_direct = loo_i(i, centered_eight)

    assert_almost_equal(result_with_weights.elpd, result_direct.elpd, decimal=10)
    assert_almost_equal(result_with_weights.pareto_k, result_direct.pareto_k, decimal=10)


def test_loo_i_jacobian(centered_eight):
    y_obs = centered_eight.observed_data["obs"].values
    y_positive = y_obs + np.abs(y_obs.min()) + 1

    log_jacobian_values = -np.log(2) - 0.5 * np.log(y_positive)
    log_jacobian = xr.DataArray(
        log_jacobian_values,
        dims=["school"],
        coords={"school": centered_eight.observed_data["obs"].coords["school"]},
    )

    result_no_jac = loo_i(0, centered_eight)
    result_with_jac = loo_i(0, centered_eight, log_jacobian=log_jacobian)

    expected_elpd = result_no_jac.elpd + log_jacobian_values[0]
    assert_almost_equal(result_with_jac.elpd, expected_elpd, decimal=10)
    assert_almost_equal(result_with_jac.p, result_no_jac.p, decimal=10)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_loo_multidimensional_observations():
    rng = np.random.default_rng(42)

    multi_dim_data = azb.from_dict(
        {
            "posterior": {"mu": rng.normal(size=(4, 100))},
            "log_likelihood": {"y": rng.normal(size=(4, 100, 3, 4))},
            "observed_data": {"y": rng.normal(size=(3, 4))},
        }
    )

    result = loo(multi_dim_data, pointwise=True)

    assert result.n_data_points == 12
    assert result.elpd_i.shape == (3, 4)
    assert result.pareto_k.shape == (3, 4)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_loo_i_multidimensional_indexing():
    rng = np.random.default_rng(42)

    multi_dim_data = azb.from_dict(
        {
            "posterior": {"mu": rng.normal(size=(4, 100))},
            "log_likelihood": {"y": rng.normal(size=(4, 100, 3, 4))},
            "observed_data": {"y": rng.normal(size=(3, 4))},
        }
    )

    result_int = loo_i(5, multi_dim_data)
    assert result_int.n_data_points == 1
    assert result_int.elpd is not None

    result_dict = loo_i({"y_dim_0": 1, "y_dim_1": 2}, multi_dim_data)
    assert result_dict.n_data_points == 1
    assert result_dict.elpd is not None

    loo_full = loo(multi_dim_data, pointwise=True)
    expected_elpd = loo_full.elpd_i.isel(y_dim_0=1, y_dim_1=2).item()
    assert_almost_equal(result_dict.elpd, expected_elpd, decimal=10)


def test_loo_var_name_explicit(centered_eight):
    result_explicit = loo(centered_eight, var_name="obs", pointwise=True)
    result_auto = loo(centered_eight, pointwise=True)

    assert_almost_equal(result_explicit.elpd, result_auto.elpd, decimal=10)
    assert_array_equal(result_explicit.pareto_k.values, result_auto.pareto_k.values)


def test_loo_consistency_between_pointwise_modes(centered_eight):
    result_pointwise_true = loo(centered_eight, pointwise=True)
    result_pointwise_false = loo(centered_eight, pointwise=False)

    assert_almost_equal(result_pointwise_true.elpd, result_pointwise_false.elpd, decimal=10)
    assert_almost_equal(result_pointwise_true.p, result_pointwise_false.p, decimal=10)
    assert_almost_equal(result_pointwise_true.se, result_pointwise_false.se, decimal=10)

    assert result_pointwise_true.elpd_i is not None
    assert result_pointwise_true.pareto_k is not None
    assert result_pointwise_false.elpd_i is None
    assert result_pointwise_false.pareto_k is None


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_loo_with_extreme_values():
    rng = np.random.default_rng(42)

    extreme_data = azb.from_dict(
        {
            "posterior": {"theta": rng.normal(size=(2, 50, 5))},
            "log_likelihood": {"obs": rng.normal(size=(2, 50, 5)) * 10 - 100},
            "observed_data": {"obs": rng.normal(size=5)},
        }
    )

    result = loo(extreme_data, pointwise=True)

    assert np.isfinite(result.elpd)
    assert np.all(np.isfinite(result.elpd_i.values))
    assert np.all(np.isfinite(result.pareto_k.values))


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_loo_i_with_different_var_names():
    rng = np.random.default_rng(42)

    multi_var_data = azb.from_dict(
        {
            "posterior": {"mu": rng.normal(size=(2, 50))},
            "log_likelihood": {
                "y1": rng.normal(size=(2, 50, 5)),
                "y2": rng.normal(size=(2, 50, 3)),
            },
            "observed_data": {
                "y1": rng.normal(size=5),
                "y2": rng.normal(size=3),
            },
        }
    )

    result_y1 = loo_i(0, multi_var_data, var_name="y1")
    assert result_y1.n_data_points == 1
    assert result_y1.elpd is not None

    result_y2 = loo_i(0, multi_var_data, var_name="y2")
    assert result_y2.n_data_points == 1
    assert result_y2.elpd is not None

    assert result_y1.elpd != result_y2.elpd
