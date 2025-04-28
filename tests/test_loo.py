# pylint: disable=redefined-outer-name

import numpy as np
import pytest
import xarray as xr
from arviz_base import load_arviz_data
from numpy.testing import assert_allclose, assert_almost_equal
from xarray import DataArray

from arviz_stats import (
    compare,
    loo,
    loo_approximate_posterior,
    loo_expectations,
    loo_metrics,
    loo_pit,
    loo_subsample,
    update_loo_subsample,
)
from arviz_stats.loo import _calculate_ics, _diff_srs_estimator
from arviz_stats.utils import ELPDData, get_log_likelihood_dataset


@pytest.fixture(name="centered_eight", scope="session")
def fixture_centered_eight():
    return load_arviz_data("centered_eight")


@pytest.fixture(name="non_centered_eight", scope="session")
def fixture_non_centered_eight():
    return load_arviz_data("non_centered_eight")


@pytest.fixture(name="anes", scope="session")
def fixture_anes():
    return load_arviz_data("anes")


@pytest.fixture(name="radon", scope="session")
def fixture_radon():
    return load_arviz_data("radon")


@pytest.fixture(scope="module")
def multivariable_log_likelihood(centered_eight):
    centered_eight = centered_eight.copy()
    new_arr = DataArray(
        np.zeros(centered_eight.log_likelihood["obs"].values.shape),
        dims=["chain", "draw", "school"],
        coords=centered_eight.log_likelihood.coords,
    )
    centered_eight.log_likelihood["decoy"] = new_arr
    return centered_eight


@pytest.fixture(scope="module")
def log_densities(centered_eight):
    log_lik = get_log_likelihood_dataset(centered_eight, var_names="obs")
    rng = np.random.default_rng(seed=42)

    p_values = rng.normal(size=(log_lik.chain.size, log_lik.draw.size))
    q_values = rng.normal(loc=-1.0, size=(log_lik.chain.size, log_lik.draw.size))

    log_p_da = xr.DataArray(
        p_values,
        dims=["chain", "draw"],
        coords={"chain": log_lik.chain, "draw": log_lik.draw},
    )
    log_q_da = xr.DataArray(
        q_values,
        dims=["chain", "draw"],
        coords={"chain": log_lik.chain, "draw": log_lik.draw},
    )

    log_p_np = p_values.ravel()
    log_q_np = q_values.ravel()

    return {
        "log_lik": log_lik,
        "dataarray": (log_p_da, log_q_da),
        "numpy": (log_p_np, log_q_np),
    }


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
    if pointwise:
        assert_almost_equal(
            loo_data.pareto_k, [0.43, 0.39, 0.49, 0.47, 0.44, 0.55, 0.31, 0.52], decimal=1
        )
        assert loo_data.elpd_i.shape == (8,)


@pytest.mark.parametrize("method", ["stacking", "BB-pseudo-BMA", "pseudo-BMA"])
def test_compare_same(centered_eight, method):
    data_dict = {"first": centered_eight, "second": centered_eight}

    weight = compare(data_dict, method=method)["weight"].to_numpy()
    assert_allclose(weight[0], weight[1])
    assert_allclose(np.sum(weight), 1.0)


def test_compare_unknown_method(centered_eight, non_centered_eight):
    model_dict = {"centered": centered_eight, "non_centered": non_centered_eight}
    with pytest.raises(ValueError):
        compare(model_dict, method="Unknown")


@pytest.mark.parametrize("method", ["stacking", "BB-pseudo-BMA", "pseudo-BMA"])
def test_compare_different(centered_eight, non_centered_eight, method):
    model_dict = {"centered": centered_eight, "non_centered": non_centered_eight}
    weight = compare(model_dict, method=method)["weight"]
    assert weight["non_centered"] > weight["centered"]
    assert_allclose(np.sum(weight), 1.0)


def test_compare_different_size(centered_eight):
    centered_eight_subset = centered_eight.sel(school="Choate")
    model_dict = {"centered": centered_eight, "centered__subset": centered_eight_subset}
    with pytest.raises(ValueError):
        compare(model_dict)


def test_compare_multiple_obs(multivariable_log_likelihood, centered_eight, non_centered_eight):
    compare_dict = {
        "centered_eight": centered_eight,
        "non_centered_eight": non_centered_eight,
        "problematic": multivariable_log_likelihood,
    }
    with pytest.raises(ValueError, match="Encountered error trying to compute"):
        compare(
            compare_dict,
        )
    assert compare(compare_dict, var_name="obs") is not None


def test_calculate_ics(centered_eight, non_centered_eight):
    idata_dict = {"centered": centered_eight, "non_centered": non_centered_eight}
    elpddata_dict = {key: loo(value) for key, value in idata_dict.items()}
    mixed_dict = {"centered": idata_dict["centered"], "non_centered": elpddata_dict["non_centered"]}
    idata_out = _calculate_ics(idata_dict)
    elpddata_out = _calculate_ics(elpddata_dict)
    mixed_out = _calculate_ics(mixed_dict)
    for model in idata_dict:
        assert idata_out[model]["elpd"] == elpddata_out[model]["elpd"]
        assert idata_out[model]["elpd"] == mixed_out[model]["elpd"]
        assert idata_out[model]["p"] == elpddata_out[model]["p"]
        assert idata_out[model]["p"] == mixed_out[model]["p"]


def test_calculate_ics_pointwise_error(centered_eight, non_centered_eight):
    in_dict = {
        "centered": loo(centered_eight, pointwise=True),
        "non_centered": loo(non_centered_eight, pointwise=False),
    }
    with pytest.raises(ValueError, match="should have been calculated with pointwise=True"):
        _calculate_ics(in_dict)


@pytest.mark.parametrize(
    "kind, probs, expected_vals",
    [
        ("mean", None, 3.81),
        ("quantile", [0.25, 0.75], [-6.26, 14.44]),
    ],
)
def test_loo_expectations(centered_eight, kind, probs, expected_vals):
    loo_exp_vals = loo_expectations(centered_eight, kind=kind, probs=probs)

    if kind == "quantile":
        assert loo_exp_vals.shape == (2, 8)
    else:
        assert loo_exp_vals.shape == (8,)

    assert_almost_equal(loo_exp_vals.sel({"school": "Choate"}), expected_vals, decimal=2)


@pytest.mark.parametrize(
    "kind, round_to, expected_mean, expected_se",
    [
        ("mae", 2, 8.78, 2.43),
        ("mse", "2g", 120.0, 66.0),
        ("rmse", None, 11.1565, 2.9405),
    ],
)
def test_loo_metrics(centered_eight, kind, round_to, expected_mean, expected_se):
    metrics = loo_metrics(centered_eight, kind=kind, round_to=round_to)
    assert_almost_equal(metrics.mean, expected_mean, decimal=4)
    assert_almost_equal(metrics.se, expected_se, decimal=4)


@pytest.mark.parametrize(
    "kind, round_to, expected_mean, expected_se",
    [
        ("acc", 2, 0.82, 0.02),
        ("acc_balanced", "2g", 0.81, 0.0039),
    ],
)
def test_loo_metrics_acc(anes, kind, round_to, expected_mean, expected_se):
    metrics = loo_metrics(anes, kind=kind, round_to=round_to)
    assert_almost_equal(metrics.mean, expected_mean, decimal=4)
    assert_almost_equal(metrics.se, expected_se, decimal=4)


def test_loo_metrics_invalid_kind(centered_eight):
    with pytest.raises(ValueError, match="kind must be one of"):
        loo_metrics(centered_eight, kind="invalid_kind")


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"var_names": ["obs"]},
        {"log_weights": "arr"},
    ],
)
def test_loo_pit(centered_eight, args):
    var_names = args.get("var_names", None)
    log_weights = args.get("log_weights", None)
    if log_weights == "arr":
        log_weights = get_log_likelihood_dataset(centered_eight, var_names=var_names)

    loo_pit_values = loo_pit(centered_eight, var_names=var_names, log_weights=log_weights)
    assert np.all(loo_pit_values >= 0)
    assert np.all(loo_pit_values <= 1)


def test_loo_pit_discrete(centered_eight):
    centered_eight.observed_data["obs"] = centered_eight.observed_data["obs"].astype(int)

    loo_pit_values = loo_pit(centered_eight)
    assert np.all(loo_pit_values >= 0)
    assert np.all(loo_pit_values <= 1)


@pytest.mark.parametrize("input_type", ["dataarray", "numpy"])
def test_loo_approx_basic(centered_eight, log_densities, input_type):
    log_p, log_q = log_densities[input_type]
    log_lik = log_densities["log_lik"]

    n_samples = log_lik.chain.size * log_lik.draw.size
    n_data_points = np.prod(
        [log_lik[dim].size for dim in log_lik.dims if dim not in ["chain", "draw"]]
    )

    result = loo_approximate_posterior(centered_eight, log_p=log_p, log_q=log_q, var_name="obs")

    assert result.kind == "loo"
    assert result.n_samples == n_samples
    assert result.n_data_points == n_data_points
    assert isinstance(result.elpd, float)
    assert isinstance(result.se, float)
    assert isinstance(result.p, float)
    assert isinstance(result.warning, bool)


@pytest.mark.parametrize("pointwise", [True, False])
def test_loo_approx_pointwise(centered_eight, log_densities, pointwise):
    log_p, log_q = log_densities["dataarray"]
    log_lik = log_densities["log_lik"]
    n_data_points = np.prod(
        [log_lik[dim].size for dim in log_lik.dims if dim not in ["chain", "draw"]]
    )

    result = loo_approximate_posterior(
        centered_eight,
        log_p=log_p,
        log_q=log_q,
        pointwise=pointwise,
        var_name="obs",
    )

    if pointwise:
        assert result.elpd_i is not None
        assert result.pareto_k is not None
        assert result.elpd_i.shape == (n_data_points,)
        assert result.pareto_k.shape == (n_data_points,)
    else:
        assert result.elpd_i is None
        assert result.pareto_k is None


@pytest.mark.parametrize(
    "error_case,error_type,error_match",
    [
        ("wrong_type", TypeError, None),
        ("length_mismatch", ValueError, "Size of log_p"),
        ("missing_dims", ValueError, "must have dimension 'chain'"),
        ("dim_size_mismatch", ValueError, r"Size of dimension 'chain' in log_p"),
    ],
)
def test_loo_approx_errors(centered_eight, log_densities, error_case, error_type, error_match):
    log_p_da, log_q_da = log_densities["dataarray"]
    log_p_np, log_q_np = log_densities["numpy"]
    log_lik = log_densities["log_lik"]

    kwargs = {}

    if error_case == "wrong_type":
        kwargs = {"log_p": list(log_p_np), "log_q": log_q_np}

    elif error_case == "length_mismatch":
        kwargs = {"log_p": np.random.randn(log_p_np.size - 1), "log_q": log_q_np}

    elif error_case == "missing_dims":
        broken_p = xr.DataArray(log_p_da.values.reshape(-1), dims=["sample"])
        kwargs = {"log_p": broken_p, "log_q": log_q_da}

    elif error_case == "dim_size_mismatch":
        mismatched_p_values = np.random.randn(log_lik.chain.size - 1, log_lik.draw.size)
        mismatched_p = xr.DataArray(
            mismatched_p_values,
            dims=["chain", "draw"],
            coords={
                "chain": log_lik.chain[:-1],
                "draw": log_lik.draw,
            },
        )
        kwargs = {"log_p": mismatched_p, "log_q": log_q_da}

    with pytest.raises(error_type, match=error_match):
        loo_approximate_posterior(centered_eight, var_name="obs", **kwargs)


@pytest.mark.parametrize("pointwise", [True, False])
def test_loo_subsample(radon, pointwise):
    observations = 200
    loo_sub = loo_subsample(radon, observations=observations, pointwise=pointwise, var_name="y")

    assert isinstance(loo_sub, ELPDData)
    assert loo_sub.kind == "loo"
    assert loo_sub.subsample_size == observations
    assert isinstance(loo_sub.elpd, float)
    assert isinstance(loo_sub.se, float) and loo_sub.se >= 0
    assert isinstance(loo_sub.p, float)
    assert isinstance(loo_sub.subsampling_se, float) and loo_sub.subsampling_se >= 0
    assert loo_sub.n_data_points == radon.observed_data.y.size

    if pointwise:
        assert hasattr(loo_sub, "elpd_i")
        assert hasattr(loo_sub, "pareto_k")
        assert loo_sub.elpd_i is not None
        assert loo_sub.pareto_k is not None
        assert loo_sub.elpd_i.dims == ("obs_id",)
        assert loo_sub.elpd_i.shape == (loo_sub.n_data_points,)
        assert loo_sub.pareto_k.dims == ("obs_id_subsample",)
        assert loo_sub.pareto_k.shape == (observations,)
        assert np.isnan(loo_sub.elpd_i).sum() == loo_sub.n_data_points - observations
        assert not np.isnan(loo_sub.elpd_i).all()
    else:
        assert not hasattr(loo_sub, "elpd_i") or loo_sub.elpd_i is None
        assert not hasattr(loo_sub, "pareto_k") or loo_sub.pareto_k is None


@pytest.mark.parametrize("input_type", ["dataarray", "numpy"])
@pytest.mark.parametrize("pointwise", [True, False])
def test_loo_subsample_approx_posterior(radon, log_densities, input_type, pointwise):
    observations = 200
    log_p, log_q = log_densities[input_type]

    loo_sub_approx = loo_subsample(
        radon,
        observations=observations,
        log_p=log_p,
        log_q=log_q,
        pointwise=pointwise,
        var_name="y",
    )

    assert isinstance(loo_sub_approx, ELPDData)
    assert loo_sub_approx.kind == "loo"
    assert loo_sub_approx.subsample_size == observations
    assert isinstance(loo_sub_approx.elpd, float)
    assert isinstance(loo_sub_approx.se, float) and loo_sub_approx.se >= 0
    assert isinstance(loo_sub_approx.p, float)
    assert isinstance(loo_sub_approx.subsampling_se, float) and loo_sub_approx.subsampling_se >= 0
    assert loo_sub_approx.n_data_points == radon.observed_data.y.size

    if pointwise:
        assert hasattr(loo_sub_approx, "elpd_i")
        assert hasattr(loo_sub_approx, "pareto_k")
        assert loo_sub_approx.elpd_i is not None
        assert loo_sub_approx.pareto_k is not None
        assert loo_sub_approx.elpd_i.dims == ("obs_id",)
        assert loo_sub_approx.elpd_i.shape == (loo_sub_approx.n_data_points,)
        assert loo_sub_approx.pareto_k.dims == ("obs_id_subsample",)
        assert loo_sub_approx.pareto_k.shape == (observations,)
        assert np.isnan(loo_sub_approx.elpd_i).sum() == loo_sub_approx.n_data_points - observations
        assert not np.isnan(loo_sub_approx.elpd_i).all()
    else:
        assert not hasattr(loo_sub_approx, "elpd_i") or loo_sub_approx.elpd_i is None
        assert not hasattr(loo_sub_approx, "pareto_k") or loo_sub_approx.pareto_k is None


def test_difference_estimator():
    n_data_points = 10
    subsample_size = 4
    elpd_loo_i_sample = np.array([-1.0, -1.5, -0.5, -1.2])
    lpd_approx_sample = np.array([-0.9, -1.4, -0.4, -1.1])
    lpd_approx_all = np.array([-0.9, -1.4, -0.4, -1.1, -1.0, -1.3, -0.6, -1.0, -0.8, -1.5])

    expected_elpd_loo_hat = -11.0
    expected_subsampling_se = 0.0
    expected_se = np.sqrt(0.98)

    elpd_loo_hat, subsampling_se, se = _diff_srs_estimator(
        elpd_loo_i_sample=elpd_loo_i_sample,
        lpd_approx_sample=lpd_approx_sample,
        lpd_approx_all=lpd_approx_all,
        n_data_points=n_data_points,
        subsample_size=subsample_size,
    )

    assert_allclose(elpd_loo_hat, expected_elpd_loo_hat)
    assert_almost_equal(subsampling_se, expected_subsampling_se)
    assert_allclose(se, expected_se)


def test_loo_subsample_errors(radon):
    n_total = radon.observed_data.y.size
    with pytest.raises(ValueError, match="Number of observations must be between 1 and"):
        loo_subsample(radon, observations=0, var_name="y")
    with pytest.raises(ValueError, match="Number of observations must be between 1 and"):
        loo_subsample(radon, observations=n_total + 1, var_name="y")
    with pytest.raises(TypeError, match="observations must be an integer"):
        loo_subsample(radon, observations=50.5, var_name="y")


def test_update_loo_subsample(radon):
    initial_observations = 100
    initial_loo = loo_subsample(radon, observations=initial_observations, var_name="y")

    additional_observations = 400
    updated_loo = update_loo_subsample(
        initial_loo, radon, observations=additional_observations, var_name="y"
    )

    assert isinstance(updated_loo, ELPDData)
    assert updated_loo.kind == "loo"
    assert updated_loo.subsample_size == initial_observations + additional_observations
    assert isinstance(updated_loo.elpd, float)
    assert isinstance(updated_loo.se, float) and updated_loo.se >= 0
    assert isinstance(updated_loo.p, float)
    assert isinstance(updated_loo.subsampling_se, float) and updated_loo.subsampling_se >= 0
    assert np.isfinite(updated_loo.se)
    assert updated_loo.subsampling_se <= initial_loo.subsampling_se
    assert updated_loo.elpd_i is not None
    assert updated_loo.pareto_k is not None
    assert updated_loo.elpd_i.dims == ("obs_id",)
    assert updated_loo.elpd_i.shape == (updated_loo.n_data_points,)
    assert updated_loo.pareto_k.dims == ("obs_id_subsample",)
    assert updated_loo.pareto_k.shape == (updated_loo.subsample_size,)
    assert np.sum(~np.isnan(updated_loo.elpd_i.values)) == updated_loo.subsample_size


def test_update_loo_subsample_pointwise_false(radon):
    initial_observations = 100
    initial_loo_no_pointwise = loo_subsample(
        radon, observations=initial_observations, pointwise=False, var_name="y"
    )

    additional_observations = 50
    with pytest.raises(ValueError, match="Original loo_subsample result must have pointwise=True"):
        update_loo_subsample(
            initial_loo_no_pointwise, radon, observations=additional_observations, var_name="y"
        )


@pytest.mark.parametrize("input_type", ["dataarray", "numpy"])
def test_update_loo_subsample_approx_posterior(radon, log_densities, input_type):
    log_p, log_q = log_densities[input_type]
    initial_observations = 100
    initial_loo = loo_subsample(
        radon,
        observations=initial_observations,
        var_name="y",
        log_p=log_p,
        log_q=log_q,
    )

    additional_observations = 50
    updated_with_implicit = update_loo_subsample(
        initial_loo, radon, observations=additional_observations, var_name="y"
    )

    assert isinstance(updated_with_implicit, ELPDData)
    assert updated_with_implicit.subsample_size == initial_observations + additional_observations
    assert updated_with_implicit.approx_posterior

    assert_allclose(updated_with_implicit.elpd, updated_with_implicit.elpd, rtol=1e-10)
    assert_allclose(updated_with_implicit.se, updated_with_implicit.se, rtol=1e-10)


def test_update_loo_subsample_errors(radon):
    initial_observations = 100
    initial_loo = loo_subsample(radon, observations=initial_observations, var_name="y")

    n_total = radon.observed_data.y.size
    with pytest.raises(ValueError, match="Cannot add 919 observations when only 819 are available"):
        update_loo_subsample(initial_loo, radon, observations=n_total)

    existing_indices = np.where(~np.isnan(initial_loo.elpd_i.values.flatten()))[0]
    with pytest.raises(ValueError, match="New indices .* overlap with existing indices"):
        update_loo_subsample(initial_loo, radon, observations=existing_indices[:5])
