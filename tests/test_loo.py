# pylint: disable=redefined-outer-name, unused-import, unused-argument
# ruff: noqa: F811, F401
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal

from .helpers import (
    centered_eight,
    centered_eight_with_sigma,
    datatree,
    datatree_binary,
    importorskip,
    non_centered_eight,
)

azb = importorskip("arviz_base")
xr = importorskip("xarray")
sp = importorskip("scipy")

from arviz_stats import (
    compare,
    loo,
    loo_approximate_posterior,
    loo_expectations,
    loo_metrics,
    loo_moment_match,
    loo_pit,
    loo_subsample,
    update_subsample,
)
from arviz_stats.loo import _calculate_ics
from arviz_stats.loo.loo_expectations import _get_function_khat
from arviz_stats.utils import ELPDData, get_log_likelihood_dataset


@pytest.fixture(scope="module")
def multivariable_log_likelihood(centered_eight):
    centered_eight = centered_eight.copy()
    new_arr = xr.DataArray(
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
    with pytest.raises(TypeError, match="Encountered error trying to compute"):
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
    loo_exp_vals, _ = loo_expectations(centered_eight, kind=kind, probs=probs)

    if kind == "quantile":
        assert loo_exp_vals.shape == (2, 8)
    else:
        assert loo_exp_vals.shape == (8,)

    assert_almost_equal(loo_exp_vals.sel({"school": "Choate"}), expected_vals, decimal=2)


@pytest.mark.parametrize("kind", ["mean", "var", "quantile"])
def test_loo_expectations_khat(centered_eight, datatree, kind):
    probs = [0.25, 0.75] if kind == "quantile" else None
    result, khat = loo_expectations(centered_eight, kind=kind, probs=probs)

    assert np.all(np.isfinite(khat.values))
    assert np.all(khat.values >= -0.5) and np.all(khat.values <= 1.5)

    if kind == "quantile":
        expected_dims = tuple(d for d in result.dims if d != "quantile")
        assert khat.dims == expected_dims
        assert khat.shape == tuple(result.sizes[d] for d in expected_dims)
    else:
        assert khat.dims == result.dims
        assert khat.shape == result.shape

    assert len(np.unique(khat.values.flatten())) > 1

    for dim_name in khat.dims:
        khat_coord_vals = khat.coords[dim_name].values
        result_coord_vals = result.coords[dim_name].values
        if khat_coord_vals.dtype.kind in ("U", "S", "O"):
            assert np.array_equal(khat_coord_vals, result_coord_vals)
        else:
            assert_allclose(khat_coord_vals, result_coord_vals)

    _, khat_check = loo_expectations(datatree, var_name="y", kind=kind, probs=probs)
    n_samples = (
        datatree.log_likelihood["y"].sizes["chain"] * datatree.log_likelihood["y"].sizes["draw"]
    )
    good_k = min(1 - 1 / np.log10(n_samples), 0.7) if n_samples > 1 else 0.7
    if np.any(khat_check.values > good_k):
        with pytest.warns(UserWarning, match="Estimated shape parameter of Pareto distribution"):
            loo_expectations(datatree, var_name="y", kind=kind, probs=probs)


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
        ("acc", 2, 0.43, 0.19),
        ("acc_balanced", "2g", 0.42, 0.045),
    ],
)
def test_loo_metrics_acc(datatree_binary, kind, round_to, expected_mean, expected_se):
    metrics = loo_metrics(datatree_binary, kind=kind, round_to=round_to)
    assert_almost_equal(metrics.mean, expected_mean, decimal=4)
    assert_almost_equal(metrics.se, expected_se, decimal=4)


def test_loo_metrics_invalid_kind(centered_eight):
    with pytest.raises(ValueError, match="kind must be one of"):
        loo_metrics(centered_eight, kind="invalid_kind")


@pytest.mark.parametrize(
    "values_type, expected_behavior",
    [
        ("single_value", "returns_khat"),
        ("two_values", "returns_khat"),
        ("all_nan", "returns_khat"),
        ("all_inf", "returns_khat"),
        ("mixed_finite_nonfinite", "returns_khat"),
        ("extreme_weights", "returns_khat"),
        ("constant_tail_weights", "returns_khat"),
    ],
)
def test_get_function_khat(values_type, expected_behavior):
    rng = np.random.default_rng(seed=42)
    n_samples = 100
    log_weights = rng.normal(0, 0.1, n_samples)

    if values_type == "single_value":
        values = np.full(n_samples, 2.5)
    elif values_type == "two_values":
        values = np.array([1.0, 2.0] * (n_samples // 2))
    elif values_type == "all_nan":
        values = np.full(n_samples, np.nan)
    elif values_type == "all_inf":
        values = np.full(n_samples, np.inf)
    elif values_type == "mixed_finite_nonfinite":
        values = rng.normal(0, 1, n_samples)
        values[::10] = np.nan
        values[5::10] = np.inf
    elif values_type == "extreme_weights":
        log_weights = np.full(n_samples, -50.0)
        log_weights[-5:] = 0.0
        values = rng.normal(0, 1, n_samples)
    elif values_type == "constant_tail_weights":
        log_weights = np.zeros(n_samples)
        log_weights[-20:] = -1e-10
        values = rng.normal(0, 1, n_samples)

    khat = _get_function_khat(values, log_weights, kind="mean")

    if expected_behavior == "returns_khat":
        assert isinstance(khat, float | np.floating)


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"var_names": ["obs"]},
        {"log_weights": "arr"},
        {"var_names": ["obs"]},
        {"var_names": "obs"},
    ],
)
def test_loo_pit(centered_eight, args):
    var_names = args.get("var_names", None)
    log_weights = args.get("log_weights", None)

    if log_weights == "arr":
        log_weights = get_log_likelihood_dataset(centered_eight, var_names=var_names)

    loo_pit_values = loo_pit(
        centered_eight,
        var_names=var_names,
        log_weights=log_weights,
    )
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
def test_loo_subsample_loglik_errors(centered_eight, method):
    def log_lik_fn(obs_da, datatree):
        _ = datatree.posterior["missing_param"]
        return obs_da * 0

    with pytest.raises(KeyError, match="Variable not found in posterior"):
        loo_subsample(
            centered_eight,
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
            centered_eight,
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


def test_loo_subsample_validation_errors(centered_eight):
    with pytest.raises(ValueError, match="Number of observations must be between 1 and"):
        loo_subsample(centered_eight, observations=0, var_name="obs")
    with pytest.raises(TypeError, match="observations must be an integer"):
        loo_subsample(centered_eight, observations=4.2, var_name="obs")


@pytest.fixture(scope="function")
def loo_orig(datatree):
    return loo(datatree, pointwise=True, var_name="y")


@pytest.fixture(scope="module")
def moment_match_data(datatree):
    log_likelihood_data = get_log_likelihood_dataset(datatree, var_names="y")["y"]
    posterior_draws = azb.extract(datatree, group="posterior", combined=False)

    a_params = posterior_draws["theta"]
    b_param = posterior_draws["mu"]
    log_sigma_param = np.log(posterior_draws["tau"])

    param_list_for_ds = {}

    for hierarchy_name in a_params.coords["hierarchy"].values:
        fixed_name = f"theta_{hierarchy_name}"
        hierarchy_specific_a = a_params.sel(hierarchy=hierarchy_name).drop_vars("hierarchy")
        param_list_for_ds[fixed_name] = hierarchy_specific_a

    param_list_for_ds["b_main"] = b_param
    param_list_for_ds["log_sigma"] = log_sigma_param

    upars_ds = xr.Dataset(param_list_for_ds)
    upars_da_generated = azb.dataset_to_dataarray(
        upars_ds, sample_dims=["chain", "draw"], new_dim="unconstrained_parameter"
    )
    upars_da_final = upars_da_generated.transpose("chain", "draw", "unconstrained_parameter")

    def log_prob_upars_fn(upars_arg):
        return upars_arg.sum("unconstrained_parameter") * 0.01

    def log_lik_i_upars_fn(upars_arg, i):
        obs_dims = [dim for dim in log_likelihood_data.dims if dim not in ["chain", "draw"]]
        original_log_lik_for_i = log_likelihood_data.stack(i=obs_dims).isel(i=i)
        perturbation = upars_arg.sum("unconstrained_parameter") * 0.0001
        return original_log_lik_for_i + perturbation

    return upars_da_final, log_prob_upars_fn, log_lik_i_upars_fn, upars_ds


def test_loo_moment_match(datatree, loo_orig, moment_match_data):
    upars_da, log_prob_fn, log_lik_i_fn, _ = moment_match_data
    max_k = np.nanmax(loo_orig.pareto_k.values)
    k_threshold = np.nanpercentile(loo_orig.pareto_k.values, 90)
    assert k_threshold < max_k

    orig_bad_k_count = np.sum(loo_orig.pareto_k.values > k_threshold)
    assert orig_bad_k_count > 0

    loo_mm = loo_moment_match(
        datatree,
        loo_orig,
        upars=upars_da,
        log_prob_upars_fn=log_prob_fn,
        log_lik_i_upars_fn=log_lik_i_fn,
        k_threshold=k_threshold,
        var_name="y",
        cov=True,
        split=True,
        max_iters=50,
    )

    assert isinstance(loo_mm, ELPDData)
    assert loo_mm.method == "loo_moment_match"
    mm_bad_k_count = np.sum(loo_mm.pareto_k.values > k_threshold)
    assert mm_bad_k_count <= orig_bad_k_count
    assert loo_mm.elpd >= loo_orig.elpd

    assert hasattr(loo_mm, "p_loo_i"), "loo_mm object should have p_loo_i attribute"


def test_loo_moment_match_optional_upars(datatree, loo_orig, moment_match_data):
    upars_da, log_prob_fn, log_lik_i_fn, upars_ds = moment_match_data

    loo_mm_explicit = loo_moment_match(
        datatree,
        loo_orig,
        upars=upars_da,
        log_prob_upars_fn=log_prob_fn,
        log_lik_i_upars_fn=log_lik_i_fn,
        var_name="y",
        max_iters=2,
    )
    assert isinstance(loo_mm_explicit, ELPDData)

    datatree_with_unconstrained = datatree.copy()
    unconstrained_group = xr.DataTree(dataset=upars_ds, name="unconstrained_posterior")
    datatree_with_unconstrained["unconstrained_posterior"] = unconstrained_group

    loo_mm_implicit = loo_moment_match(
        datatree_with_unconstrained,
        loo_orig,
        upars=None,
        log_prob_upars_fn=log_prob_fn,
        log_lik_i_upars_fn=log_lik_i_fn,
        var_name="y",
        max_iters=2,
    )
    assert isinstance(loo_mm_implicit, ELPDData)

    with pytest.raises(
        ValueError,
        match="upars must be provided or data must contain an 'unconstrained_posterior' group",
    ):
        loo_moment_match(
            datatree,
            loo_orig,
            upars=None,
            log_prob_upars_fn=log_prob_fn,
            log_lik_i_upars_fn=log_lik_i_fn,
            var_name="y",
        )


def test_loo_moment_match_no_problematic_k(datatree, loo_orig, moment_match_data):
    upars_da, log_prob_fn, log_lik_i_fn, _ = moment_match_data
    k_threshold = np.nanmax(loo_orig.pareto_k.values) + 0.1

    loo_mm = loo_moment_match(
        datatree,
        loo_orig,
        upars=upars_da,
        log_prob_upars_fn=log_prob_fn,
        log_lik_i_upars_fn=log_lik_i_fn,
        k_threshold=k_threshold,
        var_name="y",
    )

    assert loo_mm.elpd == loo_orig.elpd
    assert_allclose(loo_mm.pareto_k.values, loo_orig.pareto_k.values)


def test_loo_moment_match_errors(datatree, moment_match_data):
    upars_da, log_prob_fn, log_lik_i_fn, _ = moment_match_data

    loo_non_pointwise = loo(datatree, pointwise=False, var_name="y")
    with pytest.raises(
        ValueError,
        match="Moment matching requires pointwise LOO results with Pareto k values. "
        "Please compute the initial LOO with pointwise=True",
    ):
        loo_moment_match(
            datatree,
            loo_non_pointwise,
            upars=upars_da,
            log_prob_upars_fn=log_prob_fn,
            log_lik_i_upars_fn=log_lik_i_fn,
            var_name="y",
        )

    loo_pointwise = loo(datatree, pointwise=True, var_name="y")
    bad_upars = xr.DataArray(np.random.randn(10), dims=["sample"])
    with pytest.raises(ValueError, match="upars must have dimensions"):
        loo_moment_match(
            datatree,
            loo_pointwise,
            upars=bad_upars,
            log_prob_upars_fn=log_prob_fn,
            log_lik_i_upars_fn=log_lik_i_fn,
            var_name="y",
        )


def test_loo_moment_match_function_errors(datatree, loo_orig):
    upars_da = xr.DataArray(
        np.random.randn(4, 500, 10),
        dims=["chain", "draw", "unconstrained_parameter"],
        coords={"chain": [0, 1, 2, 3], "draw": np.arange(500)},
    )

    def bad_log_prob_fn(upars):  # pylint: disable=unused-argument
        return xr.DataArray(np.array([1.0]))

    def good_log_lik_fn(upars, i):  # pylint: disable=unused-argument
        return xr.DataArray(
            np.random.randn(upars.chain.size, upars.draw.size),
            dims=["chain", "draw"],
            coords={"chain": upars.chain, "draw": upars.draw},
        )

    with pytest.raises((ValueError, IndexError)):
        loo_moment_match(
            datatree,
            loo_orig,
            upars=upars_da,
            log_prob_upars_fn=bad_log_prob_fn,
            log_lik_i_upars_fn=good_log_lik_fn,
            k_threshold=0.7,
            var_name="y",
        )


def test_log_weights_storage(centered_eight):
    loo_pw_true = loo(centered_eight, pointwise=True)
    assert loo_pw_true.log_weights is not None
    assert loo_pw_true.log_weights.shape == (8, 4, 500)

    loo_pw_false = loo(centered_eight, pointwise=False)
    assert loo_pw_false.log_weights is not None
    assert loo_pw_false.log_weights.shape == (8, 4, 500)

    loo_sub_pw_true = loo_subsample(centered_eight, observations=4, pointwise=True, seed=42)
    assert loo_sub_pw_true.log_weights is not None
    assert isinstance(loo_sub_pw_true.log_weights, xr.Dataset)
    assert "obs" in loo_sub_pw_true.log_weights
    assert loo_sub_pw_true.log_weights["obs"].shape == (4, 4, 500)

    loo_sub_pw_false = loo_subsample(centered_eight, observations=4, pointwise=False, seed=42)
    assert loo_sub_pw_false.log_weights is not None
    assert loo_sub_pw_false.log_weights.shape == (4, 4, 500)

    loo_updated = update_subsample(loo_sub_pw_true, centered_eight, observations=2, seed=43)
    assert loo_updated.log_weights is not None
    assert isinstance(loo_updated.log_weights, xr.Dataset)
    assert "obs" in loo_updated.log_weights
    assert loo_updated.log_weights["obs"].shape[0] >= 5

    loo_with_weights = loo(
        centered_eight,
        pointwise=True,
        log_weights=loo_pw_true.log_weights,
        pareto_k=loo_pw_true.pareto_k,
    )
    assert loo_with_weights.log_weights is not None
    assert_array_equal(loo_with_weights.log_weights.values, loo_pw_true.log_weights.values)


def test_log_weights_input_formats(centered_eight):
    loo_result = loo(centered_eight, pointwise=True)
    log_weights_da = loo_result.log_weights

    loo_exp_da, khat_da = loo_expectations(centered_eight, kind="mean", log_weights=log_weights_da)
    loo_exp_elpddata, khat_elpddata = loo_expectations(
        centered_eight, kind="mean", log_weights=loo_result
    )
    assert_array_equal(loo_exp_da.values, loo_exp_elpddata.values)
    assert_array_equal(khat_da.values, khat_elpddata.values)

    loo_pit_da = loo_pit(centered_eight, log_weights=log_weights_da)
    loo_pit_elpddata = loo_pit(centered_eight, log_weights=loo_result)
    assert_array_equal(loo_pit_da["obs"].values, loo_pit_elpddata["obs"].values)

    metrics_da = loo_metrics(centered_eight, kind="rmse", log_weights=log_weights_da)
    metrics_elpddata = loo_metrics(centered_eight, kind="rmse", log_weights=loo_result)
    assert metrics_da.mean == metrics_elpddata.mean


def test_log_weights_input_formats_subsample(centered_eight):
    loo_result = loo(centered_eight, pointwise=True)
    loo_sub_elpddata = loo_subsample(
        centered_eight,
        observations=4,
        pointwise=True,
        var_name="obs",
        log_weights=loo_result,
    )
    assert loo_sub_elpddata is not None
    assert loo_sub_elpddata.log_weights is not None
    assert loo_sub_elpddata.log_weights["obs"].shape == (4, 4, 500)


def test_log_weights_reuse(centered_eight):
    loo_result = loo(centered_eight, pointwise=True)
    assert loo_result.log_weights is not None
    assert loo_result.log_weights.shape == (8, 4, 500)
    assert loo_result.log_weights.values.shape == (8, 4, 500)

    loo_exp, khat = loo_expectations(
        centered_eight, kind="mean", log_weights=loo_result.log_weights
    )
    assert loo_exp is not None
    assert khat is not None
    assert loo_exp.shape == (8,)

    loo_pit_vals = loo_pit(centered_eight, log_weights=loo_result.log_weights)
    assert loo_pit_vals is not None
    assert "obs" in loo_pit_vals
    assert loo_pit_vals["obs"].shape == (8,)

    metrics = loo_metrics(centered_eight, kind="rmse", log_weights=loo_result.log_weights)
    assert metrics is not None
    assert hasattr(metrics, "mean")


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
