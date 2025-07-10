# pylint: disable=redefined-outer-name, unused-import
# ruff: noqa: F811
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal

from .helpers import datatree, datatree_binary, importorskip  # noqa: F401

azb = importorskip("arviz_base")
xr = importorskip("xarray")

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
from arviz_stats.utils import ELPDData, get_log_likelihood_dataset


@pytest.fixture(name="centered_eight", scope="session")
def fixture_centered_eight():
    return azb.load_arviz_data("centered_eight")


@pytest.fixture(name="non_centered_eight", scope="session")
def fixture_non_centered_eight():
    return azb.load_arviz_data("non_centered_eight")


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


@pytest.fixture(scope="module")
def log_lik_fn():
    def _log_likelihood_eight_schools(data, theta):
        sigma = 12.5
        log_lik = -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * ((data - theta) / sigma) ** 2
        return log_lik

    return _log_likelihood_eight_schools


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
def test_loo_subsample(centered_eight, pointwise, method, log_lik_fn):
    observations = 4
    loo_sub = loo_subsample(
        centered_eight,
        observations=observations,
        pointwise=pointwise,
        var_name="obs",
        method=method,
        log_lik_fn=log_lik_fn,
        param_names=["theta"],
    )

    assert isinstance(loo_sub, ELPDData)
    assert loo_sub.kind == "loo"
    assert loo_sub.subsample_size == observations
    assert isinstance(loo_sub.elpd, float)
    assert isinstance(loo_sub.se, float) and loo_sub.se >= 0
    assert isinstance(loo_sub.p, float)
    assert isinstance(loo_sub.subsampling_se, float) and loo_sub.subsampling_se >= 0
    assert loo_sub.n_data_points == centered_eight.observed_data.obs.size

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
    else:
        assert not hasattr(loo_sub, "elpd_i") or loo_sub.elpd_i is None
        assert not hasattr(loo_sub, "pareto_k") or loo_sub.pareto_k is None


@pytest.mark.parametrize("method", ["lpd", "plpd"])
def test_update_loo_subsample(centered_eight, method, log_lik_fn):
    initial_observations = 3
    initial_loo = loo_subsample(
        centered_eight,
        observations=initial_observations,
        var_name="obs",
        method=method,
        log_lik_fn=log_lik_fn,
        param_names=["theta"],
    )

    additional_observations = 5
    updated_loo = update_subsample(
        initial_loo,
        centered_eight,
        observations=additional_observations,
        var_name="obs",
        method=method,
        log_lik_fn=log_lik_fn,
        param_names=["theta"],
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
    assert updated_loo.elpd_i.dims == ("school",)
    assert updated_loo.elpd_i.shape == (updated_loo.n_data_points,)
    assert updated_loo.pareto_k.dims == updated_loo.elpd_i.dims
    assert updated_loo.pareto_k.shape == (updated_loo.n_data_points,)
    assert np.sum(~np.isnan(updated_loo.elpd_i.values)) == updated_loo.subsample_size
    assert np.sum(~np.isnan(updated_loo.pareto_k.values)) == updated_loo.subsample_size


def test_loo_subsample_errors(centered_eight):
    n_total = centered_eight.observed_data.obs.size
    with pytest.raises(ValueError, match="Number of observations must be between 1 and"):
        loo_subsample(centered_eight, observations=0, var_name="obs")
    with pytest.raises(ValueError, match="Number of observations must be between 1 and"):
        loo_subsample(centered_eight, observations=n_total + 1, var_name="obs")
    with pytest.raises(TypeError, match="observations must be an integer"):
        loo_subsample(centered_eight, observations=4.2, var_name="obs")


def test_update_loo_subsample_errors(centered_eight):
    initial_observations = 3
    initial_loo = loo_subsample(centered_eight, observations=initial_observations, var_name="obs")

    n_total = centered_eight.observed_data.obs.size
    with pytest.raises(ValueError, match="Cannot add .* observations when only .* are available"):
        update_subsample(initial_loo, centered_eight, observations=n_total)

    existing_indices = np.where(~np.isnan(initial_loo.elpd_i.values.flatten()))[0]
    with pytest.raises(ValueError, match="New indices .* overlap with existing indices"):
        update_subsample(initial_loo, centered_eight, observations=existing_indices[:5])


def test_update_subsample_preserves_thin_factor(centered_eight):
    initial_observations = 2
    thin_factor = 2
    initial_loo = loo_subsample(
        centered_eight, observations=initial_observations, var_name="obs", thin=thin_factor
    )
    assert initial_loo.thin_factor == thin_factor

    updated_loo = update_subsample(initial_loo, centered_eight, observations=2)

    assert updated_loo.thin_factor == thin_factor


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

    temp_ds = xr.Dataset(param_list_for_ds)
    upars_da_generated = temp_ds.to_array(dim="unconstrained_parameter", name="upars_values")
    upars_da_final = upars_da_generated.transpose("chain", "draw", "unconstrained_parameter")

    def log_prob_upars_fn(upars_arg):
        return upars_arg.sum("unconstrained_parameter") * 0.01

    def log_lik_i_upars_fn(upars_arg, i):
        obs_dims = [dim for dim in log_likelihood_data.dims if dim not in ["chain", "draw"]]
        original_log_lik_for_i = log_likelihood_data.stack(i=obs_dims).isel(i=i)
        perturbation = upars_arg.sum("unconstrained_parameter") * 0.0001
        return original_log_lik_for_i + perturbation

    return upars_da_final, log_prob_upars_fn, log_lik_i_upars_fn


def test_loo_moment_match(datatree, loo_orig, moment_match_data):
    upars_da, log_prob_fn, log_lik_i_fn = moment_match_data
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


def test_loo_moment_match_no_problematic_k(datatree, loo_orig, moment_match_data):
    upars_da, log_prob_fn, log_lik_i_fn = moment_match_data
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
    upars_da, log_prob_fn, log_lik_i_fn = moment_match_data

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


def test_loo_weights_reuse_and_elpddata_input(centered_eight):
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
    assert hasattr(metrics, "se")

    loo_exp_elpddata, khat_elpddata = loo_expectations(
        centered_eight, kind="mean", log_weights=loo_result
    )
    assert loo_exp_elpddata is not None
    assert khat_elpddata is not None
    assert_array_equal(loo_exp.values, loo_exp_elpddata.values)

    loo_pit_elpddata = loo_pit(centered_eight, log_weights=loo_result)
    assert loo_pit_elpddata is not None
    assert_array_equal(loo_pit_vals["obs"].values, loo_pit_elpddata["obs"].values)

    metrics_elpddata = loo_metrics(centered_eight, kind="rmse", log_weights=loo_result)
    assert metrics_elpddata is not None
    assert metrics_elpddata.mean == metrics.mean
    assert metrics_elpddata.se == metrics.se

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


def test_log_weights_storage_and_reuse(centered_eight):
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

    loo_with_weights = loo(centered_eight, pointwise=True, log_weights=loo_pw_true.log_weights)
    assert loo_with_weights.log_weights is not None
    assert_array_equal(loo_with_weights.log_weights.values, loo_pw_true.log_weights.values)

    loo_exp, khat = loo_expectations(centered_eight, kind="mean", log_weights=loo_pw_true)
    assert loo_exp is not None
    assert khat is not None
    assert loo_exp.shape == (8,)

    loo_pit_vals = loo_pit(centered_eight, log_weights=loo_pw_true)
    assert loo_pit_vals is not None
    assert "obs" in loo_pit_vals
    assert loo_pit_vals["obs"].shape == (8,)

    metrics = loo_metrics(centered_eight, kind="rmse", log_weights=loo_pw_true)
    assert metrics is not None
    assert hasattr(metrics, "mean")
    assert hasattr(metrics, "se")

    loo_sub_elpddata = loo_subsample(
        centered_eight,
        observations=4,
        pointwise=True,
        var_name="obs",
        log_weights=loo_pw_true,
        seed=42,
    )
    assert loo_sub_elpddata is not None
    assert loo_sub_elpddata.log_weights is not None
    assert loo_sub_elpddata.log_weights["obs"].shape == (4, 4, 500)
