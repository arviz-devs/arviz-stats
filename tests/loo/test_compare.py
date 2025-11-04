"""Test compare function."""

# pylint: disable=redefined-outer-name, unused-argument
import copy

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal

from ..helpers import importorskip

azb = importorskip("arviz_base")
sp = importorskip("scipy")

from arviz_stats import compare, loo, loo_subsample, update_subsample
from arviz_stats.loo import _calculate_ics


def log_lik_fn_subsample(obs_da, datatree):
    theta = datatree.posterior["theta"]
    sigma = datatree.constant_data["sigma"]
    return sp.stats.norm.logpdf(obs_da, loc=theta, scale=sigma)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("method", ["stacking", "BB-pseudo-BMA", "pseudo-BMA"])
def test_compare_same(centered_eight, method):
    data_dict = {"first": centered_eight, "second": centered_eight}

    weight = compare(data_dict, method=method)["weight"].to_numpy()
    assert_allclose(weight[0], weight[1])
    assert_allclose(np.sum(weight), 1.0)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_compare_unknown_method(centered_eight, non_centered_eight):
    model_dict = {"centered": centered_eight, "non_centered": non_centered_eight}
    with pytest.raises(
        ValueError,
        match="Invalid method 'Unknown'. Available methods: stacking, bb-pseudo-bma, pseudo-bma",
    ):
        compare(model_dict, method="Unknown")


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("method", ["stacking", "BB-pseudo-BMA", "pseudo-BMA"])
def test_compare_different(centered_eight, non_centered_eight, method):
    model_dict = {"centered": centered_eight, "non_centered": non_centered_eight}
    weight = compare(model_dict, method=method)["weight"]
    assert weight["non_centered"] > weight["centered"]
    assert_allclose(np.sum(weight), 1.0)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_compare_multiple_different_sizes(centered_eight):
    centered_eight_subset1 = centered_eight.sel(school=["Choate"])
    centered_eight_subset2 = centered_eight.sel(school=["Choate", "Deerfield"])
    model_dict = {
        "model_a": centered_eight,
        "model_b": centered_eight_subset1,
        "model_c": centered_eight,
        "model_d": centered_eight_subset2,
    }
    with pytest.raises(ValueError) as exc_info:
        compare(model_dict)
    error_msg = str(exc_info.value)
    expected_msg = (
        "All models must have the same number of observations, but models have inconsistent "
        "observation counts: 'model_b' (1), 'model_d' (2), 'model_a' (8), 'model_c' (8)"
    )
    assert error_msg == expected_msg


@pytest.mark.filterwarnings("ignore::UserWarning")
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


@pytest.mark.filterwarnings("ignore::UserWarning")
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


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_calculate_ics_pointwise_error(centered_eight, non_centered_eight):
    in_dict = {
        "centered": loo(centered_eight, pointwise=True),
        "non_centered": loo(non_centered_eight, pointwise=False),
    }
    with pytest.raises(ValueError, match="Model .* is missing pointwise ELPD values"):
        _calculate_ics(in_dict)


def test_compare_mixed_elpd_methods(centered_eight, non_centered_eight):
    loo_result = loo(centered_eight, pointwise=True)
    kfold_result = loo(non_centered_eight, pointwise=True)
    kfold_result = copy.deepcopy(kfold_result)
    kfold_result.kind = "loo_kfold"

    compare_dict = {
        "loo_model": loo_result,
        "kfold_model": kfold_result,
    }

    with pytest.warns(UserWarning, match="Comparing LOO-CV to K-fold-CV"):
        result = compare(compare_dict)

    assert len(result) == 2
    assert "loo_model" in result.index
    assert "kfold_model" in result.index
    assert_allclose(result["weight"].sum(), 1.0)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_compare_unsupported_mixed_methods(centered_eight):
    loo_result = loo(centered_eight, pointwise=True)

    waic_result = copy.deepcopy(loo_result)
    waic_result.kind = "waic"

    compare_dict = {
        "loo_model": loo_result,
        "waic_model": waic_result,
    }

    with pytest.raises(
        ValueError, match="Cannot compare models with incompatible cross-validation methods.*waic"
    ):
        compare(compare_dict)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_compare_subsampled(centered_eight_with_sigma, centered_eight):
    loo_sub1 = loo_subsample(
        centered_eight_with_sigma,
        observations=np.array([0, 1, 2, 3]),
        var_name="obs",
        method="plpd",
        log_lik_fn=log_lik_fn_subsample,
        param_names=["theta"],
        pointwise=True,
    )

    loo_sub2 = loo_subsample(
        centered_eight_with_sigma,
        observations=np.array([0, 1, 2, 3]),
        var_name="obs",
        method="plpd",
        log_lik_fn=log_lik_fn_subsample,
        param_names=["theta"],
        pointwise=True,
    )

    loo_sub3 = loo_subsample(
        centered_eight_with_sigma,
        observations=np.array([4, 5, 6, 7]),
        var_name="obs",
        method="plpd",
        log_lik_fn=log_lik_fn_subsample,
        param_names=["theta"],
        pointwise=True,
    )

    loo_updated = update_subsample(
        loo_sub1,
        centered_eight_with_sigma,
        observations=2,
        var_name="obs",
        method="plpd",
        log_lik_fn=log_lik_fn_subsample,
        param_names=["theta"],
        seed=123,
    )

    loo_full = loo(centered_eight, pointwise=True)

    comparison_subsampled = compare({"model1": loo_sub1, "model2": loo_sub2})
    assert "subsampling_dse" in comparison_subsampled.columns
    assert np.isfinite(comparison_subsampled["subsampling_dse"].values).all()
    assert np.isfinite(comparison_subsampled["dse"].values).all()
    assert_almost_equal(comparison_subsampled["elpd_diff"].iloc[0], 0.0, decimal=4)

    comparison_updated = compare({"model1": loo_sub1, "model2": loo_updated})
    assert "subsampling_dse" in comparison_updated.columns
    assert np.isfinite(comparison_updated["subsampling_dse"].values).all()

    with pytest.warns(UserWarning, match="Different subsamples used in 'model_a' and 'model_b'"):
        comparison_diff_subsample = compare({"model_a": loo_sub1, "model_b": loo_sub3})
    assert "subsampling_dse" in comparison_diff_subsample.columns
    assert np.isfinite(comparison_diff_subsample["subsampling_dse"].values).all()

    comparison_regular = compare({"model1": loo_full, "model2": loo_full})
    assert "subsampling_dse" not in comparison_regular.columns
    assert np.isfinite(comparison_regular["dse"].values).all()
    assert_almost_equal(comparison_regular["elpd_diff"].iloc[0], 0.0, decimal=4)


@pytest.mark.parametrize("method", ["BB-pseudo-BMA", "pseudo-BMA"])
def test_compare_single_model(centered_eight, method):
    single_dict = {"model": centered_eight}
    result = compare(single_dict, method=method)

    assert len(result) == 1
    assert result.index[0] == "model"
    assert result["rank"].iloc[0] == 0
    assert result["elpd_diff"].iloc[0] == 0.0
    assert result["dse"].iloc[0] == 0.0
    assert result["weight"].iloc[0] == 1.0
    assert_allclose(result["weight"].sum(), 1.0)


@pytest.mark.parametrize("method", ["stacking", "BB-pseudo-BMA", "pseudo-BMA"])
def test_compare_many_models(centered_eight, method):
    models = {f"model_{i}": centered_eight for i in range(7)}
    result = compare(models, method=method)

    assert len(result) == 7
    assert result["rank"].min() == 0
    assert result["rank"].max() == 6
    assert_allclose(result["weight"].sum(), 1.0)
    assert result.iloc[0]["elpd_diff"] == 0.0
    assert result.iloc[0]["dse"] == 0.0

    for i in range(len(result)):
        assert result.iloc[i]["rank"] == i


@pytest.mark.parametrize("method", ["stacking", "BB-pseudo-BMA", "pseudo-BMA"])
def test_compare_columns(centered_eight, non_centered_eight, method):
    model_dict = {"centered": centered_eight, "non_centered": non_centered_eight}
    result = compare(model_dict, method=method)

    expected_cols = ["rank", "elpd", "p", "elpd_diff", "weight", "se", "dse", "warning"]
    assert all(col in result.columns for col in expected_cols)
    assert result["rank"].dtype == int
    assert result["elpd"].dtype == float
    assert result["p"].dtype == float
    assert result["weight"].dtype == float
    assert result["warning"].dtype == bool


@pytest.mark.parametrize("method", ["stacking", "BB-pseudo-BMA", "pseudo-BMA"])
def test_compare_best_model_properties(centered_eight, non_centered_eight, method):
    model_dict = {"centered": centered_eight, "non_centered": non_centered_eight}
    result = compare(model_dict, method=method)

    best_model = result.iloc[0]
    assert best_model["rank"] == 0
    assert best_model["elpd_diff"] == 0.0
    assert best_model["dse"] == 0.0
    assert best_model["elpd"] >= result.iloc[1]["elpd"]


@pytest.mark.parametrize("method", ["stacking", "BB-pseudo-BMA", "pseudo-BMA"])
def test_compare_weights_properties(centered_eight, non_centered_eight, method):
    model_dict = {"centered": centered_eight, "non_centered": non_centered_eight}
    result = compare(model_dict, method=method)

    assert (result["weight"] >= 0).all()
    assert (result["weight"] <= 1).all()
    assert_allclose(result["weight"].sum(), 1.0)


def test_compare_sorting(centered_eight, non_centered_eight):
    model_dict = {"centered": centered_eight, "non_centered": non_centered_eight}
    result = compare(model_dict)

    elpd_values = result["elpd"].values
    assert all(elpd_values[i] >= elpd_values[i + 1] for i in range(len(elpd_values) - 1))


@pytest.mark.parametrize("n_models", [2, 3, 5, 8])
@pytest.mark.parametrize("method", ["stacking", "BB-pseudo-BMA", "pseudo-BMA"])
def test_compare_weight_sum(centered_eight, n_models, method):
    models = {f"model_{i}": centered_eight for i in range(n_models)}
    result = compare(models, method=method)
    assert_allclose(result["weight"].sum(), 1.0)


def test_compare_mixed_idata_elpd(centered_eight, non_centered_eight):
    loo_result = loo(centered_eight, pointwise=True)

    mixed_dict = {
        "idata_model": non_centered_eight,
        "elpd_model": loo_result,
    }
    result = compare(mixed_dict)

    assert len(result) == 2
    assert "idata_model" in result.index
    assert "elpd_model" in result.index
    assert_allclose(result["weight"].sum(), 1.0)


def test_compare_elpd_diff_consistency(centered_eight, non_centered_eight):
    model_dict = {"centered": centered_eight, "non_centered": non_centered_eight}
    result = compare(model_dict)

    best_elpd = result.iloc[0]["elpd"]
    second_elpd = result.iloc[1]["elpd"]
    reported_diff = result.iloc[1]["elpd_diff"]

    expected_diff = best_elpd - second_elpd
    assert_almost_equal(reported_diff, expected_diff, decimal=5)


def test_compare_all_methods_produce_same_ranking(centered_eight, non_centered_eight):
    model_dict = {"centered": centered_eight, "non_centered": non_centered_eight}

    rankings = {}
    for method in ["stacking", "BB-pseudo-BMA", "pseudo-BMA"]:
        result = compare(model_dict, method=method)
        rankings[method] = list(result.index)

    assert rankings["stacking"] == rankings["BB-pseudo-BMA"]
    assert rankings["stacking"] == rankings["pseudo-BMA"]


def test_compare_finite_values(centered_eight, non_centered_eight):
    model_dict = {"centered": centered_eight, "non_centered": non_centered_eight}
    result = compare(model_dict)

    numeric_cols = ["elpd", "p", "elpd_diff", "weight", "se", "dse"]
    for col in numeric_cols:
        assert np.all(np.isfinite(result[col]))


def test_compare_se_values(centered_eight, non_centered_eight):
    model_dict = {"centered": centered_eight, "non_centered": non_centered_eight}
    result = compare(model_dict)

    for se_val in result["se"]:
        assert se_val > 0

    for dse_val in result["dse"]:
        assert dse_val >= 0


def test_compare_rank_ordering(centered_eight):
    models = {f"model_{i}": centered_eight for i in range(5)}
    result = compare(models)

    ranks = result["rank"].values
    expected_ranks = np.arange(len(models))
    assert_array_equal(ranks, expected_ranks)


def test_compare_identical_models_equal_weights(centered_eight):
    model_dict = {"model_a": centered_eight, "model_b": centered_eight, "model_c": centered_eight}
    result = compare(model_dict, method="stacking")

    weights = result["weight"].values
    assert_allclose(weights[0], weights[1], rtol=1e-5)
    assert_allclose(weights[1], weights[2], rtol=1e-5)


def test_compare_elpd_values_ordered(centered_eight):
    models = {f"model_{i}": centered_eight for i in range(4)}
    result = compare(models)

    for i in range(len(result) - 1):
        assert result.iloc[i]["elpd"] >= result.iloc[i + 1]["elpd"]


def test_compare_empty_dict_error():
    with pytest.raises((ValueError, KeyError, IndexError)):
        compare({})


def test_compare_identical_elpd_values(centered_eight):
    models = {f"model_{i}": centered_eight for i in range(3)}
    result = compare(models)

    elpd_values = result["elpd"].unique()
    assert len(elpd_values) == 1


def test_compare_no_negative_weights(centered_eight, non_centered_eight):
    model_dict = {"centered": centered_eight, "non_centered": non_centered_eight}

    for method in ["stacking", "BB-pseudo-BMA", "pseudo-BMA"]:
        result = compare(model_dict, method=method)
        assert (result["weight"] >= 0).all()


def test_compare_elpd_diff_relative_to_best(centered_eight, non_centered_eight):
    model_dict = {"centered": centered_eight, "non_centered": non_centered_eight}
    result = compare(model_dict)

    best_elpd = result.iloc[0]["elpd"]

    for i in range(len(result)):
        expected_diff = best_elpd - result.iloc[i]["elpd"]
        assert_almost_equal(result.iloc[i]["elpd_diff"], expected_diff, decimal=10)


def test_compare_order_stat_check(centered_eight, rng):
    models = {}
    base_loo = loo(centered_eight, pointwise=True)

    for i in range(12):
        loo_result = copy.deepcopy(base_loo)
        shift = rng.normal(0, 0.1, size=loo_result.elpd_i.shape)
        loo_result.elpd_i = loo_result.elpd_i + shift
        loo_result.elpd = np.sum(loo_result.elpd_i)
        models[f"model_{i}"] = loo_result

    with pytest.warns(
        UserWarning,
        match="Difference in performance potentially due to chance.*10.1007/s11222-024-10442-4",
    ):
        result = compare(models)
    assert len(result) == 12
    assert_allclose(result["weight"].sum(), 1.0)


def test_compare_order_stat_check_identical_models(centered_eight):
    models = {f"model_{i}": centered_eight for i in range(12)}
    with pytest.warns(UserWarning, match="All models have nearly identical performance"):
        result = compare(models)
    assert len(result) == 12
    assert_allclose(result["elpd"].values, result["elpd"].values[0])
    assert_allclose(result["weight"].sum(), 1.0)


def test_compare_order_stat_check_few_models(centered_eight):
    models = {f"model_{i}": centered_eight for i in range(11)}
    result = compare(models)
    assert len(result) == 11


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_compare_order_stat_check_subsampling(centered_eight_with_sigma, rng):
    base_loo_sub = loo_subsample(
        centered_eight_with_sigma,
        observations=np.array([0, 1, 2, 3]),
        var_name="obs",
        method="plpd",
        log_lik_fn=log_lik_fn_subsample,
        param_names=["theta"],
        pointwise=True,
    )

    models = {}
    for i in range(12):
        loo_sub = copy.deepcopy(base_loo_sub)
        shift = rng.normal(0, 0.1, size=loo_sub.elpd_i.shape)
        loo_sub.elpd_i = loo_sub.elpd_i + shift
        loo_sub.elpd = np.sum(loo_sub.elpd_i)
        models[f"model_{i}"] = loo_sub

    result = compare(models)
    assert len(result) == 12
    assert "subsampling_dse" in result.columns
