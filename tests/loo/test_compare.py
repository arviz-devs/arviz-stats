# pylint: disable=redefined-outer-name, unused-import, unused-argument
# ruff: noqa: F811, F401
import copy

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal

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
