# pylint: disable=redefined-outer-name

import numpy as np
import pytest
from arviz_base import load_arviz_data
from numpy.testing import assert_allclose, assert_almost_equal
from xarray import DataArray

from arviz_stats import compare, loo, loo_pit
from arviz_stats.loo import _calculate_ics
from arviz_stats.utils import get_log_likelihood_dataset


@pytest.fixture(name="centered_eight", scope="session")
def fixture_centered_eight():
    return load_arviz_data("centered_eight")


@pytest.fixture(name="non_centered_eight", scope="session")
def fixture_non_centered_eight():
    return load_arviz_data("non_centered_eight")


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
    "args",
    [
        {},
        {"y_obs": "obs"},
        {"y_obs": "obs", "y_pred": "obs"},
        {"log_weights": "arr"},
    ],
)
def test_loo_pit(centered_eight, args):
    y_obs = args.get("y", None)
    y_pred = args.get("y_hat", None)
    log_weights = args.get("log_weights", None)
    if log_weights == "arr":
        log_weights = get_log_likelihood_dataset(centered_eight, var_names=y_obs)

    loo_pit_values = loo_pit(centered_eight, y_obs=y_obs, y_pred=y_pred, log_weights=log_weights)
    assert np.all(loo_pit_values >= 0)
    assert np.all(loo_pit_values <= 1)
