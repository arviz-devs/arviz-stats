# pylint: disable=redefined-outer-name, unused-import, unused-argument
# ruff: noqa: F811, F401
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from ..helpers import importorskip

azb = importorskip("arviz_base")

from arviz_stats import loo, loo_pit
from arviz_stats.utils import ELPDData, get_log_likelihood_dataset


def test_loo_pit_invalid_var_name(centered_eight):
    with pytest.raises(KeyError):
        loo_pit(centered_eight, var_names="nonexistent")


def test_loo_pit_elpddata_without_log_weights(centered_eight):
    loo_result_no_weights = ELPDData(
        elpd=-30.0,
        se=3.0,
        p=2.0,
        good_k=0.7,
        n_samples=100,
        n_data_points=8,
        warning=False,
        kind="loo",
        scale="log",
        log_weights=None,
    )

    with pytest.raises(ValueError, match="ELPDData object does not contain log_weights"):
        loo_pit(centered_eight, log_weights=loo_result_no_weights)


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


def test_log_weights_input_formats(centered_eight):
    loo_result = loo(centered_eight, pointwise=True)
    log_weights_da = loo_result.log_weights

    loo_pit_da = loo_pit(centered_eight, log_weights=log_weights_da)
    loo_pit_elpddata = loo_pit(centered_eight, log_weights=loo_result)
    assert_array_equal(loo_pit_da["obs"].values, loo_pit_elpddata["obs"].values)


def test_loo_pit_all_var_names(centered_eight):
    result = loo_pit(centered_eight)

    assert "obs" in result
    assert np.all(result["obs"].values >= 0)
    assert np.all(result["obs"].values <= 1)


def test_loo_pit_multiple_vars():
    rng = np.random.default_rng(42)

    multi_var_data = azb.from_dict(
        {
            "posterior": {"mu": rng.normal(size=(2, 50))},
            "posterior_predictive": {
                "y1": rng.normal(size=(2, 50, 5)),
                "y2": rng.normal(size=(2, 50, 3)),
            },
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

    result = loo_pit(multi_var_data)

    assert "y1" in result
    assert "y2" in result
    assert result["y1"].shape == (5,)
    assert result["y2"].shape == (3,)
    assert np.all(result["y1"].values >= 0)
    assert np.all(result["y1"].values <= 1)
    assert np.all(result["y2"].values >= 0)
    assert np.all(result["y2"].values <= 1)


def test_loo_pit_single_var_list(centered_eight):
    result = loo_pit(centered_eight, var_names=["obs"])

    assert "obs" in result
    assert result["obs"].shape == (8,)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_loo_pit_multidimensional():
    rng = np.random.default_rng(42)

    multi_dim_data = azb.from_dict(
        {
            "posterior": {"mu": rng.normal(size=(2, 50))},
            "posterior_predictive": {"y": rng.normal(size=(2, 50, 3, 4))},
            "log_likelihood": {"y": rng.normal(size=(2, 50, 3, 4))},
            "observed_data": {"y": rng.normal(size=(3, 4))},
        }
    )

    result = loo_pit(multi_dim_data)

    assert "y" in result
    assert result["y"].shape == (3, 4)
    assert np.all(result["y"].values >= 0)
    assert np.all(result["y"].values <= 1)


def test_loo_pit_with_precomputed_log_weights(centered_eight):
    loo_result = loo(centered_eight, pointwise=True)

    result_with_weights = loo_pit(centered_eight, log_weights=loo_result)
    result_without_weights = loo_pit(centered_eight)

    assert result_with_weights["obs"].shape == result_without_weights["obs"].shape
    assert np.all(result_with_weights["obs"].values >= 0)
    assert np.all(result_with_weights["obs"].values <= 1)
