"""Test probability integral transform for PSIS-LOO-CV."""

# pylint: disable=redefined-outer-name, unused-argument
import pytest

from ..helpers import importorskip

np = importorskip("numpy")
xr = importorskip("xarray")
azb = importorskip("arviz_base")

from numpy.testing import assert_almost_equal

from arviz_stats import loo_pit
from arviz_stats.loo.helper_loo import _get_r_eff
from arviz_stats.utils import get_log_likelihood_dataset


def test_loo_pit_invalid_var_name(centered_eight):
    with pytest.raises(KeyError):
        loo_pit(centered_eight, var_names="nonexistent")


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"var_names": ["obs"]},
        {"var_names": "obs"},
    ],
)
def test_loo_pit(centered_eight, args):
    var_names = args.get("var_names", None)

    loo_pit_values = loo_pit(
        centered_eight,
        var_names=var_names,
    )
    assert np.all(loo_pit_values >= 0)
    assert np.all(loo_pit_values <= 1)


def test_loo_pit_discrete(centered_eight):
    centered_eight.observed_data["obs"] = centered_eight.observed_data["obs"].astype(int)

    loo_pit_values = loo_pit(centered_eight)
    assert np.all(loo_pit_values >= 0)
    assert np.all(loo_pit_values <= 1)


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


def test_loo_pit_precomputed_weights(centered_eight):
    result_auto = loo_pit(centered_eight)

    var_names = ["obs"]
    log_likelihood = get_log_likelihood_dataset(centered_eight, var_names=var_names)
    n_samples = log_likelihood.chain.size * log_likelihood.draw.size
    r_eff = _get_r_eff(centered_eight, n_samples)

    log_weights_computed, pareto_k_computed = log_likelihood["obs"].azstats.psislw(
        dim=["chain", "draw"],
        r_eff=r_eff,
    )

    log_weights_ds = xr.Dataset({"obs": log_weights_computed})
    pareto_k_ds = xr.Dataset({"obs": pareto_k_computed})

    result_precomputed = loo_pit(
        centered_eight,
        var_names="obs",
        log_weights=log_weights_ds,
        pareto_k=pareto_k_ds,
    )

    assert_almost_equal(result_precomputed["obs"].values, result_auto["obs"].values, decimal=10)


def test_loo_pit_random_state_reproducibility(rng):
    discrete_data = azb.from_dict(
        {
            "posterior": {"mu": rng.normal(size=(2, 50))},
            "posterior_predictive": {"y": rng.integers(0, 10, size=(2, 50, 8)).astype(float)},
            "log_likelihood": {"y": rng.normal(size=(2, 50, 8))},
            "observed_data": {"y": np.array([3.0, 5.0, 7.0, 2.0, 8.0, 4.0, 6.0, 1.0])},
        }
    )

    result_1 = loo_pit(discrete_data, random_state=42)
    result_2 = loo_pit(discrete_data, random_state=42)
    result_3 = loo_pit(discrete_data, random_state=123)

    assert_almost_equal(result_1["y"].values, result_2["y"].values, decimal=10)
    assert not np.allclose(result_1["y"].values, result_3["y"].values)


def test_loo_pit_random_state_with_discrete(rng):
    discrete_data = azb.from_dict(
        {
            "posterior": {"mu": rng.normal(size=(2, 50))},
            "posterior_predictive": {"y": rng.integers(0, 10, size=(2, 50, 8)).astype(float)},
            "log_likelihood": {"y": rng.normal(size=(2, 50, 8))},
            "observed_data": {"y": np.array([3.0, 5.0, 7.0, 2.0, 8.0, 4.0, 6.0, 1.0])},
        }
    )

    result_1 = loo_pit(discrete_data, random_state=42)
    result_2 = loo_pit(discrete_data, random_state=42)

    assert_almost_equal(result_1["y"].values, result_2["y"].values, decimal=10)
