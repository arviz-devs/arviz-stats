"""Test survival analysis."""

# pylint: disable=redefined-outer-name
import numpy as np
import pytest

from .helpers import importorskip

azb = importorskip("arviz_base")

from arviz_stats.survival import generate_survival_curves, kaplan_meier


@pytest.fixture
def survival_datatree():
    rng = np.random.default_rng(42)
    times = rng.exponential(10, 30)
    status = np.ones_like(times)
    return azb.from_dict({"observed_data": {"times": times}, "constant_data": {"times": status}})


@pytest.fixture
def survival_datatree_censored():
    rng = np.random.default_rng(42)
    times = rng.exponential(10, 30)
    status = rng.binomial(1, 0.7, 30)
    return azb.from_dict({"observed_data": {"times": times}, "constant_data": {"times": status}})


@pytest.fixture
def survival_datatree_predictive():
    rng = np.random.default_rng(42)
    n_chains = 4
    n_draws = 50
    n_obs = 30
    observed_times = rng.exponential(10, n_obs)
    predictive_times = rng.exponential(10, (n_chains, n_draws, n_obs))
    return azb.from_dict(
        {
            "posterior_predictive": {"times": predictive_times},
            "observed_data": {"times": observed_times},
        }
    )


def test_kaplan_meier_basic(survival_datatree):
    result = kaplan_meier(survival_datatree, var_names=["times"], group="observed_data")
    assert "times" in result.data_vars
    assert "plot_axis" in result.dims
    assert "km_points" in result.dims
    assert list(result.plot_axis.values) == ["x", "y"]


def test_kaplan_meier_survival_decreases(survival_datatree):
    result = kaplan_meier(survival_datatree, var_names=["times"], group="observed_data")
    survival_probs = result["times"].values[1]
    assert np.all(np.diff(survival_probs) <= 0)


def test_kaplan_meier_starts_near_one(survival_datatree):
    result = kaplan_meier(survival_datatree, var_names=["times"], group="observed_data")
    survival_probs = result["times"].values[1]
    assert survival_probs[0] <= 1.0
    assert survival_probs[0] > 0.9


def test_kaplan_meier_with_censoring(survival_datatree_censored):
    result = kaplan_meier(survival_datatree_censored, var_names=["times"], group="observed_data")
    assert "times" in result.data_vars
    survival_probs = result["times"].values[1]
    assert np.all(np.diff(survival_probs) <= 0)


def test_kaplan_meier_no_constant_data():
    rng = np.random.default_rng(42)
    times = rng.exponential(10, 30)
    dt = azb.from_dict({"observed_data": {"times": times}})
    with pytest.warns(UserWarning, match="No 'constant_data' group found"):
        result = kaplan_meier(dt, var_names=["times"], group="observed_data")
        assert "times" in result.data_vars


def test_kaplan_meier_missing_status_variable():
    rng = np.random.default_rng(42)
    times = rng.exponential(10, 30)
    other_var = np.ones(30)
    dt = azb.from_dict({"observed_data": {"times": times}, "constant_data": {"other": other_var}})
    with pytest.warns(UserWarning, match="No status variable found"):
        result = kaplan_meier(dt, var_names=["times"], group="observed_data")
        assert "times" in result.data_vars


def test_kaplan_meier_multiple_vars():
    rng = np.random.default_rng(42)
    times1 = rng.exponential(10, 30)
    times2 = rng.exponential(15, 30)
    status1 = np.ones_like(times1)
    status2 = np.ones_like(times2)
    dt = azb.from_dict(
        {
            "observed_data": {"times1": times1, "times2": times2},
            "constant_data": {"times1": status1, "times2": status2},
        }
    )
    result = kaplan_meier(dt, var_names=["times1", "times2"], group="observed_data")
    assert "times1" in result.data_vars
    assert "times2" in result.data_vars


def test_kaplan_meier_sorted_times(survival_datatree):
    result = kaplan_meier(survival_datatree, var_names=["times"], group="observed_data")
    times = result["times"].values[0]
    assert np.all(np.diff(times) >= 0)


def test_generate_survival_curves_basic(survival_datatree_predictive):
    result = generate_survival_curves(
        survival_datatree_predictive,
        var_names=["times"],
        group="posterior_predictive",
        num_samples=10,
    )
    assert "times" in result.data_vars
    assert "plot_axis" in result.dims
    assert "sample" in result.dims
    assert list(result.plot_axis.values) == ["x", "y"]


def test_generate_survival_curves_decreasing(survival_datatree_predictive):
    result = generate_survival_curves(
        survival_datatree_predictive,
        var_names=["times"],
        group="posterior_predictive",
        num_samples=5,
    )
    times_array = result["times"].values[0]
    probs_array = result["times"].values[1]
    for i in range(times_array.shape[0]):
        valid_idx = ~np.isnan(times_array[i])
        if valid_idx.sum() > 1:
            assert np.all(np.diff(probs_array[i][valid_idx]) <= 0)


def test_generate_survival_curves_with_extrapolation(survival_datatree_predictive):
    result = generate_survival_curves(
        survival_datatree_predictive,
        var_names=["times"],
        group="posterior_predictive",
        num_samples=10,
        extrapolation_factor=1.5,
    )
    max_observed = survival_datatree_predictive.observed_data["times"].max().values
    times_array = result["times"].values[0]
    valid_times = times_array[~np.isnan(times_array)]
    assert np.all(valid_times <= max_observed * 1.5)


def test_generate_survival_curves_no_extrapolation(survival_datatree_predictive):
    result = generate_survival_curves(
        survival_datatree_predictive,
        var_names=["times"],
        group="posterior_predictive",
        num_samples=10,
        extrapolation_factor=None,
    )
    assert "times" in result.data_vars


def test_generate_survival_curves_output_shape(survival_datatree_predictive):
    num_samples = 15
    result = generate_survival_curves(
        survival_datatree_predictive,
        var_names=["times"],
        group="posterior_predictive",
        num_samples=num_samples,
    )
    times_data = result["times"].values
    assert times_data.shape[0] == 2
    assert times_data.shape[1] <= num_samples


def test_generate_survival_curves_unique_sorted(survival_datatree_predictive):
    result = generate_survival_curves(
        survival_datatree_predictive,
        var_names=["times"],
        group="posterior_predictive",
        num_samples=5,
    )
    times_array = result["times"].values[0]
    for i in range(times_array.shape[0]):
        valid_times = times_array[i][~np.isnan(times_array[i])]
        if len(valid_times) > 1:
            assert np.all(np.diff(valid_times) >= 0)


def test_generate_survival_curves_probs_in_range(survival_datatree_predictive):
    result = generate_survival_curves(
        survival_datatree_predictive,
        var_names=["times"],
        group="posterior_predictive",
        num_samples=10,
    )
    probs_array = result["times"].values[1]
    valid_probs = probs_array[~np.isnan(probs_array)]
    assert np.all(valid_probs >= 0)
    assert np.all(valid_probs <= 1)


def test_kaplan_meier_all_censored():
    rng = np.random.default_rng(42)
    times = rng.exponential(10, 30)
    status = np.zeros_like(times)
    dt = azb.from_dict({"observed_data": {"times": times}, "constant_data": {"times": status}})
    result = kaplan_meier(dt, var_names=["times"], group="observed_data")
    survival_probs = result["times"].values[1]
    assert np.all(survival_probs == 1.0)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_kaplan_meier_single_event():
    times = np.array([5.0])
    status = np.array([1.0])
    dt = azb.from_dict({"observed_data": {"times": times}, "constant_data": {"times": status}})
    result = kaplan_meier(dt, var_names=["times"], group="observed_data")
    assert "times" in result.data_vars


def test_generate_survival_curves_small_sample():
    rng = np.random.default_rng(42)
    observed_times = rng.exponential(10, 10)
    predictive_times = rng.exponential(10, (2, 10, 10))
    dt = azb.from_dict(
        {
            "posterior_predictive": {"times": predictive_times},
            "observed_data": {"times": observed_times},
        }
    )
    result = generate_survival_curves(
        dt, var_names=["times"], group="posterior_predictive", num_samples=5
    )
    assert "times" in result.data_vars


def test_kaplan_meier_tied_events():
    times = np.array([1.0, 1.0, 2.0, 2.0, 3.0])
    status = np.ones_like(times)
    dt = azb.from_dict({"observed_data": {"times": times}, "constant_data": {"times": status}})
    result = kaplan_meier(dt, var_names=["times"], group="observed_data")
    survival_probs = result["times"].values[1]
    assert len(survival_probs) == 3
    assert np.all(np.diff(survival_probs) <= 0)


def test_kaplan_meier_string_var_name():
    rng = np.random.default_rng(42)
    times = rng.exponential(10, 20)
    status = np.ones_like(times)
    dt = azb.from_dict({"observed_data": {"times": times}, "constant_data": {"times": status}})
    result = kaplan_meier(dt, var_names="times", group="observed_data")
    assert "times" in result.data_vars


def test_kaplan_meier_early_censoring():
    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    status = np.array([0, 1, 1, 1, 1])
    dt = azb.from_dict({"observed_data": {"times": times}, "constant_data": {"times": status}})
    result = kaplan_meier(dt, var_names=["times"], group="observed_data")
    survival_probs = result["times"].values[1]
    assert survival_probs[0] == 1.0


def test_kaplan_meier_late_censoring():
    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    status = np.array([1, 1, 1, 1, 0])
    dt = azb.from_dict({"observed_data": {"times": times}, "constant_data": {"times": status}})
    result = kaplan_meier(dt, var_names=["times"], group="observed_data")
    survival_probs = result["times"].values[1]
    assert np.all(np.diff(survival_probs) <= 0)
    assert survival_probs[-1] > 0


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_kaplan_meier_identical_times():
    times = np.array([5.0, 5.0, 5.0, 5.0])
    status = np.ones_like(times)
    dt = azb.from_dict({"observed_data": {"times": times}, "constant_data": {"times": status}})
    result = kaplan_meier(dt, var_names=["times"], group="observed_data")
    survival_probs = result["times"].values[1]
    assert len(survival_probs) == 1


def test_generate_survival_curves_extreme_extrapolation_low():
    rng = np.random.default_rng(42)
    observed_times = rng.exponential(10, 20)
    predictive_times = rng.exponential(10, (2, 10, 20))
    dt = azb.from_dict(
        {
            "posterior_predictive": {"times": predictive_times},
            "observed_data": {"times": observed_times},
        }
    )
    result = generate_survival_curves(
        dt,
        var_names=["times"],
        group="posterior_predictive",
        num_samples=5,
        extrapolation_factor=0.1,
    )
    assert "times" in result.data_vars


def test_generate_survival_curves_extreme_extrapolation_high():
    rng = np.random.default_rng(42)
    observed_times = rng.exponential(10, 20)
    predictive_times = rng.exponential(10, (2, 10, 20))
    dt = azb.from_dict(
        {
            "posterior_predictive": {"times": predictive_times},
            "observed_data": {"times": observed_times},
        }
    )
    result = generate_survival_curves(
        dt,
        var_names=["times"],
        group="posterior_predictive",
        num_samples=5,
        extrapolation_factor=10.0,
    )
    assert "times" in result.data_vars
    times_array = result["times"].values[0]
    valid_times = times_array[~np.isnan(times_array)]
    assert len(valid_times) > 0


def test_generate_survival_curves_string_var_name():
    rng = np.random.default_rng(42)
    observed_times = rng.exponential(10, 20)
    predictive_times = rng.exponential(10, (2, 10, 20))
    dt = azb.from_dict(
        {
            "posterior_predictive": {"times": predictive_times},
            "observed_data": {"times": observed_times},
        }
    )
    result = generate_survival_curves(
        dt, var_names="times", group="posterior_predictive", num_samples=5
    )
    assert "times" in result.data_vars


def test_kaplan_meier_ends_near_zero():
    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    status = np.ones_like(times)
    dt = azb.from_dict({"observed_data": {"times": times}, "constant_data": {"times": status}})
    result = kaplan_meier(dt, var_names=["times"], group="observed_data")
    survival_probs = result["times"].values[1]
    assert survival_probs[-1] < 0.3


def test_generate_survival_curves_first_prob_less_than_one():
    rng = np.random.default_rng(42)
    observed_times = rng.exponential(10, 20)
    predictive_times = rng.exponential(10, (2, 10, 20))
    dt = azb.from_dict(
        {
            "posterior_predictive": {"times": predictive_times},
            "observed_data": {"times": observed_times},
        }
    )
    result = generate_survival_curves(
        dt, var_names=["times"], group="posterior_predictive", num_samples=5
    )
    probs_array = result["times"].values[1]
    for i in range(probs_array.shape[0]):
        valid_probs = probs_array[i][~np.isnan(probs_array[i])]
        if len(valid_probs) > 0:
            assert valid_probs[0] <= 1.0
