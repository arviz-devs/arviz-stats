"""Test helper statistics."""

# pylint: disable=redefined-outer-name
import numpy as np
import pytest

from .helpers import importorskip

azb = importorskip("arviz_base")

from arviz_stats.helper_stats import isotonic_fit, point_interval_unique, point_unique


@pytest.fixture
def binary_datatree():
    rng = np.random.default_rng(42)
    n_chains = 4
    n_draws = 100
    n_obs = 50

    observed = rng.binomial(1, 0.6, n_obs)
    posterior_pred = rng.binomial(1, 0.6, (n_chains, n_draws, n_obs))

    return azb.from_dict(
        {
            "posterior_predictive": {"y": posterior_pred},
            "observed_data": {"y": observed},
        }
    )


@pytest.fixture
def categorical_datatree():
    rng = np.random.default_rng(42)
    n_chains = 4
    n_draws = 100
    n_obs = 50

    observed = rng.integers(0, 3, n_obs)
    posterior_pred = rng.integers(0, 3, (n_chains, n_draws, n_obs))

    return azb.from_dict(
        {
            "posterior_predictive": {"y": posterior_pred},
            "observed_data": {"y": observed},
        }
    )


@pytest.fixture
def continuous_datatree():
    rng = np.random.default_rng(42)
    n_chains = 4
    n_draws = 100
    n_obs = 30

    observed = rng.normal(5, 2, n_obs)
    posterior_pred = rng.normal(5, 2, (n_chains, n_draws, n_obs))

    return azb.from_dict(
        {
            "posterior_predictive": {"y": posterior_pred},
            "observed_data": {"y": observed},
        }
    )


def test_isotonic_fit_binary(binary_datatree):
    result = isotonic_fit(
        binary_datatree,
        var_names=["y"],
        group="posterior_predictive",
        n_bootstrap=10,
        ci_prob=0.94,
        data_type="binary",
    )
    assert "y" in result.data_vars
    assert "plot_axis" in result.dims
    assert "x_values" in result.dims
    assert list(result.plot_axis.values) == ["x", "y", "y_bottom", "y_top"]
    assert result["y"].shape[0] == 4


def test_isotonic_fit_binary_string_var_name(binary_datatree):
    result = isotonic_fit(
        binary_datatree,
        var_names="y",
        group="posterior_predictive",
        n_bootstrap=10,
        ci_prob=0.94,
        data_type="binary",
    )
    assert "y" in result.data_vars


def test_isotonic_fit_categorical(categorical_datatree):
    result = isotonic_fit(
        categorical_datatree,
        var_names=["y"],
        group="posterior_predictive",
        n_bootstrap=10,
        ci_prob=0.94,
        data_type="categorical",
    )
    assert "y 0 vs others" in result.data_vars
    assert "y 1 vs others" in result.data_vars
    assert "y 2 vs others" in result.data_vars


def test_isotonic_fit_ordinal(categorical_datatree):
    result = isotonic_fit(
        categorical_datatree,
        var_names=["y"],
        group="posterior_predictive",
        n_bootstrap=10,
        ci_prob=0.94,
        data_type="ordinal",
    )
    assert "y cumul≤0" in result.data_vars
    assert "y cumul≤1" in result.data_vars


def test_isotonic_fit_invalid_data_type(binary_datatree):
    with pytest.raises(ValueError, match="Invalid data_type"):
        isotonic_fit(
            binary_datatree,
            var_names=["y"],
            group="posterior_predictive",
            n_bootstrap=10,
            ci_prob=0.94,
            data_type="invalid",
        )


def test_isotonic_fit_none_var_names(binary_datatree):
    result = isotonic_fit(
        binary_datatree,
        var_names=None,
        group="posterior_predictive",
        n_bootstrap=10,
        ci_prob=0.94,
        data_type="binary",
    )
    assert "y" in result.data_vars


def test_isotonic_fit_categorical_with_continuous_data(continuous_datatree):
    with pytest.warns(UserWarning, match="appears to be continuous"):
        with pytest.raises(IndexError):
            isotonic_fit(
                continuous_datatree,
                var_names=["y"],
                group="posterior_predictive",
                n_bootstrap=10,
                ci_prob=0.94,
                data_type="categorical",
            )


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_isotonic_fit_categorical_with_binary_data(binary_datatree):
    with pytest.warns(UserWarning, match="appears to be binary"):
        isotonic_fit(
            binary_datatree,
            var_names=["y"],
            group="posterior_predictive",
            n_bootstrap=10,
            ci_prob=0.94,
            data_type="categorical",
        )


def test_isotonic_fit_ci_bounds(binary_datatree):
    result = isotonic_fit(
        binary_datatree,
        var_names=["y"],
        group="posterior_predictive",
        n_bootstrap=10,
        ci_prob=0.94,
        data_type="binary",
    )
    y_values = result["y"].values
    assert np.all(y_values[2] <= y_values[1])
    assert np.all(y_values[1] <= y_values[3])


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_point_interval_unique(binary_datatree):
    result = point_interval_unique(
        binary_datatree, var_names=["y"], group="posterior_predictive", ci_prob=0.94
    )
    assert "y" in result.data_vars
    assert "plot_axis" in result.dims
    assert "x_values" in result.dims
    assert list(result.plot_axis.values) == ["x", "y", "y_bottom", "y_top"]
    assert result["y"].shape[0] == 4


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_point_interval_unique_none_var_names(binary_datatree):
    result = point_interval_unique(
        binary_datatree, var_names=None, group="posterior_predictive", ci_prob=0.94
    )
    assert "y" in result.data_vars


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_point_interval_unique_ci_bounds(binary_datatree):
    result = point_interval_unique(
        binary_datatree, var_names=["y"], group="posterior_predictive", ci_prob=0.94
    )
    y_values = result["y"].values
    assert np.all(y_values[2] <= y_values[1])


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_point_interval_unique_multiple_vars(categorical_datatree):
    result = point_interval_unique(
        categorical_datatree, var_names=["y"], group="posterior_predictive", ci_prob=0.94
    )
    assert "y" in result.data_vars
    unique_values = result["y"].values[0]
    assert len(unique_values) > 1


def test_point_unique(binary_datatree):
    result = point_unique(binary_datatree, var_names=["y"])
    assert "y" in result.data_vars
    assert "plot_axis" in result.dims
    assert "x_values" in result.dims
    assert list(result.plot_axis.values) == ["x", "y"]
    assert result["y"].shape[0] == 2


def test_point_unique_none_var_names(binary_datatree):
    result = point_unique(binary_datatree, var_names=None)
    assert "y" in result.data_vars


def test_point_unique_returns_counts(binary_datatree):
    result = point_unique(binary_datatree, var_names=["y"])
    y_values = result["y"].values
    unique_vals = y_values[0]
    counts = y_values[1]
    assert len(unique_vals) == len(counts)
    assert np.all(counts > 0)
    assert np.all(counts == counts.astype(int))


def test_point_unique_categorical(categorical_datatree):
    result = point_unique(categorical_datatree, var_names=["y"])
    y_values = result["y"].values
    unique_vals = y_values[0]
    counts = y_values[1]
    assert len(unique_vals) >= 2
    assert np.sum(counts) == categorical_datatree.observed_data["y"].size


def test_isotonic_fit_output_shape_consistency(binary_datatree):
    result = isotonic_fit(
        binary_datatree,
        var_names=["y"],
        group="posterior_predictive",
        n_bootstrap=5,
        ci_prob=0.89,
        data_type="binary",
    )
    y_data = result["y"].values
    assert y_data.shape[0] == 4
    for i in range(4):
        assert len(y_data[i]) == len(y_data[0])
