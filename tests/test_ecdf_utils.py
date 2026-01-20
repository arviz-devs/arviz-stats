"""Test ECDF utilities."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from scipy.special._ufuncs import _hypergeom_cdf

from .helpers import importorskip

azb = importorskip("arviz_base")
from arviz_stats.ecdf_utils import (
    _hypergeometric_cdf_lookup,
    _scalar_lookup,
    compute_ecdf,
    ecdf_pit,
    get_pointwise_confidence_band,
    simulate_confidence_bands,
)


@pytest.mark.parametrize("x_vals", [0, 1, [2, 3]])
def test_hypergeom_cdf(x_vals):
    suc = 8000
    pop = 200
    draws = 1000
    expected_cdf = _hypergeom_cdf(x_vals, draws, suc, pop)
    calculated_cdf = _hypergeometric_cdf_lookup(x_vals, pop, draws, suc)
    assert_array_almost_equal(calculated_cdf, expected_cdf)


def test_hypergeom_cdf_array():
    suc = 5000
    pop = 150
    draws = 500
    x_vals = np.array([10, 20, 30, 40])
    expected_cdf = _hypergeom_cdf(x_vals, draws, suc, pop)
    calculated_cdf = _hypergeometric_cdf_lookup(x_vals, pop, draws, suc)
    assert_array_almost_equal(calculated_cdf, expected_cdf)


@pytest.mark.parametrize(
    "x_val, expected",
    [
        (-5, 0.0),
        (-1, 0.0),
        (1000, 1.0),
        (500, 1.0),
    ],
)
def test_scalar_lookup_boundary(x_val, expected):
    result = _scalar_lookup(x_val, 100, 50, 50)
    assert result == expected


def test_compute_ecdf_uniform():
    rng = np.random.default_rng(42)
    sample = rng.uniform(0, 1, 100)
    eval_points = np.array([0.25, 0.5, 0.75])
    ecdf_vals = compute_ecdf(sample, eval_points)
    assert len(ecdf_vals) == len(eval_points)
    assert np.all((ecdf_vals >= 0) & (ecdf_vals <= 1))
    assert np.all(np.diff(ecdf_vals) >= 0)


def test_compute_ecdf_sorted():
    sample = np.array([1, 2, 3, 4, 5])
    eval_points = np.array([2.5, 3.5])
    ecdf_vals = compute_ecdf(sample, eval_points)
    assert_array_almost_equal(ecdf_vals, [0.4, 0.6])


@pytest.mark.parametrize("prob", [0.9, 0.95, 0.99])
def test_get_pointwise_confidence_band(prob):
    ndraws = 100
    eval_points = np.linspace(0.1, 0.9, 9)
    lower, upper = get_pointwise_confidence_band(prob, ndraws, eval_points)
    assert len(lower) == len(eval_points)
    assert len(upper) == len(eval_points)
    assert np.all(lower <= upper)
    assert np.all((lower >= 0) & (lower <= 1))
    assert np.all((upper >= 0) & (upper <= 1))


def test_confidence_bands_single_chain():
    rng = np.random.default_rng(42)
    n_draws = 50
    n_chains = 1
    eval_points = np.linspace(0.1, 0.9, 5)
    prob = 0.9
    n_simulations = 100
    result = simulate_confidence_bands(n_draws, n_chains, eval_points, prob, n_simulations, rng)
    assert isinstance(result, float)
    assert 0 <= result <= 1


def test_confidence_bands_multiple_chains():
    rng = np.random.default_rng(42)
    n_draws = 50
    n_chains = 4
    eval_points = np.linspace(0.1, 0.9, n_draws - 1)
    prob = 0.9
    n_simulations = 50
    result = simulate_confidence_bands(n_draws, n_chains, eval_points, prob, n_simulations, rng)
    assert isinstance(result, float)
    assert 0 <= result <= 1


@pytest.mark.parametrize(
    "n_vals, ci_prob, n_chains",
    [
        (100, 0.95, 1),
        (200, 0.9, 4),
    ],
)
def test_ecdf_pit(n_vals, ci_prob, n_chains):
    rng = np.random.default_rng(42)
    vals = rng.uniform(0, 1, n_vals)
    n_simulations = 50
    eval_points, ecdf, lower, upper = ecdf_pit(vals, ci_prob, n_simulations, n_chains, rng)
    assert len(eval_points) == len(ecdf)
    assert len(ecdf) == len(lower)
    assert len(lower) == len(upper)
    assert np.all((ecdf >= 0) & (ecdf <= 1))
    assert np.all(lower <= upper)


def test_ecdf_pit_deterministic():
    vals = np.linspace(0, 1, 100)
    ci_prob = 0.95
    n_simulations = 50
    eval_points, ecdf, _, _ = ecdf_pit(vals, ci_prob, n_simulations)
    assert len(eval_points) == 101
    assert np.all(np.diff(ecdf) >= 0)
