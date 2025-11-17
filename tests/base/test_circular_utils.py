"""Tests for circular_utils."""

import numpy as np
from numpy.testing import assert_allclose

from arviz_stats.base.circular_utils import (
    circular_diff,
    circular_mean,
    circular_sd,
    circular_var,
    mean_resultant_length,
)


def test_circular_diff_zero():
    """Test that difference of identical angles is zero."""
    angles = np.array([0, np.pi / 2, np.pi, -np.pi / 2])
    result = circular_diff(angles, angles)
    assert_allclose(result, np.zeros_like(angles), atol=1e-10)


def test_circular_diff_positive():
    """Test positive angular differences."""
    array0 = np.array([np.pi / 2])
    array1 = np.array([0])
    result = circular_diff(array0, array1)
    assert_allclose(result, np.pi / 2, atol=1e-10)


def test_circular_diff_negative():
    """Test negative angular differences."""
    array0 = np.array([0])
    array1 = np.array([np.pi / 2])
    result = circular_diff(array0, array1)
    assert_allclose(result, -np.pi / 2, atol=1e-10)


def test_circular_diff_wrapping_positive():
    """Test wrapping for differences > π."""
    array0 = np.array([-np.pi + 0.1])
    array1 = np.array([np.pi - 0.1])
    result = circular_diff(array0, array1)
    assert_allclose(result, 0.2, atol=1e-10)


def test_circular_diff_wrapping_negative():
    """Test wrapping for differences < -π."""
    array0 = np.array([np.pi - 0.1])
    array1 = np.array([-np.pi + 0.1])
    result = circular_diff(array0, array1)
    assert_allclose(result, -0.2, atol=1e-10)


def test_circular_diff_array():
    """Test with array inputs."""
    array0 = np.array([0, np.pi / 4, np.pi / 2, np.pi])
    array1 = np.array([0, 0, 0, 0])
    expected = np.array([0, np.pi / 4, np.pi / 2, -np.pi])
    result = circular_diff(array0, array1)
    assert_allclose(result, expected, atol=1e-10)


def test_circular_diff_2d():
    """Test with 2D array inputs."""
    array0 = np.array([[0, np.pi / 2], [np.pi, -np.pi / 2]])
    array1 = np.array([[0, 0], [0, 0]])
    expected = np.array([[0, np.pi / 2], [-np.pi, -np.pi / 2]])
    result = circular_diff(array0, array1)
    assert_allclose(result, expected, atol=1e-10)


def test_circular_mean_identical_angles():
    """Test mean of identical angles."""
    angles = np.array([[0, 0, 0], [np.pi / 2, np.pi / 2, np.pi / 2]])
    result = circular_mean(angles)
    expected = np.array([0, np.pi / 2])
    assert_allclose(result, expected, atol=1e-10)


def test_circular_mean_opposite_angles():
    """Test mean of opposite angles (should be near 0 or π)."""
    angles = np.array([[0, np.pi]])
    result = circular_mean(angles)
    assert np.isfinite(result).all()


def test_circular_mean_uniform_circle():
    """Test mean of uniformly distributed angles around circle."""
    angles = np.array([[0, np.pi / 2, np.pi, -np.pi / 2]])
    result = circular_mean(angles)
    assert np.isfinite(result).all()


def test_circular_mean_weighted():
    """Test weighted circular mean."""
    angles = np.array([[0, np.pi / 2]])
    weights = np.array([[0.9, 0.1]])
    result = circular_mean(angles, weights=weights)
    assert result[0] < np.pi / 4


def test_circular_mean_positive_quadrant():
    """Test mean of angles in positive quadrant."""
    angles = np.array([[0, np.pi / 4, np.pi / 2]])
    result = circular_mean(angles)
    assert 0 < result[0] < np.pi / 2


def test_circular_mean_2d_input():
    """Test with 2D input arrays."""
    angles = np.array([[0, np.pi / 4, np.pi / 2], [np.pi, -np.pi / 2, -np.pi / 4]])
    result = circular_mean(angles)
    assert result.shape == (2,)
    assert np.isfinite(result).all()


def test_circular_mean_weights_normalized():
    """Test that weights are used correctly."""
    angles = np.array([[0, 0, np.pi / 2]])
    weights = np.array([[0.5, 0.5, 0]])
    result = circular_mean(angles, weights=weights)
    assert_allclose(result, [0], atol=1e-10)


def test_mrl_identical_angles():
    """Test MRL of identical angles is 1."""
    angles = np.array([[0, 0, 0], [np.pi, np.pi, np.pi]])
    result = mean_resultant_length(angles)
    expected = np.array([1.0, 1.0])
    assert_allclose(result, expected, atol=1e-10)


def test_mrl_uniform_circle():
    """Test MRL of uniformly distributed angles is close to 0."""
    n_angles = 8
    angles = np.array([np.linspace(-np.pi, np.pi, n_angles, endpoint=False)])
    result = mean_resultant_length(angles)
    assert_allclose(result, [0], atol=0.01)


def test_mrl_opposite_angles():
    """Test MRL of opposite angles."""
    angles = np.array([[0, np.pi]])
    result = mean_resultant_length(angles)
    assert_allclose(result, [0], atol=1e-10)


def test_mrl_weighted():
    """Test weighted MRL."""
    angles = np.array([[0, np.pi]])
    weights = np.array([[1, 0]])
    result = mean_resultant_length(angles, weights=weights)
    assert_allclose(result, [1.0], atol=1e-10)


def test_mrl_range():
    """Test that MRL is always in [0, 1]."""
    rng = np.random.default_rng(42)
    angles = rng.uniform(-np.pi, np.pi, size=(10, 20))
    result = mean_resultant_length(angles)
    assert (result >= 0).all()
    assert (result <= 1).all()


def test_mrl_concentrated_angles():
    """Test MRL for concentrated angles."""
    rng = np.random.default_rng(42)
    angles = rng.normal(0, 0.1, size=(5, 100))
    result = mean_resultant_length(angles)
    assert (result > 0.9).all()


def test_circular_var_identical_angles():
    """Test variance of identical angles is 0."""
    angles = np.array([[0, 0, 0], [np.pi, np.pi, np.pi]])
    result = circular_var(angles)
    expected = np.array([0.0, 0.0])
    assert_allclose(result, expected, atol=1e-10)


def test_circular_var_uniform_circle():
    """Test variance of uniformly distributed angles is close to 1."""
    n_angles = 8
    angles = np.array([np.linspace(-np.pi, np.pi, n_angles, endpoint=False)])
    result = circular_var(angles)
    assert_allclose(result, [1], atol=0.01)


def test_circular_var_relationship_to_mrl():
    """Test that circular_var = 1 - mean_resultant_length."""
    rng = np.random.default_rng(42)
    angles = rng.uniform(-np.pi, np.pi, size=(10, 20))
    var_result = circular_var(angles)
    mrl_result = mean_resultant_length(angles)
    assert_allclose(var_result, 1 - mrl_result, atol=1e-10)


def test_circular_var_range():
    """Test that circular variance is in [0, 1]."""
    rng = np.random.default_rng(42)
    angles = rng.uniform(-np.pi, np.pi, size=(10, 20))
    result = circular_var(angles)
    assert (result >= 0).all()
    assert (result <= 1).all()


def test_circular_var_weighted():
    """Test weighted circular variance."""
    angles = np.array([[0, np.pi]])
    weights = np.array([[1, 0]])
    result = circular_var(angles, weights=weights)
    assert_allclose(result, [0.0], atol=1e-10)


def test_circular_sd_identical_angles():
    """Test SD of identical angles is 0."""
    angles = np.array([[0, 0, 0], [np.pi, np.pi, np.pi]])
    result = circular_sd(angles)
    expected = np.array([0.0, 0.0])
    assert_allclose(result, expected, atol=1e-10)


def test_circular_sd_uniform_circle():
    """Test SD of uniformly distributed angles is large."""
    n_angles = 8
    angles = np.array([np.linspace(-np.pi, np.pi, n_angles, endpoint=False)])
    result = circular_sd(angles)
    assert result[0] > 2.0


def test_circular_sd_non_negative():
    """Test that circular SD is non-negative."""
    rng = np.random.default_rng(42)
    angles = rng.uniform(-np.pi, np.pi, size=(10, 20))
    result = circular_sd(angles)
    assert (result >= 0).all()


def test_circular_sd_relationship_to_mrl():
    """Test that circular_sd = sqrt(-2 * log(mean_resultant_length))."""
    rng = np.random.default_rng(42)
    angles = rng.uniform(-np.pi, np.pi, size=(10, 20))
    angles = rng.normal(0, 1, size=(10, 20))
    sd_result = circular_sd(angles)
    mrl_result = mean_resultant_length(angles)
    expected = np.sqrt(-2 * np.log(mrl_result))
    assert_allclose(sd_result, expected, atol=1e-10)


def test_circular_sd_concentrated_angles():
    """Test SD for concentrated angles is small."""
    rng = np.random.default_rng(42)
    angles = rng.normal(0, 0.1, size=(5, 100))
    result = circular_sd(angles)
    assert (result < 0.5).all()


def test_circular_sd_weighted():
    """Test weighted circular SD."""
    angles = np.array([[0, np.pi]])
    weights = np.array([[1, 0]])
    result = circular_sd(angles, weights=weights)
    assert_allclose(result, [0.0], atol=1e-10)


def test_circular_sd_handles_zero_mrl():
    """Test that SD handles near-zero MRL (returns inf or very large value)."""
    angles = np.array([[0, np.pi]])
    result = circular_sd(angles)
    assert result[0] > 5.0 or np.isinf(result[0])
