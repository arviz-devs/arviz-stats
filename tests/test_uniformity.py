"""Tests for uniformity_test array interface."""

# pylint: disable=no-self-use, protected-access
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import beta, binom, expon

from arviz_stats.base import array_stats


def _cauchy_combination(ps, truncate=True):
    ps = np.asarray(ps, dtype=float)
    cauchy_vals = np.tan((0.5 - ps) * np.pi)
    if truncate:
        idx = ps < 0.5
        if not np.any(idx):
            raise ValueError(
                "Cannot compute truncated Cauchy combination test. No p-values below 0.5 found."
            )
        cauchy_vals = cauchy_vals[idx]
    mean_val = np.mean(cauchy_vals)
    return 0.5 - np.arctan(mean_val) / np.pi


class TestShapleyMean:
    """Tests for the closed-form Shapley contribution helper."""

    def test_empty_array(self):
        result = array_stats._shapley_mean(np.array([]))
        assert len(result) == 0

    def test_single_element(self):
        result = array_stats._shapley_mean(np.array([5.0]))
        assert_allclose(result, [0.0])
        assert len(result) == 1

    def test_two_elements(self):
        """Manual calculation for n=2.

        harmonic_number = 1 + 1/2 = 1.5
        For i=0: mean_others = 2/1 = 2
            shapley[0] = (1/2)*1 + ((1.5-1)/2)*(1-2) = 0.5 - 0.25 = 0.25
        For i=1: mean_others = 1/1 = 1
            shapley[1] = (1/2)*2 + ((1.5-1)/2)*(2-1) = 1 + 0.25 = 1.25
        """
        result = array_stats._shapley_mean(np.array([1.0, 2.0]))
        expected = np.array([0.25, 1.25])
        assert_allclose(result, expected, atol=1e-10)
        assert len(result) == 2

    def test_mixed_values(self):
        x = np.array([-0.2, 0.0, 2.0, 3.1, 4.2])
        result = array_stats._shapley_mean(x)
        assert len(result) == 5
        assert np.all(np.isfinite(result))


class TestCauchyCombination:
    """Tests for the Cauchy combination logic embedded in the test functions."""

    def test_non_truncated(self):
        ps = np.array([0.1, 0.2, 0.3])
        cauchy_vals = np.tan((0.5 - ps) * np.pi)
        expected = 0.5 - np.arctan(np.mean(cauchy_vals)) / np.pi
        result = _cauchy_combination(ps, truncate=False)
        assert_allclose(result, expected, atol=1e-10)
        assert 0 <= result <= 1

    def test_truncated(self):
        ps = np.array([0.1, 0.2, 0.3, 0.4, 0.7, 0.8])
        cauchy_vals = np.tan((0.5 - ps) * np.pi)
        idx = ps < 0.5
        truncated_mean = np.mean(cauchy_vals[idx])
        expected_trunc = 0.5 - np.arctan(truncated_mean) / np.pi
        result = _cauchy_combination(ps, truncate=True)
        assert_allclose(result, expected_trunc, atol=1e-10)
        assert 0 <= result <= 1

    def test_truncated_no_values_below_half_raises(self):
        ps = np.array([0.5, 0.7, 0.8, 0.9])
        with pytest.raises(ValueError, match="No p-values below 0.5 found"):
            _cauchy_combination(ps, truncate=True)

    def test_compute_cauchy_transform(self):
        """tan((0.5 - x) * pi) at known points."""
        assert_allclose(np.tan((0.5 - 0.5) * np.pi), 0.0, atol=1e-10)
        assert_allclose(np.tan((0.5 - 0.25) * np.pi), 1.0, atol=1e-10)
        assert_allclose(np.tan((0.5 - 0.75) * np.pi), -1.0, atol=1e-10)


class TestUniformityTestPotC:
    def test_basic_values(self):
        """For sorted x = [0.1, 0.9], n=2:

        Beta(1,2).cdf(0.1) ≈ 0.19
        Beta(2,1).cdf(0.9) ≈ 0.99
        ps = 2*min(cdf, 1-cdf) = [0.38, 0.02] (both < 0.5 for truncation)
        """
        x = np.array([0.1, 0.9])
        p_value, shapley = array_stats.uniformity_test(x, method="pot_c")

        x_sorted = np.sort(x)
        n = len(x)
        i_vals = np.arange(1, n + 1)
        probs = beta.cdf(x_sorted, i_vals, i_vals[::-1])
        ps = 2 * np.minimum(probs, 1 - probs)
        expected_p = _cauchy_combination(ps, truncate=True)

        assert_allclose(p_value, expected_p, atol=1e-10)
        assert 0 <= p_value <= 1
        assert len(shapley) == len(x)

    def test_uniform_samples_high_pvalue(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(size=200)
        p_value, shapley = array_stats.uniformity_test(x, method="pot_c")
        assert p_value > 0.05, f"Uniform sample yielded p={p_value}, expected > 0.05"
        assert len(shapley) == len(x)

    def test_non_uniform_samples_low_pvalue(self):
        rng = np.random.default_rng(123)
        x = rng.beta(2, 5, size=200)
        p_value, _ = array_stats.uniformity_test(x, method="pot_c")
        assert p_value < 0.05, f"Non-uniform sample yielded p={p_value}, expected < 0.05"

    def test_shapley_length_matches_input(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(size=50)
        _, shapley = array_stats.uniformity_test(x, method="pot_c")
        assert shapley.shape == (len(x),)

    def test_batched_2d(self):
        rng = np.random.default_rng(7)
        x = rng.uniform(size=(3, 100))
        p_value, shapley = array_stats.uniformity_test(x, axis=-1, method="pot_c")
        assert p_value.shape == (3,)
        assert shapley.shape == (3, 100)


class TestUniformityTestPritC:
    def test_basic_values(self):
        """For x = [0.1, 0.9], n=2:

        sorted x = [0.1, 0.9]
        ranks = searchsorted(side='right') = [1, 2]
        Test binomial probabilities to ensure some ps < 0.5 for truncation.
        """
        x = np.array([0.1, 0.9])
        p_value, shapley = array_stats.uniformity_test(x, method="prit_c")

        x_sorted = np.sort(x)
        len_x = len(x_sorted)
        ranks = np.searchsorted(x_sorted, x_sorted, side="right")
        probs1 = binom.cdf(ranks, len_x, x_sorted)
        probs2 = binom.cdf(ranks - 1, len_x, x_sorted)
        ps = 2 * np.minimum(probs1, 1 - probs2)
        expected_p = _cauchy_combination(ps, truncate=True)

        assert_allclose(p_value, expected_p, atol=1e-10)
        assert len(shapley) == len(x)

    def test_uniform_samples_high_pvalue(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(size=200)
        p_value, shapley = array_stats.uniformity_test(x, method="prit_c")
        assert p_value > 0.05, f"Uniform sample yielded p={p_value}, expected > 0.05"
        assert len(shapley) == len(x)

    def test_non_uniform_samples_low_pvalue(self):
        rng = np.random.default_rng(123)
        x = rng.beta(2, 5, size=200)
        p_value, _ = array_stats.uniformity_test(x, method="prit_c")
        assert p_value < 0.05, f"Non-uniform sample yielded p={p_value}, expected < 0.05"

    def test_batched_2d(self):
        rng = np.random.default_rng(7)
        x = rng.uniform(size=(3, 100))
        p_value, shapley = array_stats.uniformity_test(x, axis=-1, method="prit_c")
        assert p_value.shape == (3,)
        assert shapley.shape == (3, 100)


class TestUniformityTestPietC:
    def test_basic_values(self):
        """pe = expon.cdf(-log(x)), ps = 2*min(pe, 1-pe), then Cauchy combination."""
        x = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        p_value, shapley = array_stats.uniformity_test(x, method="piet_c")

        pe = expon.cdf(-np.log(x))
        ps = 2 * np.minimum(pe, 1 - pe)
        expected_p = _cauchy_combination(ps, truncate=False)

        assert_allclose(p_value, expected_p, atol=1e-10)
        assert len(shapley) == len(x)

    def test_uniform_samples_high_pvalue(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(size=200)
        p_value, shapley = array_stats.uniformity_test(x, method="piet_c")
        assert p_value > 0.05, f"Uniform sample yielded p={p_value}, expected > 0.05"
        assert len(shapley) == len(x)

    def test_non_uniform_samples_low_pvalue(self):
        rng = np.random.default_rng(123)
        x = rng.beta(0.5, 0.5, size=500)
        p_value, _ = array_stats.uniformity_test(x, method="piet_c")
        assert p_value < 0.05, f"Non-uniform sample yielded p={p_value}, expected < 0.05"

    def test_batched_2d(self):
        rng = np.random.default_rng(7)
        x = rng.uniform(size=(3, 100))
        p_value, shapley = array_stats.uniformity_test(x, axis=-1, method="piet_c")
        assert p_value.shape == (3,)
        assert shapley.shape == (3, 100)


class TestUniformityTestGeneral:
    def test_invalid_method_raises(self):
        x = np.array([0.1, 0.5, 0.9])
        with pytest.raises(ValueError, match="Requested method"):
            array_stats.uniformity_test(x, method="invalid")

    def test_default_method_is_pot_c(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(size=50)
        p_default, sh_default = array_stats.uniformity_test(x)
        p_pot, sh_pot = array_stats.uniformity_test(x, method="pot_c")
        assert_allclose(p_default, p_pot)
        assert_allclose(sh_default, sh_pot)

    @pytest.mark.parametrize("method", ["pot_c", "prit_c", "piet_c"])
    def test_pvalue_in_unit_interval(self, method):
        rng = np.random.default_rng(0)
        x = rng.uniform(size=50)
        p_value, _ = array_stats.uniformity_test(x, method=method)
        assert 0 <= p_value <= 1

    @pytest.mark.parametrize("method", ["pot_c", "prit_c", "piet_c"])
    def test_axis_none(self, method):
        """axis=None should flatten the input."""
        rng = np.random.default_rng(11)
        x = rng.uniform(size=(4, 50))
        p_value, shapley = array_stats.uniformity_test(x, axis=None, method=method)
        assert np.ndim(p_value) == 0
        assert shapley.shape == (200,)

    @pytest.mark.parametrize("method", ["pot_c", "prit_c", "piet_c"])
    def test_3d_batched(self, method):
        """Works with higher-rank batch dimensions."""
        rng = np.random.default_rng(99)
        x = rng.uniform(size=(2, 3, 80))
        p_value, shapley = array_stats.uniformity_test(x, axis=-1, method=method)
        assert p_value.shape == (2, 3)
        assert shapley.shape == (2, 3, 80)
