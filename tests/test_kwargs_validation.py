"""
Comprehensive tests for kwargs validation across arviz-stats functions.

These tests ensure that functions properly raise TypeError when given
unexpected keyword arguments, preventing silent failures from typos.

Related to issue #142 and PR #143.
"""

# pylint: disable=redefined-outer-name, no-self-use, unexpected-keyword-arg
import pytest
from arviz_base import load_arviz_data


class TestKwargsValidationSamplingDiagnostics:
    """Test kwargs validation for sampling diagnostic functions."""

    @pytest.fixture(scope="class")
    def idata(self):
        """Load test data."""
        return load_arviz_data("non_centered_eight")

    def test_ess_rejects_invalid_kwargs(self, idata):
        """Test that ess() raises TypeError for invalid kwargs."""
        import arviz_stats as azs

        with pytest.raises(TypeError, match=".*unexpected keyword argument.*"):
            azs.ess(idata.posterior, invalid_kwarg="test")

        # Also test common typos
        with pytest.raises(TypeError, match=".*unexpected keyword argument.*"):
            azs.ess(idata.posterior, methods="bulk")  # should be 'method'

    def test_rhat_rejects_invalid_kwargs(self, idata):
        """Test that rhat() raises TypeError for invalid kwargs."""
        import arviz_stats as azs

        with pytest.raises(TypeError, match=".*unexpected keyword argument.*"):
            azs.rhat(idata.posterior, invalid_kwarg="test")

        # Test common typo
        with pytest.raises(TypeError, match=".*unexpected keyword argument.*"):
            azs.rhat(idata.posterior, methods="rank")  # should be 'method'

    def test_mcse_rejects_invalid_kwargs(self, idata):
        """Test that mcse() raises TypeError for invalid kwargs."""
        import arviz_stats as azs

        with pytest.raises(TypeError, match=".*unexpected keyword argument.*"):
            azs.mcse(idata.posterior, invalid_kwarg="test")

        # Test common typo
        with pytest.raises(TypeError, match=".*unexpected keyword argument.*"):
            azs.mcse(idata.posterior, methods="mean")  # should be 'method'

    def test_bfmi_rejects_invalid_kwargs(self, idata):
        """Test that bfmi() raises TypeError for invalid kwargs."""
        import arviz_stats as azs

        with pytest.raises(TypeError, match=".*unexpected keyword argument.*"):
            azs.bfmi(idata, invalid_kwarg="test")

    def test_diagnose_rejects_invalid_kwargs(self, idata):
        """Test that diagnose() raises TypeError for invalid kwargs."""
        import arviz_stats as azs

        with pytest.raises(TypeError, match=".*unexpected keyword argument.*"):
            azs.diagnose(idata, invalid_kwarg="test")


class TestKwargsValidationSummaryStatistics:
    """Test kwargs validation for summary statistic functions."""

    @pytest.fixture(scope="class")
    def idata(self):
        """Load test data."""
        return load_arviz_data("non_centered_eight")

    def test_summary_rejects_invalid_kwargs(self, idata):
        """Test that summary() raises TypeError for invalid kwargs."""
        import arviz_stats as azs

        with pytest.raises(TypeError, match=".*unexpected keyword argument.*"):
            azs.summary(idata, invalid_kwarg="test")

        # Test common typo
        with pytest.raises(TypeError, match=".*unexpected keyword argument.*"):
            azs.summary(idata, sample_dim=["chain", "draw"])  # should be 'sample_dims'

    def test_mean_rejects_invalid_kwargs(self, idata):
        """Test that mean() raises TypeError for invalid kwargs."""
        import arviz_stats as azs

        with pytest.raises(TypeError, match=".*unexpected keyword argument.*"):
            azs.mean(idata.posterior, invalid_kwarg="test")

        # Test common typo (errors with "multiple values" due to internal dims variable)
        with pytest.raises(TypeError):
            azs.mean(idata.posterior, dims="draw")  # should be 'dim'

    def test_median_rejects_invalid_kwargs(self, idata):
        """Test that median() raises TypeError for invalid kwargs."""
        import arviz_stats as azs

        with pytest.raises(TypeError, match=".*unexpected keyword argument.*"):
            azs.median(idata.posterior, invalid_kwarg="test")

        # Test common typo (errors with "multiple values" due to internal dims variable)
        with pytest.raises(TypeError):
            azs.median(idata.posterior, dims="draw")  # should be 'dim'

    def test_mode_rejects_invalid_kwargs(self, idata):
        """Test that mode() raises TypeError for invalid kwargs."""
        import arviz_stats as azs

        with pytest.raises(TypeError, match=".*unexpected keyword argument.*"):
            azs.mode(idata.posterior, invalid_kwarg="test")

        # Test common typo (errors with "multiple values" due to internal dims variable)
        with pytest.raises(TypeError):
            azs.mode(idata.posterior, dims="draw")  # should be 'dim'

    def test_ci_in_rope_rejects_invalid_kwargs(self, idata):
        """Test that ci_in_rope() raises TypeError for invalid kwargs."""
        import arviz_stats as azs

        with pytest.raises(TypeError, match=".*unexpected keyword argument.*"):
            azs.ci_in_rope(idata.posterior["mu"], rope=(-1, 1), invalid_kwarg="test")


class TestKwargsValidationVisualizationFunctions:
    """Test kwargs validation for visualization/density functions."""

    @pytest.fixture(scope="class")
    def idata(self):
        """Load test data."""
        return load_arviz_data("non_centered_eight")

    def test_qds_rejects_invalid_kwargs(self, idata):
        """Test that qds() raises TypeError for invalid kwargs."""
        import arviz_stats as azs

        with pytest.raises(TypeError, match=".*unexpected keyword argument.*"):
            azs.qds(idata.posterior["mu"], invalid_kwarg="test")

    def test_ecdf_rejects_invalid_kwargs(self, idata):
        """Test that ecdf() raises TypeError for invalid kwargs."""
        import arviz_stats as azs

        with pytest.raises(TypeError, match=".*unexpected keyword argument.*"):
            azs.ecdf(idata.posterior["mu"], invalid_kwarg="test")


class TestKwargsValidationLOOFunctions:
    """Test kwargs validation for LOO/model comparison functions."""

    @pytest.fixture(scope="class")
    def idata(self):
        """Load test data."""
        return load_arviz_data("non_centered_eight")

    def test_loo_rejects_invalid_kwargs(self, idata):
        """Test that loo() raises TypeError for invalid kwargs."""
        import arviz_stats as azs

        with pytest.raises(TypeError, match=".*unexpected keyword argument.*"):
            azs.loo(idata, invalid_kwarg="test")

    def test_compare_rejects_invalid_kwargs(self, idata):
        """Test that compare() raises TypeError for invalid kwargs."""
        import arviz_stats as azs

        with pytest.raises(TypeError, match=".*unexpected keyword argument.*"):
            azs.compare({"model": idata}, invalid_kwarg="test")


class TestKwargsValidationMetrics:
    """Test kwargs validation for metric functions."""

    @pytest.fixture(scope="class")
    def idata(self):
        """Load test data."""
        return load_arviz_data("non_centered_eight")

    def test_bayesian_r2_rejects_invalid_kwargs(self, idata):
        """Test that bayesian_r2() raises TypeError for invalid kwargs."""
        import arviz_stats as azs

        with pytest.raises(TypeError, match=".*unexpected keyword argument.*"):
            azs.bayesian_r2(idata, invalid_kwarg="test")

    def test_metrics_rejects_invalid_kwargs(self, idata):
        """Test that metrics() raises TypeError for invalid kwargs."""
        import arviz_stats as azs

        with pytest.raises(TypeError, match=".*unexpected keyword argument.*"):
            azs.metrics(idata, invalid_kwarg="test")


class TestKwargsValidationOtherFunctions:
    """Test kwargs validation for other utility functions."""

    @pytest.fixture(scope="class")
    def idata(self):
        """Load test data."""
        return load_arviz_data("non_centered_eight")

    def test_thin_rejects_invalid_kwargs(self, idata):
        """Test that thin() raises TypeError for invalid kwargs."""
        import arviz_stats as azs

        with pytest.raises(TypeError, match=".*unexpected keyword argument.*"):
            azs.thin(idata, factor=2, invalid_kwarg="test")

    def test_weight_predictions_rejects_invalid_kwargs(self, idata):
        """Test that weight_predictions() raises TypeError for invalid kwargs."""
        import arviz_stats as azs

        with pytest.raises(TypeError, match=".*unexpected keyword argument.*"):
            azs.weight_predictions(idata.posterior_predictive, invalid_kwarg="test")

    def test_bayes_factor_rejects_invalid_kwargs(self, idata):
        """Test that bayes_factor() raises TypeError for invalid kwargs."""
        import arviz_stats as azs

        with pytest.raises(TypeError, match=".*unexpected keyword argument.*"):
            azs.bayes_factor(idata.posterior, var_names="mu", invalid_kwarg="test")

    def test_psense_rejects_invalid_kwargs(self, idata):
        """Test that psense() raises TypeError for invalid kwargs."""
        import arviz_stats as azs

        with pytest.raises(TypeError, match=".*unexpected keyword argument.*"):
            azs.psense(idata, var_name="mu", invalid_kwarg="test")


class TestKwargsValidationAccessors:
    """Test kwargs validation for accessor methods."""

    @pytest.fixture(scope="class")
    def data_array(self):
        """Load test data and extract DataArray."""
        idata = load_arviz_data("non_centered_eight")
        return idata.posterior["mu"]

    def test_accessor_ess_rejects_invalid_kwargs(self, data_array):
        """Test that .azstats.ess() raises TypeError for invalid kwargs."""
        with pytest.raises(TypeError, match=".*unexpected keyword argument.*"):
            data_array.azstats.ess(invalid_kwarg="test")

    def test_accessor_rhat_rejects_invalid_kwargs(self, data_array):
        """Test that .azstats.rhat() raises TypeError for invalid kwargs."""
        with pytest.raises(TypeError, match=".*unexpected keyword argument.*"):
            data_array.azstats.rhat(invalid_kwarg="test")

    def test_accessor_mcse_rejects_invalid_kwargs(self, data_array):
        """Test that .azstats.mcse() raises TypeError for invalid kwargs."""
        with pytest.raises(TypeError, match=".*unexpected keyword argument.*"):
            data_array.azstats.mcse(invalid_kwarg="test")

    def test_accessor_mean_rejects_invalid_kwargs(self, data_array):
        """Test that .azstats.mean() raises TypeError for invalid kwargs."""
        with pytest.raises(TypeError, match=".*unexpected keyword argument.*"):
            data_array.azstats.mean(invalid_kwarg="test")

    def test_accessor_median_rejects_invalid_kwargs(self, data_array):
        """Test that .azstats.median() raises TypeError for invalid kwargs."""
        with pytest.raises(TypeError, match=".*unexpected keyword argument.*"):
            data_array.azstats.median(invalid_kwarg="test")

    def test_accessor_mode_rejects_invalid_kwargs(self, data_array):
        """Test that .azstats.mode() raises TypeError for invalid kwargs."""
        with pytest.raises(TypeError, match=".*unexpected keyword argument.*"):
            data_array.azstats.mode(invalid_kwarg="test")
