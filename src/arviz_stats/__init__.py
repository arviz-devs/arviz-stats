# pylint: disable=wildcard-import
"""Statistical computation and diagnostics for ArviZ."""

try:
    from arviz_stats.utils import *
    from arviz_stats.accessors import *
    from arviz_stats.loo import (
        loo,
        loo_expectations,
        loo_metrics,
        loo_pit,
        loo_approximate_posterior,
        loo_subsample,
        update_subsample,
        loo_moment_match,
        reloo,
        compare,
        SamplingWrapper,
    )
    from arviz_stats.psense import psense, psense_summary
    from arviz_stats.metrics import kl_divergence, metrics, r2_score, wasserstein
    from arviz_stats.sampling_diagnostics import ess, mcse, rhat, rhat_nested
    from arviz_stats.summary import summary, ci_in_rope
    from arviz_stats.manipulation import thin
    from arviz_stats.bayes_factor import bayes_factor
    from arviz_stats.visualization import eti, hdi, histogram, kde

except ModuleNotFoundError:
    pass
