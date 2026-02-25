# pylint: disable=wildcard-import
"""Statistical computation and diagnostics for ArviZ."""

from arviz_stats._version import __version__

try:
    from arviz_stats.utils import *
    from arviz_stats.accessors import *
    from arviz_stats.loo import (
        loo,
        loo_i,
        loo_expectations,
        loo_metrics,
        loo_r2,
        loo_score,
        loo_pit,
        loo_approximate_posterior,
        loo_subsample,
        update_subsample,
        loo_moment_match,
        reloo,
        compare,
        SamplingWrapper,
        loo_kfold,
        lfo_cv,
    )
    from arviz_stats.psense import psense, psense_summary
    from arviz_stats.metrics import bayesian_r2, kl_divergence, metrics, residual_r2, wasserstein
    from arviz_stats.sampling_diagnostics import bfmi, ess, mcse, rhat, rhat_nested
    from arviz_stats.summary import summary, ci_in_rope, mean, median, mode
    from arviz_stats.manipulation import thin, weight_predictions
    from arviz_stats.bayes_factor import bayes_factor
    from arviz_stats.visualization import ecdf, eti, hdi, histogram, kde, qds
    from arviz_stats.survival import kaplan_meier, generate_survival_curves

except ModuleNotFoundError:
    pass
