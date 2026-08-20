# File generated with docstub

from arviz_stats._version import __version__

try:
    from arviz_stats.accessors import *
    from arviz_stats.bayes_factor import bayes_factor
    from arviz_stats.loo import (
        SamplingWrapper,
        compare,
        loo,
        loo_approximate_posterior,
        loo_expectations,
        loo_i,
        loo_influence,
        loo_kfold,
        loo_metrics,
        loo_moment_match,
        loo_pit,
        loo_r2,
        loo_score,
        loo_subsample,
        reloo,
        update_subsample,
    )
    from arviz_stats.manipulation import thin, weight_predictions
    from arviz_stats.metrics import (
        bayesian_r2,
        kl_divergence,
        metrics,
        residual_r2,
        wasserstein,
    )
    from arviz_stats.psense import psense, psense_summary
    from arviz_stats.sampling_diagnostics import (
        bfmi,
        diagnose,
        ess,
        mcse,
        rhat,
        rhat_nested,
    )
    from arviz_stats.summary import (
        ci_in_rope,
        iqr,
        mad,
        mean,
        median,
        mode,
        std,
        summary,
        var,
    )
    from arviz_stats.survival import generate_survival_curves, kaplan_meier
    from arviz_stats.utils import *
    from arviz_stats.visualization import ecdf, eti, hdi, histogram, kde, kde2d, qds

except ModuleNotFoundError:
    pass
