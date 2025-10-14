# pylint: disable=redefined-outer-name, unused-import, unused-argument
# ruff: noqa: F811, F401
import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from ..helpers import importorskip

azb = importorskip("arviz_base")

from arviz_stats import loo, loo_score


@pytest.mark.parametrize("kind", ["crps", "scrps"])
def test_loo_score_basic(centered_eight, kind):
    result = loo_score(centered_eight, kind=kind)

    assert hasattr(result, "mean")
    assert hasattr(result, "se")
    assert np.isfinite(result.mean)
    assert np.isfinite(result.se)


@pytest.mark.parametrize("kind", ["crps", "scrps"])
def test_loo_score_log_weights(centered_eight, kind):
    loo_result = loo(centered_eight, pointwise=True)

    r1 = loo_score(centered_eight, kind=kind)
    r2 = loo_score(
        centered_eight,
        log_weights=loo_result.log_weights,
        pareto_k=loo_result.pareto_k,
        kind=kind,
    )

    assert_almost_equal(r1.mean, r2.mean, decimal=10)
    assert_almost_equal(r1.se, r2.se, decimal=10)


@pytest.mark.parametrize("kind", ["crps", "scrps"])
@pytest.mark.parametrize(
    "scenario, pattern",
    [
        ("missing_sample_dim", "y_pred must include sample dimension 'draw'"),
        ("obs_dim_size_mismatch", "Size mismatch in observation dim 'school'"),
        ("loglik_sample_dim_mismatch", "Size mismatch in sample dimension 'draw'"),
    ],
)
def test_loo_score_validation_errors(centered_eight, kind, scenario, pattern):
    broken = centered_eight.copy()

    if scenario == "missing_sample_dim":
        broken.posterior_predictive["obs"] = broken.posterior_predictive["obs"].mean("draw")
    elif scenario == "obs_dim_size_mismatch":
        broken.observed_data["obs"] = broken.observed_data["obs"].isel(school=slice(0, 7))
    elif scenario == "loglik_sample_dim_mismatch":
        broken.log_likelihood["obs"] = broken.log_likelihood["obs"].isel(draw=slice(0, -1))

    with pytest.raises(ValueError, match=pattern):
        loo_score(broken, kind=kind)
