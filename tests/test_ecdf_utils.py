"""Test PIT computations for plotting distributions."""

# pylint: disable=redefined-outer-name
import numpy as np
import pytest
from numpy.testing import assert_allclose

from .helpers import importorskip

azb = importorskip("arviz_base")

from arviz_stats.ecdf_utils import (
    compute_pit_for_histogram,
    compute_pit_for_kde,
    compute_pit_for_qds,
)

PIT_FUNCTIONS = {
    "histogram": compute_pit_for_histogram,
    "kde": compute_pit_for_kde,
    "qds": compute_pit_for_qds,
}


@pytest.fixture(scope="module")
def posterior():
    rng = np.random.default_rng(0)
    dt = azb.from_dict(
        {"posterior": {"mu": rng.normal(size=(1, 200)), "theta": rng.normal(size=(1, 200, 3))}}
    )
    return dt.posterior.dataset


@pytest.mark.parametrize("kind", PIT_FUNCTIONS)
def test_pit_chain_draw(posterior, kind):
    stats = getattr(posterior.azstats, kind)(dim=["chain", "draw"])
    pit = PIT_FUNCTIONS[kind](posterior, stats, sample_dims=["chain", "draw"])
    assert pit["mu"].sizes == {"sample": 200}
    assert pit["theta"].sizes == {"sample": 200, "theta_dim_0": 3}
    for var in pit.data_vars.values():
        assert np.all((var >= 0) & (var <= 1))


@pytest.mark.parametrize("kind", PIT_FUNCTIONS)
@pytest.mark.parametrize("sample_dim", ["draw", "sample"])
def test_pit_single_sample_dim(posterior, kind, sample_dim):
    expected_stats = getattr(posterior.azstats, kind)(dim=["chain", "draw"])
    expected = PIT_FUNCTIONS[kind](posterior, expected_stats, sample_dims=["chain", "draw"])

    if sample_dim == "draw":
        data = posterior.squeeze("chain", drop=True)
    else:
        data = posterior.stack(sample=["chain", "draw"]).drop_vars(["sample", "chain", "draw"])
    stats = getattr(data.azstats, kind)(dim=[sample_dim])
    pit = PIT_FUNCTIONS[kind](data, stats, sample_dims=[sample_dim])

    # PIT values outside the support are drawn at random within 0.5 / n_samples of 0 or 1
    eps = 0.5 / 200
    for var_name, var in pit.data_vars.items():
        assert "sample" in var.dims
        assert_allclose(var.transpose(*expected[var_name].dims), expected[var_name], atol=eps)
