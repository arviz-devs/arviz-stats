"""Test posterior predictive check diagnostics."""

# pylint: disable=redefined-outer-name
import numpy as np
import pytest
from scipy.stats import norm

from .helpers import importorskip

azb = importorskip("arviz_base")
xr = importorskip("xarray")

from arviz_stats.ppc import text_loo_pit, text_ppc_pit


def _ppc_data(offset=0.0):
    """Build synthetic posterior predictive and observed data."""
    rng = np.random.default_rng(371)
    predictive = {
        "y": rng.normal(size=(2, 500, 100)),
        "z": rng.normal(size=(2, 500, 100)),
    }
    observed = {"y": rng.normal(size=100) + offset, "z": rng.normal(size=100) + offset}
    posterior = rng.normal(0, 1, size=(2, 500, 100))
    log_likelihood = norm(posterior, 1).logpdf(observed["y"])

    var_names = ["y", "z"]
    return azb.from_dict(
        {
            "posterior": {var: posterior for var in var_names},
            "posterior_predictive": {var: predictive[var] for var in var_names},
            "prior_predictive": {var: predictive[var] for var in var_names},
            "log_likelihood": {var: log_likelihood for var in var_names},
            "observed_data": {var: observed[var] for var in var_names},
        },
        dims={var: ["obs"] for var in var_names},
        coords={"obs": np.arange(100)},
    )


@pytest.fixture
def calibrated_ppc():
    return _ppc_data()


@pytest.fixture
def miscalibrated_ppc():
    return _ppc_data(offset=10.0)


def test_text_ppc_pit_calibrated(calibrated_ppc, capsys):
    diagnostics = text_ppc_pit(calibrated_ppc, return_diagnostics=True)
    output = capsys.readouterr().out

    assert "p-value: 0.1196" in output
    assert "alpha: 0.01" in output
    assert "0 of 100 observations flagged" in output
    assert "No deviation from uniformity detected" in output

    assert diagnostics["y"]["has_errors"] is False
    assert diagnostics["z"]["has_errors"] is False
    assert np.isclose(diagnostics["y"]["p_value"], 0.1196, atol=1e-4)
    assert np.isclose(diagnostics["y"]["alpha"], 0.01, atol=1e-2)
    assert diagnostics["y"]["n_obs"] == 100


def test_text_ppc_pit_miscalibrated(miscalibrated_ppc, capsys):
    diagnostics = text_ppc_pit(miscalibrated_ppc, return_diagnostics=True)
    output = capsys.readouterr().out

    assert "p-value: 0" in output
    assert "alpha: 0.01" in output
    assert "97 of 100 observations flagged" in output
    assert "Deviation from uniformity detected" in output

    assert diagnostics["y"]["has_errors"] is True
    assert diagnostics["z"]["has_errors"] is True
    assert np.isclose(diagnostics["y"]["p_value"], 0, atol=1e-4)
    assert np.isclose(diagnostics["y"]["alpha"], 0.01, atol=1e-2)
    assert diagnostics["y"]["n_obs"] == 100


def test_text_loo_pit_calibrated(calibrated_ppc, capsys):
    diagnostics = text_loo_pit(calibrated_ppc, return_diagnostics=True)
    output = capsys.readouterr().out

    assert "p-value: 0" in output
    assert "alpha: 0.01" in output
    assert "0 of 100 observations flagged" in output
    assert "No deviation from uniformity detected" in output

    assert diagnostics["y"]["has_errors"] is False
    assert np.isclose(diagnostics["y"]["p_value"], 0.02768, atol=1e-4)
    assert np.isclose(diagnostics["y"]["alpha"], 0.01, atol=1e-2)
    assert diagnostics["y"]["n_obs"] == 100


def test_warn_if_binary():
    from arviz_stats.ppc import warn_if_binary

    binary_data = xr.Dataset({"y": (["obs"], np.array([0, 1, 0, 1]))})
    with pytest.warns(
        UserWarning,
        match="Prior_posterior_predictive_checks.html#posterior-predictive-checks-for-binary-data",
    ):
        warn_if_binary(binary_data, None)
