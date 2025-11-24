"""Test expectations functions for PSIS-LOO-CV."""

# pylint: disable=redefined-outer-name, unused-argument
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal

from ..helpers import importorskip

azb = importorskip("arviz_base")

from arviz_stats import loo, loo_expectations, loo_metrics, loo_r2
from arviz_stats.utils import ELPDData


def test_loo_expectations_invalid_kind(centered_eight):
    with pytest.raises(ValueError, match="kind must be either"):
        loo_expectations(centered_eight, kind="invalid")


def test_loo_expectations_quantile_without_probs(centered_eight):
    with pytest.raises(ValueError, match="probs must be provided when kind is 'quantile'"):
        loo_expectations(centered_eight, kind="quantile")


def test_loo_expectations_invalid_var_name(centered_eight):
    with pytest.raises(KeyError):
        loo_expectations(centered_eight, var_name="nonexistent")


def test_loo_expectations_elpddata_without_log_weights(centered_eight):
    np.random.default_rng(42)

    loo_result_no_weights = ELPDData(
        elpd=-30.0,
        se=3.0,
        p=2.0,
        good_k=0.7,
        n_samples=100,
        n_data_points=8,
        warning=False,
        kind="loo",
        scale="log",
        log_weights=None,
    )

    with pytest.raises(ValueError, match="ELPDData object does not contain log_weights"):
        loo_expectations(centered_eight, log_weights=loo_result_no_weights)


@pytest.mark.parametrize(
    "kind, probs, expected_vals",
    [
        ("mean", None, 3.81),
        ("quantile", [0.25, 0.75], [-6.26, 14.44]),
    ],
)
def test_loo_expectations(centered_eight, kind, probs, expected_vals):
    loo_exp_vals, _ = loo_expectations(centered_eight, kind=kind, probs=probs)

    if kind == "quantile":
        assert loo_exp_vals.shape == (2, 8)
    else:
        assert loo_exp_vals.shape == (8,)

    assert_almost_equal(loo_exp_vals.sel({"school": "Choate"}), expected_vals, decimal=2)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("kind", ["mean", "var", "quantile"])
def test_loo_expectations_khat(centered_eight, datatree, kind):
    probs = [0.25, 0.75] if kind == "quantile" else None
    result, khat = loo_expectations(centered_eight, kind=kind, probs=probs)

    assert np.all(np.isfinite(khat.values))
    assert np.all(khat.values >= -0.5) and np.all(khat.values <= 1.5)

    if kind == "quantile":
        expected_dims = tuple(d for d in result.dims if d != "quantile")
        assert khat.dims == expected_dims
        assert khat.shape == tuple(result.sizes[d] for d in expected_dims)
    else:
        assert khat.dims == result.dims
        assert khat.shape == result.shape

    assert len(np.unique(khat.values.flatten())) > 1

    for dim_name in khat.dims:
        khat_coord_vals = khat.coords[dim_name].values
        result_coord_vals = result.coords[dim_name].values
        if khat_coord_vals.dtype.kind in ("U", "S", "O"):
            assert np.array_equal(khat_coord_vals, result_coord_vals)
        else:
            assert_allclose(khat_coord_vals, result_coord_vals)

    _, khat_check = loo_expectations(datatree, var_name="y", kind=kind, probs=probs)
    n_samples = (
        datatree.log_likelihood["y"].sizes["chain"] * datatree.log_likelihood["y"].sizes["draw"]
    )
    good_k = min(1 - 1 / np.log10(n_samples), 0.7) if n_samples > 1 else 0.7
    if np.any(khat_check.values > good_k):
        with pytest.warns(UserWarning, match="Estimated shape parameter of Pareto distribution"):
            loo_expectations(datatree, var_name="y", kind=kind, probs=probs)


def test_log_weights_input_formats(centered_eight):
    loo_result = loo(centered_eight, pointwise=True)
    log_weights_da = loo_result.log_weights

    loo_exp_da, khat_da = loo_expectations(centered_eight, kind="mean", log_weights=log_weights_da)
    loo_exp_elpddata, khat_elpddata = loo_expectations(
        centered_eight, kind="mean", log_weights=loo_result
    )
    assert_array_equal(loo_exp_da.values, loo_exp_elpddata.values)
    assert_array_equal(khat_da.values, khat_elpddata.values)


@pytest.mark.parametrize("kind", ["median", "sd"])
def test_loo_expectations_median_sd(centered_eight, kind):
    result, khat = loo_expectations(centered_eight, kind=kind)

    assert result.shape == (8,)
    assert khat.shape == (8,)
    assert np.all(np.isfinite(result.values))
    assert np.all(np.isfinite(khat.values))


def test_loo_expectations_single_quantile(centered_eight):
    result, khat = loo_expectations(centered_eight, kind="quantile", probs=0.5)

    assert result.shape == (8,)
    assert khat.shape == (8,)
    assert np.all(np.isfinite(result.values))
    assert np.all(np.isfinite(khat.values))


def test_loo_expectations_extreme_probs(centered_eight):
    result, khat = loo_expectations(centered_eight, kind="quantile", probs=[0.01, 0.99])

    assert result.shape == (2, 8)
    assert khat.shape == (8,)
    assert np.all(np.isfinite(result.values))
    assert np.all(np.isfinite(khat.values))


def test_loo_expectations_var_computation(centered_eight):
    result_var, khat_var = loo_expectations(centered_eight, kind="var")
    result_sd, khat_sd = loo_expectations(centered_eight, kind="sd")

    assert np.all(result_var.values >= 0)
    assert_allclose(result_sd.values, np.sqrt(result_var.values), rtol=1e-10)
    assert_array_equal(khat_var.values, khat_sd.values)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_loo_expectations_multidimensional():
    rng = np.random.default_rng(42)

    multi_dim_data = azb.from_dict(
        {
            "posterior": {"mu": rng.normal(size=(2, 50))},
            "posterior_predictive": {"y": rng.normal(size=(2, 50, 3, 4))},
            "log_likelihood": {"y": rng.normal(size=(2, 50, 3, 4))},
            "observed_data": {"y": rng.normal(size=(3, 4))},
        }
    )

    result, khat = loo_expectations(multi_dim_data, kind="mean")

    assert result.shape == (3, 4)
    assert khat.shape == (3, 4)
    assert np.all(np.isfinite(result.values))


def test_loo_expectations_with_explicit_var_name(centered_eight):
    result_explicit, khat_explicit = loo_expectations(centered_eight, var_name="obs", kind="mean")
    result_auto, khat_auto = loo_expectations(centered_eight, kind="mean")

    assert_array_equal(result_explicit.values, result_auto.values)
    assert_array_equal(khat_explicit.values, khat_auto.values)


@pytest.mark.parametrize("kind", ["mae", "mse", "rmse"])
def test_loo_metrics(centered_eight, kind):
    result = loo_metrics(centered_eight, kind=kind)

    assert hasattr(result, "_fields")
    assert hasattr(result, "mean")
    assert hasattr(result, "se")
    assert isinstance(result.mean, int | float | str)
    assert isinstance(result.se, int | float | str)


def test_loo_metrics_with_log_weights(centered_eight):
    loo_result = loo(centered_eight, pointwise=True)

    result_with_weights = loo_metrics(centered_eight, kind="rmse", log_weights=loo_result)
    result_without_weights = loo_metrics(centered_eight, kind="rmse")

    assert hasattr(result_with_weights, "mean")
    assert hasattr(result_without_weights, "mean")


def test_loo_metrics_explicit_var_name(centered_eight):
    result = loo_metrics(centered_eight, var_name="obs", kind="mae")

    assert hasattr(result, "mean")
    assert hasattr(result, "se")


def test_loo_metrics_round_to(centered_eight):
    result_2g = loo_metrics(centered_eight, kind="rmse", round_to=None)
    result_3 = loo_metrics(centered_eight, kind="rmse", round_to=3)

    assert hasattr(result_2g, "mean")
    assert hasattr(result_3, "mean")


def test_loo_r2_summary(datatree_regression):
    result = loo_r2(datatree_regression, var_name="y")
    assert isinstance(result, tuple)
    assert hasattr(result, "_fields")
    assert "mean" in result._fields
    assert "eti_lb" in result._fields
    assert "eti_ub" in result._fields


def test_loo_r2_array(datatree_regression):
    n_sims = 1000
    result = loo_r2(datatree_regression, var_name="y", summary=False, n_simulations=n_sims)
    assert isinstance(result, np.ndarray)
    assert result.shape == (n_sims,)


@pytest.mark.parametrize("point_estimate", ["mean", "median"])
def test_loo_r2_point_estimate(datatree_regression, point_estimate):
    result = loo_r2(datatree_regression, var_name="y", summary=True, point_estimate=point_estimate)
    assert point_estimate in result._fields


@pytest.mark.parametrize("ci_kind", ["hdi", "eti"])
def test_loo_r2_ci_kind(datatree_regression, ci_kind):
    result = loo_r2(datatree_regression, var_name="y", summary=True, ci_kind=ci_kind)
    assert f"{ci_kind}_lb" in result._fields
    assert f"{ci_kind}_ub" in result._fields


@pytest.mark.parametrize("ci_prob", [0.9, 0.95])
def test_loo_r2_ci_prob(datatree_regression, ci_prob):
    result = loo_r2(datatree_regression, var_name="y", summary=True, ci_prob=ci_prob)
    assert hasattr(result, "_fields")


@pytest.mark.parametrize("kind", ["circular_mean", "circular_var", "circular_sd"])
def test_loo_expectations_circular(centered_eight, kind):
    """Simple parametric checks for circular kinds: shape and finiteness of result and khat."""

    result, khat = loo_expectations(centered_eight, kind=kind)

    assert result.shape == (8,)
    assert khat.shape == (8,)
    assert np.all(np.isfinite(result.values))
    assert np.all(np.isfinite(khat.values))
