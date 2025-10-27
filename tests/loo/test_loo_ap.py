"""Test PSIS-LOO-CV for approximate posteriors."""

# pylint: disable=redefined-outer-name
import numpy as np
import pytest

from ..helpers import importorskip

azb = importorskip("arviz_base")
xr = importorskip("xarray")

from arviz_stats import loo, loo_approximate_posterior


def test_loo_approx_invalid_var_name(centered_eight, log_densities):
    log_p, log_q = log_densities["dataarray"]

    with pytest.raises(TypeError, match="No log likelihood data named nonexistent found"):
        loo_approximate_posterior(centered_eight, log_p=log_p, log_q=log_q, var_name="nonexistent")


@pytest.mark.parametrize("input_type", ["dataarray", "numpy"])
def test_loo_approx_basic(centered_eight, log_densities, input_type):
    log_p, log_q = log_densities[input_type]
    log_lik = log_densities["log_lik"]

    n_samples = log_lik.chain.size * log_lik.draw.size
    n_data_points = np.prod(
        [log_lik[dim].size for dim in log_lik.dims if dim not in ["chain", "draw"]]
    )

    result = loo_approximate_posterior(centered_eight, log_p=log_p, log_q=log_q, var_name="obs")

    assert result.kind == "loo"
    assert result.n_samples == n_samples
    assert result.n_data_points == n_data_points
    assert isinstance(result.elpd, float)
    assert isinstance(result.se, float)
    assert isinstance(result.p, float)
    assert isinstance(result.warning, bool)


@pytest.mark.parametrize("pointwise", [True, False])
def test_loo_approx_pointwise(centered_eight, log_densities, pointwise):
    log_p, log_q = log_densities["dataarray"]
    log_lik = log_densities["log_lik"]
    n_data_points = np.prod(
        [log_lik[dim].size for dim in log_lik.dims if dim not in ["chain", "draw"]]
    )

    result = loo_approximate_posterior(
        centered_eight,
        log_p=log_p,
        log_q=log_q,
        pointwise=pointwise,
        var_name="obs",
    )

    if pointwise:
        assert result.elpd_i is not None
        assert result.pareto_k is not None
        assert result.elpd_i.shape == (n_data_points,)
        assert result.pareto_k.shape == (n_data_points,)
    else:
        assert result.elpd_i is None
        assert result.pareto_k is None


@pytest.mark.parametrize(
    "error_case,error_type,error_match",
    [
        ("wrong_type", TypeError, None),
        ("length_mismatch", ValueError, "Size of log_p"),
        ("missing_dims", ValueError, "must have dimension 'chain'"),
        ("dim_size_mismatch", ValueError, r"Size of dimension 'chain' in log_p"),
    ],
)
def test_loo_approx_errors(centered_eight, log_densities, error_case, error_type, error_match, rng):
    log_p_da, log_q_da = log_densities["dataarray"]
    log_p_np, log_q_np = log_densities["numpy"]
    log_lik = log_densities["log_lik"]

    kwargs = {}

    if error_case == "wrong_type":
        kwargs = {"log_p": list(log_p_np), "log_q": log_q_np}

    elif error_case == "length_mismatch":
        kwargs = {"log_p": rng.normal(size=log_p_np.size - 1), "log_q": log_q_np}

    elif error_case == "missing_dims":
        broken_p = xr.DataArray(log_p_da.values.reshape(-1), dims=["sample"])
        kwargs = {"log_p": broken_p, "log_q": log_q_da}

    elif error_case == "dim_size_mismatch":
        mismatched_p_values = rng.normal(size=(log_lik.chain.size - 1, log_lik.draw.size))
        mismatched_p = xr.DataArray(
            mismatched_p_values,
            dims=["chain", "draw"],
            coords={
                "chain": log_lik.chain[:-1],
                "draw": log_lik.draw,
            },
        )
        kwargs = {"log_p": mismatched_p, "log_q": log_q_da}

    with pytest.raises(error_type, match=error_match):
        loo_approximate_posterior(centered_eight, var_name="obs", **kwargs)


def test_loo_approx_approx_posterior_attribute(centered_eight, log_densities):
    log_p, log_q = log_densities["dataarray"]

    result = loo_approximate_posterior(centered_eight, log_p=log_p, log_q=log_q, var_name="obs")

    assert hasattr(result, "approx_posterior")
    assert result.approx_posterior is True


def test_loo_approx_vs_standard_loo(centered_eight, log_densities):
    log_p, log_q = log_densities["dataarray"]

    result_approx = loo_approximate_posterior(
        centered_eight, log_p=log_p, log_q=log_q, var_name="obs"
    )
    result_standard = loo(centered_eight, var_name="obs")

    assert result_approx.kind == result_standard.kind
    assert result_approx.n_samples == result_standard.n_samples
    assert result_approx.n_data_points == result_standard.n_data_points


def test_loo_approx_with_log_jacobian(centered_eight, log_densities):
    log_p, log_q = log_densities["dataarray"]
    log_lik = log_densities["log_lik"]

    log_jacobian = xr.DataArray(
        np.zeros(log_lik.sizes["school"]),
        dims=["school"],
        coords={"school": log_lik.coords["school"]},
    )

    result = loo_approximate_posterior(
        centered_eight, log_p=log_p, log_q=log_q, var_name="obs", log_jacobian=log_jacobian
    )

    assert result.kind == "loo"
    assert isinstance(result.elpd, float)
    assert isinstance(result.se, float)


def test_loo_approx_with_nonzero_log_jacobian(centered_eight, log_densities):
    log_p, log_q = log_densities["dataarray"]
    log_lik = log_densities["log_lik"]

    rng = np.random.default_rng(42)
    log_jacobian = xr.DataArray(
        rng.normal(size=log_lik.sizes["school"]),
        dims=["school"],
        coords={"school": log_lik.coords["school"]},
    )

    result_with_jac = loo_approximate_posterior(
        centered_eight, log_p=log_p, log_q=log_q, var_name="obs", log_jacobian=log_jacobian
    )
    result_without_jac = loo_approximate_posterior(
        centered_eight, log_p=log_p, log_q=log_q, var_name="obs"
    )

    assert result_with_jac.elpd != result_without_jac.elpd


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_loo_approx_multidimensional():
    rng = np.random.default_rng(42)

    multi_dim_data = azb.from_dict(
        {
            "posterior": {"mu": rng.normal(size=(2, 50))},
            "log_likelihood": {"y": rng.normal(size=(2, 50, 3, 4))},
        }
    )

    log_p = xr.DataArray(
        rng.normal(size=(2, 50)),
        dims=["chain", "draw"],
        coords={"chain": [0, 1], "draw": range(50)},
    )
    log_q = xr.DataArray(
        rng.normal(size=(2, 50)),
        dims=["chain", "draw"],
        coords={"chain": [0, 1], "draw": range(50)},
    )

    result = loo_approximate_posterior(multi_dim_data, log_p=log_p, log_q=log_q, var_name="y")

    assert result.kind == "loo"
    assert result.n_data_points == 12
    assert isinstance(result.elpd, float)


def test_loo_approx_pointwise_values_shape(centered_eight, log_densities):
    log_p, log_q = log_densities["dataarray"]

    result = loo_approximate_posterior(
        centered_eight, log_p=log_p, log_q=log_q, var_name="obs", pointwise=True
    )

    assert result.elpd_i.dims == ("school",)
    assert result.pareto_k.dims == ("school",)
    assert np.all(np.isfinite(result.elpd_i.values))
    assert np.all(np.isfinite(result.pareto_k.values))


def test_loo_approx_good_k_threshold(centered_eight, log_densities):
    log_p, log_q = log_densities["dataarray"]

    result = loo_approximate_posterior(centered_eight, log_p=log_p, log_q=log_q, var_name="obs")

    assert hasattr(result, "good_k")
    assert isinstance(result.good_k, float)
    assert 0 < result.good_k <= 1


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_loo_approx_identical_log_p_log_q(centered_eight, log_densities):
    log_p, _ = log_densities["dataarray"]
    log_q = log_p.copy()

    result = loo_approximate_posterior(centered_eight, log_p=log_p, log_q=log_q, var_name="obs")

    assert isinstance(result.elpd, float)
    assert np.isfinite(result.elpd)
