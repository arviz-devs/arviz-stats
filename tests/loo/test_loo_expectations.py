# pylint: disable=redefined-outer-name, unused-import, unused-argument
# ruff: noqa: F811, F401
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal

from ..helpers import importorskip

azb = importorskip("arviz_base")

from arviz_stats import loo, loo_expectations


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
