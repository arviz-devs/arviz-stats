"""Test loo_influence function for PSIS-LOO-CV."""

# pylint: disable=redefined-outer-name, unused-argument
import pytest

from ..helpers import importorskip

np = importorskip("numpy")
azb = importorskip("arviz_base")
xr = importorskip("xarray")

from numpy.testing import assert_array_equal

from arviz_stats.loo import loo_influence


def test_loo_influential_invalid_kind(centered_eight):
    with pytest.raises(ValueError, match="kind must be one of"):
        loo_influence(centered_eight, kind="invalid")


def test_loo_influential_invalid_group(centered_eight):
    with pytest.raises(ValueError, match="group must be either"):
        loo_influence(centered_eight, group="invalid_group")


def test_loo_influential_quantile_without_probs(centered_eight):
    with pytest.raises(ValueError, match="probs must be provided when kind is 'quantile'"):
        loo_influence(centered_eight, kind="quantile")


def test_loo_influential_invalid_var_name(centered_eight):
    with pytest.raises(KeyError):
        loo_influence(centered_eight, var_names="nonexistent")


@pytest.mark.parametrize("kind", ["mean", "median", "sd", "var"])
def test_loo_influential_basic(centered_eight, kind):
    shift, khat = loo_influence(centered_eight, kind=kind)

    assert shift["obs"].shape == (8,)
    assert np.all(np.isfinite(shift["obs"].values))
    assert np.all(shift["obs"].values >= 0)

    assert np.all(np.isfinite(khat.values))
    assert khat.shape == (8,)


def test_loo_influential_standardize_mean(centered_eight):
    shift_std, _ = loo_influence(centered_eight, kind="mean", standardize=True)
    shift_raw, _ = loo_influence(centered_eight, kind="mean", standardize=False)

    assert not np.allclose(shift_std["obs"].values, shift_raw["obs"].values)
    assert np.all(shift_std["obs"].values >= 0)
    assert np.all(shift_raw["obs"].values >= 0)


def test_loo_influential_standardize_median(centered_eight):
    shift_std, _ = loo_influence(centered_eight, kind="median", standardize=True)
    shift_raw, _ = loo_influence(centered_eight, kind="median", standardize=False)

    assert not np.allclose(shift_std["obs"].values, shift_raw["obs"].values)
    assert np.all(shift_std["obs"].values >= 0)
    assert np.all(shift_raw["obs"].values >= 0)


def test_loo_influential_standardize_ignored_for_sd(centered_eight):
    """Standardization should be ignored for non-mean/median kinds."""
    shift_std, khat_std = loo_influence(centered_eight, kind="sd", standardize=True)
    shift_raw, khat_raw = loo_influence(centered_eight, kind="sd", standardize=False)

    assert_array_equal(shift_std["obs"].values, shift_raw["obs"].values)
    assert_array_equal(khat_std.values, khat_raw.values)


def test_loo_influential_quantile(centered_eight):
    shift, _ = loo_influence(centered_eight, kind="quantile", probs=[0.25, 0.75])

    assert shift.shape == (8,)
    assert np.all(np.isfinite(shift.values))
    assert np.all(shift.values >= 0)


def test_loo_influential_octiles(centered_eight):
    shift, _ = loo_influence(centered_eight, kind="octiles")

    assert shift.shape == (8,)
    assert np.all(np.isfinite(shift.values))
    assert np.all(shift.values >= 0)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("kind", ["mean", "median", "sd", "var"])
def test_loo_influential_khat(centered_eight, kind):
    _, khat = loo_influence(centered_eight, kind=kind)

    assert np.all(np.isfinite(khat.values))
    assert np.all(khat.values >= -0.5) and np.all(khat.values <= 1.5)
    assert len(np.unique(khat.values.flatten())) > 1


def test_loo_influential_explicit_var_name(centered_eight):
    shift_explicit, khat_explicit = loo_influence(centered_eight, var_names="obs", kind="mean")
    shift_auto, khat_auto = loo_influence(centered_eight, kind="mean")

    assert_array_equal(shift_explicit["obs"].values, shift_auto["obs"].values)
    assert_array_equal(khat_explicit.values, khat_auto.values)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_loo_influential_multidimensional():
    rng = np.random.default_rng(42)

    multi_dim_data = azb.from_dict(
        {
            "posterior": {
                "mu": rng.normal(size=(2, 50)),
                "coef": rng.normal(size=(2, 50, 3, 4, 2)),
            },
            "posterior_predictive": {"y": rng.normal(size=(2, 50, 3, 4))},
            "log_likelihood": {"y": rng.normal(size=(2, 50, 3, 4))},
            "observed_data": {"y": rng.normal(size=(3, 4))},
        },
        dims={"y": ["d1", "d2"], "coef": ["d1", "d2", "coef_dim"]},
    )

    shift, _ = loo_influence(multi_dim_data, kind="mean")
    assert shift["y"].shape == (3, 4)
    assert np.all(np.isfinite(shift["y"].values))
    assert np.all(shift["y"].values >= 0)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_loo_influential_posterior_group(centered_eight):
    shift, _ = loo_influence(centered_eight, group="posterior", var_names="mu", kind="mean")

    assert shift["mu"].shape == (8,)
    assert np.all(np.isfinite(shift["mu"].values))
    assert np.all(shift["mu"].values >= 0)
