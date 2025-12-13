"""Test minimal functionality."""

# pylint: disable=redefined-outer-name
import numpy as np
import pytest

from arviz_stats.base import array_stats


@pytest.fixture(scope="module")
def data_c0d1():
    rng = np.random.default_rng(19)
    return rng.normal(size=(4, 200, 3))


@pytest.fixture(scope="module")
def data_c2d0():
    rng = np.random.default_rng(19)
    return rng.normal(size=(200, 3, 4))


@pytest.fixture(scope="module")
def data_discrete():
    rng = np.random.default_rng(9)
    return rng.poisson(size=(4, 200, 3))


@pytest.fixture(scope="module")
def data_multimodal():
    rng = np.random.default_rng(9)
    m1 = rng.normal(loc=0, scale=1, size=(4, 2000, 2))
    m2 = rng.normal(loc=-15, scale=1, size=(4, 2000, 2))
    m3 = rng.normal(loc=20, scale=1, size=(4, 3000, 2))
    return np.concatenate((m1, m2, m3), axis=1)


@pytest.fixture(scope="module")
def data_discrete_multimodal():
    rng = np.random.default_rng(9)
    m1 = rng.poisson(lam=5, size=(4, 20_000, 2))
    m2 = rng.poisson(lam=20, size=(4, 20_000, 2))
    m3 = rng.poisson(lam=80, size=(4, 30_000, 2))
    return np.concatenate((m1, m2, m3), axis=1)


@pytest.mark.parametrize("axis", [None, 1, (0, 1)])
def test_hdi_axes(data_c0d1, axis):
    hdi = array_stats.hdi(data_c0d1, 0.8, axis=axis)
    assert hdi.shape[-1] == 2
    assert np.all(-2 < hdi[..., 0])
    assert np.all(hdi[..., 0] < -0.75)
    assert np.all(0.75 < hdi[..., 1])
    assert np.all(hdi[..., 1] < 2)


@pytest.mark.parametrize("axis", [None, 1, (0, 1)])
@pytest.mark.parametrize("method", ["multimodal", "multimodal_sample"])
def test_hdi_multimodal(data_multimodal, method, axis):
    hdi = array_stats.hdi(data_multimodal, 0.8, method=method, axis=axis)
    assert hdi.shape[-1] == 2
    assert hdi.shape[-2] == 3


@pytest.mark.parametrize("axis", [None, 1, (0, 1)])
def test_hdi_multimodal_discrete(data_discrete_multimodal, axis):
    hdi = array_stats.hdi(data_discrete_multimodal, 0.8, method="multimodal", axis=axis)
    assert hdi.shape[-1] == 2
    assert hdi.shape[-2] == 3


@pytest.mark.parametrize("axis", ["01", "20"])
def test_ess_axes(axis, data_c0d1, data_c2d0):
    if axis == "01":
        ess = array_stats.ess(data_c0d1, chain_axis=0, draw_axis=1)
    else:
        ess = array_stats.ess(data_c2d0, chain_axis=2, draw_axis=0)
    assert ess.shape == (3,)
    assert np.all(ess > 700)


@pytest.mark.parametrize(
    "method", ["bulk", "mean", "sd", "median", "mad", "z_scale", "folded", "identity"]
)
def test_ess_methods(method, data_c0d1):
    ess = array_stats.ess(data_c0d1, method=method, chain_axis=0, draw_axis=1)
    assert ess.shape == (3,)
    assert np.all(ess > 700)


@pytest.mark.parametrize("method", ["quantile", "tail"])
def test_ess_quantile_tail(method, data_c0d1):
    ess = array_stats.ess(data_c0d1, method=method, prob=0.9, chain_axis=0, draw_axis=1)
    assert ess.shape == (3,)
    assert np.all(ess > 700)


def test_ess_local(data_c0d1):
    ess = array_stats.ess(data_c0d1, method="local", prob=(0.2, 0.3), chain_axis=0, draw_axis=1)
    assert ess.shape == (3,)
    assert np.all(ess > 700)


def test_ess_no_chain(data_c0d1):
    ess = array_stats.ess(data_c0d1, chain_axis=None, draw_axis=1)
    assert ess.shape == (4, 3)
    assert np.all(ess > 150)


@pytest.mark.parametrize("axis", ["01", "20"])
def test_rhat_axes(axis, data_c0d1, data_c2d0):
    if axis == "01":
        rhat = array_stats.rhat(data_c0d1, chain_axis=0, draw_axis=1)
    else:
        rhat = array_stats.rhat(data_c2d0, chain_axis=2, draw_axis=0)
    assert rhat.shape == (3,)
    assert np.all(rhat < 1.05)


@pytest.mark.parametrize("method", ["rank", "split", "z_scale", "folded", "identity"])
def test_rhat_methods(method, data_c0d1):
    rhat = array_stats.rhat(data_c0d1, method=method, chain_axis=0, draw_axis=1)
    assert rhat.shape == (3,)
    assert np.all(rhat < 1.05)


def test_rhat_no_chain(data_c0d1):
    rhat = array_stats.rhat(data_c0d1, chain_axis=None, draw_axis=1)
    assert rhat.shape == (4, 3)
    assert np.all(np.isnan(rhat))


@pytest.mark.parametrize("axis", ["01", "20"])
def test_rhat_nested_axes(axis, data_c0d1, data_c2d0):
    if axis == "01":
        rhat = array_stats.rhat_nested(data_c0d1, (0, 0, 1, 1), chain_axis=0, draw_axis=1)
    else:
        rhat = array_stats.rhat_nested(data_c2d0, (0, 0, 1, 1), chain_axis=2, draw_axis=0)
    assert rhat.shape == (3,)
    assert np.all(rhat < 1.05)


def test_rhat_nested_no_chain(data_c0d1):
    rhat = array_stats.rhat_nested(data_c0d1, (0,), chain_axis=None, draw_axis=1)
    assert rhat.shape == (4, 3)
    assert np.all(np.isnan(rhat))


@pytest.mark.parametrize("method", ["rank", "split", "z_scale", "folded", "identity"])
def test_rhat_nested_methods(method, data_c0d1):
    rhat = array_stats.rhat_nested(
        data_c0d1, (0, 0, 1, 1), method=method, chain_axis=0, draw_axis=1
    )
    assert rhat.shape == (3,)
    assert np.all(rhat < 1.05)


@pytest.mark.parametrize("axis", ["01", "20"])
def test_mcse_axes(axis, data_c0d1, data_c2d0):
    if axis == "01":
        mcse = array_stats.mcse(data_c0d1, chain_axis=0, draw_axis=1)
    else:
        mcse = array_stats.mcse(data_c2d0, chain_axis=2, draw_axis=0)
    assert mcse.shape == (3,)
    assert np.all(mcse > 0.03)
    assert np.all(mcse < 0.04)


@pytest.mark.parametrize("method", ["mean", "sd", "median", "quantile"])
def test_mcse_methods(method, data_c0d1):
    kwargs = {"prob": 0.9} if method == "quantile" else {}
    mcse = array_stats.mcse(data_c0d1, method=method, chain_axis=0, draw_axis=1, **kwargs)
    assert mcse.shape == (3,)
    assert np.all(mcse > 0.01)
    assert np.all(mcse < 0.1)


def test_mcse_no_chain(data_c0d1):
    mcse = array_stats.mcse(data_c0d1, chain_axis=None, draw_axis=1)
    assert mcse.shape == (4, 3)
    assert np.all(mcse > 0.05)
    assert np.all(mcse < 0.09)


def test_bfmi_datatree_returns_datatree():
    arr0 = np.random.normal(size=(4, 1000))
    arr1 = np.cumsum(arr0, axis=1)
    result0 = array_stats.bfmi(arr0)
    assert np.all(result0 > 1)
    assert np.all(array_stats.bfmi(arr1) < 0.3)
    assert result0.shape == (4,)


@pytest.mark.parametrize("axis", ["01", "20"])
def test_pareto_min_ss_axes(axis, data_c0d1, data_c2d0):
    if axis == "01":
        pareto_min_ss = array_stats.pareto_min_ss(data_c0d1, chain_axis=0, draw_axis=1)
    else:
        pareto_min_ss = array_stats.pareto_min_ss(data_c2d0, chain_axis=2, draw_axis=0)
    assert pareto_min_ss.shape == (3,)
    assert np.all(pareto_min_ss > 9.9)
    assert np.all(pareto_min_ss < 13)


def test_pareto_min_ss_no_chain(data_c0d1):
    pareto_min_ss = array_stats.pareto_min_ss(data_c0d1, chain_axis=None, draw_axis=1)
    assert pareto_min_ss.shape == (4, 3)
    assert np.all(pareto_min_ss > 9.9)
    assert np.all(pareto_min_ss < 15)


def test_thin(data_c0d1):
    thinned = array_stats.thin(data_c0d1, chain_axis=0, draw_axis=1)
    assert thinned.shape == (4, 100, 3)
    thinned = array_stats.thin(data_c0d1, chain_axis=0, draw_axis=1, factor=10)
    assert thinned.shape == (4, 20, 3)


def test_pareto_khat():
    norm = np.random.normal(size=(4, 1000, 200))
    np.testing.assert_almost_equal(array_stats.pareto_khat(norm, tail="both").mean(), 0, decimal=0)
    t1 = np.random.standard_t(df=1, size=(4, 500, 200))
    np.testing.assert_almost_equal(array_stats.pareto_khat(t1, tail="both").mean(), 1, decimal=0)
    t2 = np.random.standard_t(df=2, size=(4, 500, 200))
    np.testing.assert_almost_equal(array_stats.pareto_khat(t2, tail="both").mean(), 0.5, decimal=1)


@pytest.mark.parametrize("tail", ["both", "left", "right"])
def test_pareto_khat_tails(data_c0d1, tail):
    khat = array_stats.pareto_khat(data_c0d1, tail=tail, chain_axis=0, draw_axis=1)
    assert khat.shape == (3,)
    assert np.all(khat < 0.7)


def test_psislw(data_c0d1):
    log_weights, khat = array_stats.psislw(data_c0d1, axis=(0, 1))
    assert log_weights.shape == (3, 4, 200)
    assert khat.shape == (3,)


@pytest.mark.parametrize("r_eff", [1.0, 0.5, 1.5])
def test_psislw_r_eff(data_c0d1, r_eff):
    log_weights, khat = array_stats.psislw(data_c0d1, r_eff=r_eff, axis=(0, 1))
    assert log_weights.shape == (3, 4, 200)
    assert khat.shape == (3,)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 0.9])
def test_power_scale_lw(alpha):
    rng = np.random.default_rng(42)
    log_lik = rng.uniform(-10, -1, size=(4, 200, 3))
    scaled_lw = array_stats.power_scale_lw(log_lik, alpha=alpha, axis=(0, 1))
    assert scaled_lw.shape == (3, 4, 200)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_power_scale_sense():
    rng = np.random.default_rng(42)
    log_lik = rng.uniform(-10, -1, size=(4, 200, 3))
    data = rng.normal(size=(4, 200, 3))
    lower_w = array_stats.power_scale_lw(log_lik, alpha=0.8, axis=-1)
    upper_w = array_stats.power_scale_lw(log_lik, alpha=0.9, axis=-1)
    sense = array_stats.power_scale_sense(
        data, lower_w, upper_w, lower_alpha=0.8, upper_alpha=0.9, chain_axis=0, draw_axis=1
    )
    assert sense.shape == (3,)


@pytest.mark.parametrize("relative", [False, True])
def test_compute_ranks(data_c0d1, relative):
    ranks = array_stats.compute_ranks(data_c0d1, axis=(0, 1), relative=relative)
    assert ranks.shape == (3, 4, 200)
    if relative:
        assert np.all(ranks >= 0) and np.all(ranks <= 1)


@pytest.mark.parametrize("bins", ["arviz", "auto", 10])
def test_get_bins(data_c0d1, bins):
    bin_edges = array_stats.get_bins(data_c0d1, axis=(0, 1), bins=bins)
    assert bin_edges.shape[0] == 3
    assert bin_edges.shape[1] > 1


@pytest.mark.parametrize("bins", [10, 20])
def test_histogram(data_c0d1, bins):
    hist, edges = array_stats.histogram(data_c0d1, bins=bins, axis=(0, 1))
    assert hist.shape[0] == 3
    assert edges.shape[0] == 3
    assert edges.shape[1] == hist.shape[1] + 1


def test_histogram_density(data_c0d1):
    hist, edges = array_stats.histogram(data_c0d1, bins=20, density=True, axis=(0, 1))
    assert hist.shape[0] == 3
    widths = np.diff(edges, axis=1)
    areas = (hist * widths).sum(axis=1)
    np.testing.assert_allclose(areas, 1.0, rtol=0.01)


@pytest.mark.parametrize("circular", [False, True])
def test_kde(data_c0d1, circular):
    grid, pdf, bw = array_stats.kde(data_c0d1, axis=(0, 1), circular=circular, grid_len=100)
    assert grid.shape == (3, 100)
    assert pdf.shape == (3, 100)
    assert bw.shape == (3,)


def test_qds(data_c0d1):
    x, y, marker_size = array_stats.qds(data_c0d1, axis=(0, 1), nquantiles=50)
    assert x.shape == (3, 50)
    assert y.shape == (3, 50)
    assert marker_size.shape == (3,)


@pytest.mark.parametrize("pit", [False, True])
def test_ecdf(data_c0d1, pit):
    x, y = array_stats.ecdf(data_c0d1, axis=(0, 1), npoints=100, pit=pit)
    assert x.shape == (3, 100)
    assert y.shape == (3, 100)
    if not pit:
        assert np.all(y[:, 0] >= 0)
        assert np.all(y[:, -1] <= 1)


def test_residual_r2():
    rng = np.random.default_rng(42)
    y_true = rng.normal(size=100)
    y_pred = rng.normal(size=(4, 100))
    r2 = array_stats.residual_r2(y_true, y_pred)
    assert r2.shape == (4,)


def test_thin_factor(data_c0d1):
    factor = array_stats.thin_factor(data_c0d1, chain_axis=0, draw_axis=1)
    assert isinstance(factor, int)
    assert factor >= 1


@pytest.mark.parametrize("reduce_func", ["mean", "min"])
def test_thin_factor_reduce(data_c0d1, reduce_func):
    factor = array_stats.thin_factor(data_c0d1, reduce_func=reduce_func, chain_axis=0, draw_axis=1)
    assert isinstance(factor, int)
    assert factor >= 1


def test_thin_factor_target_ess(data_c0d1):
    factor = array_stats.thin_factor(data_c0d1, target_ess=100, chain_axis=0, draw_axis=1)
    assert isinstance(factor, int)
    assert factor >= 1


def test_mode(data_discrete):
    mode = array_stats.mode(data_discrete, axis=(0, 1))
    assert mode.shape == (3,)


def test_mode_multimodal(data_discrete_multimodal):
    mode = array_stats.mode(data_discrete_multimodal, axis=(0, 1))
    assert mode.shape == (2,)


@pytest.mark.parametrize("factor", [2, 5])
def test_thin_factor_values(data_c0d1, factor):
    thinned = array_stats.thin(data_c0d1, factor=factor, chain_axis=0, draw_axis=1)
    assert thinned.shape == (4, 200 // factor, 3)


def test_ess_invalid_method(data_c0d1):
    with pytest.raises(ValueError, match="must be one of"):
        array_stats.ess(data_c0d1, method="invalid", chain_axis=0, draw_axis=1)


def test_rhat_invalid_method(data_c0d1):
    with pytest.raises(ValueError, match="must be one of"):
        array_stats.rhat(data_c0d1, method="invalid", chain_axis=0, draw_axis=1)


def test_mcse_invalid_method(data_c0d1):
    with pytest.raises(ValueError, match="must be one of"):
        array_stats.mcse(data_c0d1, method="invalid", chain_axis=0, draw_axis=1)


def test_hdi_invalid_prob(data_c0d1):
    with pytest.raises(ValueError, match="must be in the"):
        array_stats.hdi(data_c0d1, prob=1.5)


def test_thin_invalid_factor(data_c0d1):
    with pytest.raises(ValueError, match="must be greater than 1"):
        array_stats.thin(data_c0d1, factor=0, chain_axis=0, draw_axis=1)
