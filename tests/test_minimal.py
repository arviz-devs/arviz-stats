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