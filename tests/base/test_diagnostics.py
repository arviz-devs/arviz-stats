"""Test Diagnostic methods"""

# pylint: disable=redefined-outer-name
import os

import numpy as np
import pandas as pd
import pytest
from arviz_base import load_arviz_data, xarray_var_iter

from arviz_stats.base import array_stats

# For tests only, recommended value should be closer to 1.01-1.05
# See discussion in https://github.com/stan-dev/rstan/pull/618
GOOD_RHAT = 1.1


@pytest.fixture(scope="session")
def idata():
    centered_eight = load_arviz_data("centered_eight")
    return centered_eight.posterior


@pytest.fixture(scope="function")
def data(idata):
    return idata.ds


@pytest.mark.skip(reason="from_cmdstan not yet ported to arviz_base.")
def test_deterministic():
    """
    Test algorithm against posterior (R) convergence functions.

    posterior: https://github.com/stan-dev/posterior
    R code:
    ```
    library("posterior")
    data2 <- read.csv("blocker.2.csv", comment.char = "#")
    data1 <- read.csv("blocker.1.csv", comment.char = "#")
    output <- matrix(ncol=17, nrow=length(names(data1))-4)
    j = 0
    for (i in 1:length(names(data1))) {
        name = names(data1)[i]
        ary = matrix(c(data1[,name], data2[,name]), 1000, 2)
        if (!endsWith(name, "__"))
            j <- j + 1
            output[j,] <- c(
                posterior::rhat(ary),
                posterior::rhat_basic(ary, FALSE),
                posterior::ess_bulk(ary),
                posterior::ess_tail(ary),
                posterior::ess_mean(ary),
                posterior::ess_sd(ary),
                posterior::ess_median(ary),
                posterior::ess_basic(ary, FALSE),
                posterior::ess_quantile(ary, 0.01),
                posterior::ess_quantile(ary, 0.1),
                posterior::ess_quantile(ary, 0.3),
                posterior::mcse_mean(ary),
                posterior::mcse_sd(ary),
                posterior::mcse_median(ary),
                posterior::mcse_quantile(ary, prob=0.01),
                posterior::mcse_quantile(ary, prob=0.1),
                posterior::mcse_quantile(ary, prob=0.3))
    }
    df = data.frame(output, row.names = names(data1)[5:ncol(data1)])
    colnames(df) <- c("rhat_rank",
                        "rhat_raw",
                        "ess_bulk",
                        "ess_tail",
                        "ess_mean",
                        "ess_sd",
                        "ess_median",
                        "ess_raw",
                        "ess_quantile01",
                        "ess_quantile10",
                        "ess_quantile30",
                        "mcse_mean",
                        "mcse_sd",
                        "mcse_median",
                        "mcse_quantile01",
                        "mcse_quantile10",
                        "mcse_quantile30")
    write.csv(df, "reference_posterior.csv")
    ```
    Reference file:

    Created: 2020-08-31
    System: Ubuntu 18.04.5 LTS
    R version 4.0.2 (2020-06-22)
    posterior 0.1.2
    """
    # download input files
    here = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(here, "..", "saved_models")
    # path = os.path.join(data_directory, "stan_diagnostics", "blocker.[0-9].csv")
    posterior = None  # from_cmdstan(path)
    reference_path = os.path.join(data_directory, "stan_diagnostics", "reference_posterior.csv")
    reference = (
        pd.read_csv(reference_path, index_col=0, float_precision="high")
        .sort_index(axis=1)
        .sort_index(axis=0)
    )
    # test arviz functions
    funcs = {
        "rhat_rank": lambda x: array_stats.rhat(x, method="rank"),
        "rhat_raw": lambda x: array_stats.rhat(x, method="identity"),
        "rhat_nested_rank": lambda x: array_stats.rhat_nested(x, method="rank", superchain_ids=[0, 0, 1, 1]),
        "rhat_nested": lambda x: array_stats.rhat_nested(x, method="identity", superchain_ids=[0, 0, 1, 1]),
        "ess_bulk": lambda x: array_stats.ess(x, method="bulk"),
        "ess_tail": lambda x: array_stats.ess(x, method="tail"),
        "ess_mean": lambda x: array_stats.ess(x, method="mean"),
        "ess_sd": lambda x: array_stats.ess(x, method="sd"),
        "ess_median": lambda x: array_stats.ess(x, method="median"),
        "ess_raw": lambda x: array_stats.ess(x, method="identity"),
        "ess_quantile01": lambda x: array_stats.ess(x, method="quantile", prob=0.01),
        "ess_quantile10": lambda x: array_stats.ess(x, method="quantile", prob=0.1),
        "ess_quantile30": lambda x: array_stats.ess(x, method="quantile", prob=0.3),
        "mcse_mean": lambda x: array_stats.mcse(x, method="mean"),
        "mcse_sd": lambda x: array_stats.mcse(x, method="sd"),
        "mcse_median": lambda x: array_stats.mcse(x, method="median"),
        "mcse_quantile01": lambda x: array_stats.mcse(x, method="quantile", prob=0.01),
        "mcse_quantile10": lambda x: array_stats.mcse(x, method="quantile", prob=0.1),
        "mcse_quantile30": lambda x: array_stats.mcse(x, method="quantile", prob=0.3),
    }
    results = {}
    for key, coord_dict, _, vals in xarray_var_iter(posterior.posterior, combined=True):
        if coord_dict:
            key = f"{key}.{list(coord_dict.values())[0] + 1}"
        results[key] = {func_name: func(vals) for func_name, func in funcs.items()}
    arviz_data = pd.DataFrame.from_dict(results).T.sort_index(axis=1).sort_index(axis=0)

    # check column names
    assert set(arviz_data.columns) == set(reference.columns)

    # check parameter names
    assert set(arviz_data.index) == set(reference.index)

    # show print with pytests '-s' tag
    # np.set_printoptions(16)
    # print(abs(reference - arviz_data).max())

    # test absolute accuracy
    assert (abs(reference - arviz_data).values < 1e-8).all(None)


@pytest.mark.parametrize("method", ("rank", "split", "folded", "z_scale", "identity"))
@pytest.mark.parametrize("var_names", (None, "mu", ["mu", "tau"]))
def test_rhat(data, var_names, method):
    """Confirm R-hat statistic is close to 1 for a large
    number of samples. Also checks the correct shape"""
    rhat_data = data.azstats.filter_vars(var_names).rhat(method=method)
    for r_hat in rhat_data.data_vars.values():
        assert ((1 / GOOD_RHAT < r_hat.values) | (r_hat.values < GOOD_RHAT)).all()

    # In None case check that all varnames from rhat_data match input data
    if var_names is None:
        assert list(rhat_data.data_vars) == list(data.data_vars)


@pytest.mark.parametrize("method", ("rank", "split", "folded", "z_scale", "identity"))
def test_rhat_nan(method):
    """Confirm R-hat statistic returns nan."""
    rng = np.random.default_rng()
    data = rng.normal(size=(4, 100))
    data[0, 0] = np.nan  #  pylint: disable=unsupported-assignment-operation
    rhat_data = array_stats.rhat(data, method=method)
    assert np.isnan(rhat_data)


@pytest.mark.parametrize("method", ("rank", "split", "folded", "z_scale", "identity"))
@pytest.mark.parametrize("chain", (None, 1, 2))
@pytest.mark.parametrize("draw", (1, 2, 3, 4))
def test_rhat_shape(method, chain, draw):
    """Confirm R-hat statistic returns nan."""
    rng = np.random.default_rng()
    size = draw if chain is None else (chain, draw)
    ary = rng.normal(size=size)
    kwargs = {"chain_axis": None} if chain is None else {}
    if (chain in (None, 1)) or (draw < 4):
        rhat_data = array_stats.rhat(ary, method=method, **kwargs)
        assert np.isnan(rhat_data)
    else:
        rhat_data = array_stats.rhat(ary, method=method, **kwargs)
        assert not np.isnan(rhat_data)


def test_rhat_bad():
    """Confirm rank normalized Split R-hat statistic is
    far from 1 for a small number of samples."""
    r_hat = array_stats.rhat(np.vstack([20 + np.random.randn(1, 100), np.random.randn(1, 100)]))
    assert 1 / GOOD_RHAT > r_hat or GOOD_RHAT < r_hat


def test_rhat_bad_method():
    rng = np.random.default_rng()
    ary = rng.normal(size=(4, 100))
    with pytest.raises(ValueError):
        array_stats.rhat(ary, method="wrong_method")


@pytest.mark.parametrize(
    "method",
    (
        "bulk",
        "tail",
        "quantile",
        "local",
        "mean",
        "sd",
        "median",
        "mad",
        "z_scale",
        "folded",
        "identity",
    ),
)
@pytest.mark.parametrize("relative", (True, False))
def test_ess_array(method, relative):
    rng = np.random.default_rng()
    ary = rng.normal(size=(4, 100))
    n_low = 0.3 if relative else 150
    n_high = 2 if relative else 700
    if method in ("quantile", "tail"):
        ess_hat = array_stats.ess(ary, method=method, prob=0.34, relative=relative)
        if method == "tail":
            assert ess_hat > n_low
            assert ess_hat < n_high
            ess_hat = array_stats.ess(ary, method=method, relative=relative)
            assert ess_hat > n_low
            assert ess_hat < n_high
            ess_hat = array_stats.ess(ary, method=method, prob=(0.2, 0.8), relative=relative)
    elif method == "local":
        ess_hat = array_stats.ess(ary, method=method, prob=(0.2, 0.3), relative=relative)
    else:
        ess_hat = array_stats.ess(ary, method=method, relative=relative)
    assert ess_hat > n_low
    assert ess_hat < n_high


@pytest.mark.parametrize(
    "method",
    (
        "bulk",
        "tail",
        "quantile",
        "local",
        "mean",
        "sd",
        "median",
        "mad",
        "z_scale",
        "folded",
        "identity",
    ),
)
@pytest.mark.parametrize("relative", (True, False))
@pytest.mark.parametrize("chain", (None, 1, 2))
@pytest.mark.parametrize("draw", (1, 2, 3, 4))
@pytest.mark.parametrize("use_nan", (True, False))
def test_ess_nan(method, relative, chain, draw, use_nan):
    rng = np.random.default_rng()
    size = draw if chain is None else (chain, draw)
    data = rng.normal(size=size)
    kwargs = {"chain_axis": None} if chain is None else {}
    if use_nan:
        data[0] = np.nan
    if method in ("quantile", "tail"):
        ess_value = array_stats.ess(data, method=method, prob=0.34, relative=relative, **kwargs)
    elif method == "local":
        ess_value = array_stats.ess(
            data, method=method, prob=(0.2, 0.3), relative=relative, **kwargs
        )
    else:
        ess_value = array_stats.ess(data, method=method, relative=relative, **kwargs)
    if (draw < 4) or use_nan:
        assert np.isnan(ess_value)
    else:
        assert not np.isnan(ess_value)


def test_ess_missing_prob():
    rng = np.random.default_rng()
    ary = rng.normal(size=(4, 100))
    with pytest.raises(TypeError):
        array_stats.ess(ary, method="quantile")
    with pytest.raises(TypeError):
        array_stats.ess(ary, method="local")


def test_ess_too_many_probs():
    rng = np.random.default_rng()
    ary = rng.normal(size=(4, 100))
    with pytest.raises(ValueError):
        array_stats.ess(ary, method="local", prob=[0.1, 0.2, 0.9])


def test_ess_constant():
    assert array_stats.ess(np.ones((4, 100))) == 400


def test_ess_bad_method():
    rng = np.random.default_rng()
    ary = rng.normal(size=(4, 100))
    with pytest.raises(ValueError):
        array_stats.ess(ary, method="wrong_method")


def test_ess_ndarray_axis():
    rng = np.random.default_rng()
    ary = rng.normal(size=(3, 100, 5))
    ess_data = array_stats.ess(ary, chain_axis=0, draw_axis=1)
    assert ess_data.shape == (5,)
    ess_data = array_stats.ess(ary, chain_axis=-1, draw_axis=1)
    assert ess_data.shape == (3,)


@pytest.mark.parametrize(
    "method",
    (
        "bulk",
        "tail",
        "quantile",
        "local",
        "mean",
        "sd",
        "median",
        "mad",
        "z_scale",
        "folded",
        "identity",
    ),
)
@pytest.mark.parametrize("relative", (True, False))
def test_ess_dataset(data, method, relative):
    n_low = 100 / (data.sizes["chain"] * data.sizes["draw"]) if relative else 100
    if method in ("quantile", "tail"):
        ess_hat = data.azstats.ess(method=method, prob=0.34, relative=relative)
    elif method == "local":
        ess_hat = data.azstats.ess(method=method, prob=(0.2, 0.3), relative=relative)
    else:
        ess_hat = data.azstats.ess(method=method, relative=relative)
    assert np.all(ess_hat.mu.values > n_low)


@pytest.mark.parametrize("mcse_method", ("mean", "sd", "median", "quantile"))
def test_mcse_array(mcse_method):
    rng = np.random.default_rng()
    ary = rng.normal(size=(4, 100))
    if mcse_method == "quantile":
        mcse_hat = array_stats.mcse(ary, method=mcse_method, prob=0.34)
    else:
        mcse_hat = array_stats.mcse(ary, method=mcse_method)
    assert mcse_hat


def test_mcse_ndarray_axis():
    rng = np.random.default_rng()
    ary = rng.normal(size=(3, 100, 5))
    mcse_data = array_stats.mcse(ary, chain_axis=0, draw_axis=1)
    assert mcse_data.shape == (5,)
    mcse_data = array_stats.mcse(ary, chain_axis=-1, draw_axis=1)
    assert mcse_data.shape == (3,)


@pytest.mark.parametrize("mcse_method", ("mean", "sd", "median", "quantile"))
def test_mcse_dataset(data, mcse_method):
    if mcse_method == "quantile":
        mcse_hat = data.azstats.mcse(method=mcse_method, prob=0.34)
    else:
        mcse_hat = data.azstats.mcse(method=mcse_method)
    assert mcse_hat


@pytest.mark.parametrize("mcse_method", ("mean", "sd", "median", "quantile"))
@pytest.mark.parametrize("chain", (None, 1, 2))
@pytest.mark.parametrize("draw", (1, 2, 3, 4))
@pytest.mark.parametrize("use_nan", (True, False))
def test_mcse_nan(mcse_method, chain, draw, use_nan):
    rng = np.random.default_rng()
    size = draw if chain is None else (chain, draw)
    data = rng.normal(size=size)
    kwargs = {"chain_axis": None} if chain is None else {}
    if use_nan:
        data[0] = np.nan
    if mcse_method == "quantile":
        mcse_hat = array_stats.mcse(data, method=mcse_method, prob=0.34, **kwargs)
    else:
        mcse_hat = array_stats.mcse(data, method=mcse_method, **kwargs)
    if draw < 4 or use_nan:
        assert np.isnan(mcse_hat)
    else:
        assert not np.isnan(mcse_hat)


def test_mcse_bad_method():
    rng = np.random.default_rng()
    ary = rng.normal(size=(4, 100))
    with pytest.raises(ValueError):
        array_stats.mcse(ary, method="wrong_method")


@pytest.mark.parametrize("chains", (None, 1, 2, 3))
@pytest.mark.parametrize("draws", (2, 3, 100, 101))
def test_split_chain_dims(chains, draws):
    if chains is None:
        data = np.random.randn(draws)
    else:
        data = np.random.randn(chains, draws)
    split_data = array_stats._split_chains(data)  # pylint: disable=protected-access
    if chains is None:
        chains = 1
    assert split_data.shape == (chains * 2, draws // 2)
