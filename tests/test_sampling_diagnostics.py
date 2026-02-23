"""Test sampling diagnostics."""

# pylint: disable=redefined-outer-name
import numpy as np
import pytest

from .helpers import importorskip

azb = importorskip("arviz_base")
xr = importorskip("xarray")

from arviz_stats.sampling_diagnostics import bfmi, diagnose, ess, mcse, rhat, rhat_nested


def test_ess_datatree_returns_datatree(centered_eight):
    result = ess(centered_eight, group="posterior")
    assert isinstance(result, xr.DataTree)
    assert result.name == "posterior"


def test_ess_datatree_with_var_names(centered_eight):
    result = ess(centered_eight, group="posterior", var_names="mu")
    ds = result.dataset
    assert "mu" in ds.data_vars
    assert "theta" not in ds.data_vars


def test_ess_dataarray_returns_dataarray(centered_eight):
    da = centered_eight.posterior["mu"]
    result = ess(da)
    assert isinstance(result, xr.DataArray)


def test_ess_dataset_returns_dataset(centered_eight):
    ds = centered_eight.posterior.dataset
    result = ess(ds, var_names="mu")
    assert isinstance(result, xr.Dataset)
    assert "mu" in result.data_vars


def test_ess_list_input():
    data = [[1, 2, 3, 4], [5, 6, 7, 8]]
    result = ess(data)
    assert isinstance(result, float | np.floating | np.ndarray)


def test_ess_tuple_input():
    data = ([1, 2, 3, 4], [5, 6, 7, 8])
    result = ess(data)
    assert isinstance(result, float | np.floating | np.ndarray)


def test_rhat_datatree_returns_datatree(centered_eight):
    result = rhat(centered_eight, group="posterior")
    assert isinstance(result, xr.DataTree)
    assert result.name == "posterior"


def test_rhat_datatree_with_var_names(centered_eight):
    result = rhat(centered_eight, group="posterior", var_names="mu")
    ds = result.dataset
    assert "mu" in ds.data_vars
    assert "theta" not in ds.data_vars


def test_rhat_dataarray_returns_dataarray(centered_eight):
    da = centered_eight.posterior["mu"]
    result = rhat(da)
    assert isinstance(result, xr.DataArray)


def test_rhat_dataset_returns_dataset(centered_eight):
    ds = centered_eight.posterior.dataset
    result = rhat(ds, var_names="mu")
    assert isinstance(result, xr.Dataset)
    assert "mu" in result.data_vars


def test_rhat_list_input():
    data = [[1, 2, 3, 4], [5, 6, 7, 8]]
    result = rhat(data)
    assert isinstance(result, float | np.floating | np.ndarray)


def test_rhat_nested_datatree_returns_datatree(centered_eight):
    superchain_ids = [0, 0, 1, 1]
    result = rhat_nested(
        centered_eight, group="posterior", var_names="mu", superchain_ids=superchain_ids
    )
    assert isinstance(result, xr.DataTree)
    ds = result.dataset
    assert "mu" in ds.data_vars


def test_rhat_nested_datatree_with_var_names(centered_eight):
    superchain_ids = [0, 0, 1, 1]
    result = rhat_nested(
        centered_eight, group="posterior", var_names="mu", superchain_ids=superchain_ids
    )
    ds = result.dataset
    assert "mu" in ds.data_vars
    assert "theta" not in ds.data_vars


def test_rhat_nested_dataarray_returns_dataarray(centered_eight):
    da = centered_eight.posterior["mu"]
    superchain_ids = [0, 0, 1, 1]
    result = rhat_nested(da, superchain_ids=superchain_ids)
    assert isinstance(result, xr.DataArray)


def test_mcse_datatree_returns_datatree(centered_eight):
    result = mcse(centered_eight, group="posterior")
    assert isinstance(result, xr.DataTree)
    assert result.name == "posterior"


def test_mcse_datatree_with_var_names(centered_eight):
    result = mcse(centered_eight, group="posterior", var_names="mu")
    ds = result.dataset
    assert "mu" in ds.data_vars
    assert "theta" not in ds.data_vars


def test_mcse_dataarray_returns_dataarray(centered_eight):
    da = centered_eight.posterior["mu"]
    result = mcse(da)
    assert isinstance(result, xr.DataArray)


def test_mcse_dataset_returns_dataset(centered_eight):
    ds = centered_eight.posterior.dataset
    result = mcse(ds, var_names="mu")
    assert isinstance(result, xr.Dataset)
    assert "mu" in result.data_vars


def test_mcse_list_input():
    data = [[1, 2, 3, 4], [5, 6, 7, 8]]
    result = mcse(data)
    assert isinstance(result, float | np.floating | np.ndarray)


def test_bfmi_datatree_returns_datatree(centered_eight):
    result = bfmi(centered_eight)
    assert isinstance(result, xr.DataTree)
    assert result.name == "sample_stats"
    assert "energy" in result.dataset.data_vars


def test_bfmi_dataarray_returns_dataarray(centered_eight):
    da = centered_eight.sample_stats["energy"]
    result = bfmi(da)
    assert isinstance(result, xr.DataArray)


def test_bfmi_dataset_returns_dataset(centered_eight):
    ds = centered_eight.sample_stats.dataset
    result = bfmi(ds, var_names="energy")
    assert isinstance(result, xr.Dataset)
    assert "energy" in result.data_vars


def test_diagnose_stdout(centered_eight, capsys):
    diagnose(centered_eight, show_diagnostics=True)
    output = capsys.readouterr().out
    for keyword in ["ESS", "R-hat", "Divergences", "Treedepth", "E-BFMI"]:
        assert keyword in output

    diagnose(centered_eight, show_diagnostics=False)
    assert capsys.readouterr().out == ""


def test_diagnose_diagnostics_dict(centered_eight):
    has_errors, diagnostics = diagnose(
        centered_eight, show_diagnostics=False, return_diagnostics=True
    )

    assert isinstance(has_errors, bool)
    assert isinstance(diagnostics, dict)
    assert "ess" in diagnostics
    assert "rhat" in diagnostics
    assert "divergent" in diagnostics
    assert "treedepth" in diagnostics
    assert "bfmi" in diagnostics


def test_diagnose_var_names_flags(centered_eight):
    _, diagnostics = diagnose(
        centered_eight,
        var_names=["mu"],
        ess_min_ratio=2.0,
        rhat_max=0.0,
        bfmi_threshold=2.0,
        show_diagnostics=False,
        return_diagnostics=True,
    )

    assert diagnostics["ess"]["bad_params"] == ["mu"]
    assert diagnostics["rhat"]["bad_params"] == ["mu"]
    assert diagnostics["bfmi"]["failed_chains"] == [0, 1, 2, 3]
    assert diagnostics["divergent"]["n_divergent"] == 19
    assert diagnostics["treedepth"]["n_max"] == 0


def test_diagnose_no_sample_stats(centered_eight):
    data_no_sample_stats = azb.from_dict({"posterior": {"mu": centered_eight.posterior["mu"]}})

    _, diagnostics = diagnose(
        data_no_sample_stats,
        var_names=["mu"],
        show_diagnostics=False,
        return_diagnostics=True,
    )
    assert "divergent" not in diagnostics
    assert "treedepth" not in diagnostics
    assert "bfmi" not in diagnostics
    assert "ess" in diagnostics
    assert "rhat" in diagnostics


def test_diagnose_no_chains(centered_eight):
    data_no_chains = xr.DataTree.from_dict(
        {
            "posterior": centered_eight["posterior"].ds.rename({"chain": "not_a_chain"}),
            "sample_stats": centered_eight["sample_stats"].ds.rename({"chain": "not_a_chain"}),
        }
    )

    with pytest.warns(
        UserWarning,
        match="Chain dimension not found in data. Skipping ESS threshold check.",
    ):
        diagnose(
            data_no_chains,
            var_names=["mu"],
            sample_dims=["not_a_chain", "draw"],
            show_diagnostics=False,
            return_diagnostics=True,
        )
