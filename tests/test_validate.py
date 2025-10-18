# pylint: disable=redefined-outer-name, unused-import
# ruff: noqa: F811
import pytest

from .helpers import importorskip

azb = importorskip("arviz_base")

from arviz_stats.validate import (
    validate_ci_prob,
    validate_dims,
    validate_dims_chain_draw_axis,
    validate_prob,
)


def test_validate_dims_none():
    result = validate_dims(None)
    expected = azb.rcParams["data.sample_dims"]
    assert result == list(expected)


def test_validate_dims_string():
    result = validate_dims("chain")
    assert result == ["chain"]


def test_validate_dims_list():
    result = validate_dims(["chain", "draw"])
    assert result == ["chain", "draw"]


def test_validate_dims_tuple():
    result = validate_dims(("chain", "draw"))
    assert result == ["chain", "draw"]


def test_validate_dims_single_element():
    result = validate_dims(["sample"])
    assert result == ["sample"]


def test_validate_dims_chain_draw_axis_two_dims():
    dims, chain_axis, draw_axis = validate_dims_chain_draw_axis(["chain", "draw"])
    assert dims == ["chain", "draw"]
    assert chain_axis == -2
    assert draw_axis == -1


def test_validate_dims_chain_draw_axis_one_dim():
    dims, chain_axis, draw_axis = validate_dims_chain_draw_axis(["draw"])
    assert dims == ["draw"]
    assert chain_axis is None
    assert draw_axis == -1


def test_validate_dims_chain_draw_axis_none():
    dims, chain_axis, draw_axis = validate_dims_chain_draw_axis(None)
    expected = azb.rcParams["data.sample_dims"]
    assert dims == list(expected)
    if len(dims) == 2:
        assert chain_axis == -2
        assert draw_axis == -1
    else:
        assert chain_axis is None
        assert draw_axis == -1


def test_validate_dims_chain_draw_axis_string():
    dims, chain_axis, draw_axis = validate_dims_chain_draw_axis("draw")
    assert dims == ["draw"]
    assert chain_axis is None
    assert draw_axis == -1


def test_validate_dims_chain_draw_axis_invalid_length():
    with pytest.raises(ValueError, match="dims can only have 1 or 2 elements"):
        validate_dims_chain_draw_axis(["chain", "draw", "extra"])


def test_validate_ci_prob_none():
    result = validate_ci_prob(None)
    expected = azb.rcParams["stats.ci_prob"]
    assert result == expected


def test_validate_ci_prob_valid():
    result = validate_ci_prob(0.95)
    assert result == 0.95


def test_validate_ci_prob_boundary_upper():
    result = validate_ci_prob(1.0)
    assert result == 1.0


def test_validate_ci_prob_boundary_lower():
    result = validate_ci_prob(0.01)
    assert result == 0.01


def test_validate_prob_valid():
    result = validate_prob(0.5)
    assert result == 0.5


def test_validate_prob_boundary_upper():
    result = validate_prob(1.0)
    assert result == 1.0


def test_validate_prob_boundary_lower():
    result = validate_prob(0.001)
    assert result == 0.001


def test_validate_prob_invalid_too_high():
    with pytest.raises(ValueError, match="should be in the interval"):
        validate_prob(1.1)


def test_validate_prob_invalid_negative():
    with pytest.raises(ValueError, match="should be in the interval"):
        validate_prob(-0.1)


def test_validate_prob_invalid_zero():
    with pytest.raises(ValueError, match="should be in the interval"):
        validate_prob(0.0)


def test_validate_prob_allow_zero_true():
    result = validate_prob(0.0, allow_0=True)
    assert result == 0.0


def test_validate_prob_allow_zero_valid():
    result = validate_prob(0.5, allow_0=True)
    assert result == 0.5


def test_validate_prob_allow_zero_boundary():
    result = validate_prob(1.0, allow_0=True)
    assert result == 1.0


def test_validate_prob_allow_zero_invalid():
    with pytest.raises(ValueError, match="should be in the interval"):
        validate_prob(-0.1, allow_0=True)


def test_validate_ci_prob_invalid():
    with pytest.raises(ValueError, match="should be in the interval"):
        validate_ci_prob(1.5)
