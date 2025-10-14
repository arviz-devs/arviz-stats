# pylint: disable=redefined-outer-name, no-self-use, unused-argument
# ruff: noqa: F811
from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from ..helpers import importorskip

azb = importorskip("arviz_base")
xr = importorskip("xarray")

from arviz_stats import loo, reloo
from arviz_stats.loo.wrapper import SamplingWrapper
from arviz_stats.utils import ELPDData


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_reloo(mock_wrapper_reloo, high_k_loo_data):
    result = reloo(
        mock_wrapper_reloo,
        loo_orig=high_k_loo_data,
        k_threshold=0.7,
    )
    assert isinstance(result, ELPDData)
    assert result.kind == "loo"
    assert result.pareto_k.values[0] == 0.0
    assert result.pareto_k.values[2] == 0.0
    assert result.pareto_k.values[5] == 0.0
    assert result.pareto_k.values[1] == high_k_loo_data.pareto_k.values[1]
    assert result.pareto_k.values[3] == high_k_loo_data.pareto_k.values[3]
    assert result.pareto_k.values[4] == high_k_loo_data.pareto_k.values[4]
    assert result.pareto_k.values[6] == high_k_loo_data.pareto_k.values[6]
    assert result.pareto_k.values[7] == high_k_loo_data.pareto_k.values[7]


def test_reloo_wrapper_validation():
    with pytest.raises(TypeError, match="wrapper must be an instance of SamplingWrapper"):
        reloo("not_a_wrapper")

    class IncompleteSamplingWrapper(SamplingWrapper):  # pylint: disable=abstract-method
        def sel_observations(self, idx):
            return None, None

    incomplete_wrapper = IncompleteSamplingWrapper(model=None)
    with pytest.raises(TypeError, match="does not implement all methods required"):
        reloo(incomplete_wrapper)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_reloo_multidimensional(mock_2d_data, mock_wrapper_2d):
    loo_orig = loo(mock_2d_data, pointwise=True, var_name="log_lik")
    loo_modified = ELPDData(
        elpd=loo_orig.elpd,
        se=loo_orig.se,
        p=loo_orig.p,
        good_k=loo_orig.good_k,
        n_samples=loo_orig.n_samples,
        n_data_points=loo_orig.n_data_points,
        warning=True,
        kind=loo_orig.kind,
        scale=loo_orig.scale,
        elpd_i=loo_orig.elpd_i.copy(),
        pareto_k=loo_orig.pareto_k.copy(),
    )

    high_k_locations = [
        {"school": "school_0", "measurement": "meas_1"},
        {"school": "school_1", "measurement": "meas_2"},
        {"school": "school_2", "measurement": "meas_0"},
    ]

    for loc in high_k_locations:
        loo_modified.pareto_k.loc[loc] = 0.85

    result = reloo(mock_wrapper_2d, loo_orig=loo_modified, k_threshold=0.7)

    for loc in high_k_locations:
        assert result.pareto_k.loc[loc] == 0.0

    unchanged_loc = {"school": "school_0", "measurement": "meas_0"}
    assert result.pareto_k.loc[unchanged_loc] == loo_modified.pareto_k.loc[unchanged_loc]


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_reloo_with_log_weights(mock_wrapper_reloo):
    loo_result = loo(mock_wrapper_reloo.idata_orig, pointwise=True, var_name="obs")

    result_with_weights = reloo(
        mock_wrapper_reloo,
        loo_orig=loo_result,
        k_threshold=0.7,
        pointwise=True,
        var_name="obs",
    )

    assert result_with_weights.elpd is not None and not np.isnan(result_with_weights.elpd)
    assert result_with_weights.se is not None and not np.isnan(result_with_weights.se)
    assert result_with_weights.p is not None and not np.isnan(result_with_weights.p)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_reloo_log_weights_storage(mock_wrapper_reloo, high_k_loo_data):
    result_pw_true = reloo(
        mock_wrapper_reloo, loo_orig=high_k_loo_data, k_threshold=0.7, pointwise=True
    )

    assert result_pw_true.log_weights is not None
    bad_k_mask = high_k_loo_data.pareto_k > 0.7
    assert np.all(np.isnan(result_pw_true.log_weights.values[bad_k_mask]))
    good_k_mask = ~bad_k_mask
    assert_array_almost_equal(
        result_pw_true.log_weights.values[good_k_mask],
        high_k_loo_data.log_weights.values[good_k_mask],
    )

    result_pw_false = reloo(
        mock_wrapper_reloo, loo_orig=high_k_loo_data, k_threshold=0.7, pointwise=False
    )

    assert result_pw_false.log_weights is not None
    assert_array_almost_equal(
        result_pw_false.log_weights.values, high_k_loo_data.log_weights.values
    )


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_reloo_loo_orig_none(mock_wrapper_reloo):
    result = reloo(mock_wrapper_reloo, loo_orig=None, k_threshold=0.7, var_name="obs")
    assert isinstance(result, ELPDData)
    assert result.kind == "loo"
    assert result.elpd is not None
    assert result.se is not None


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_reloo_loo_orig_none_elpd_data_input(mock_wrapper_reloo, high_k_loo_data):
    result = reloo(mock_wrapper_reloo, log_weights=high_k_loo_data, k_threshold=0.7)
    assert isinstance(result, ELPDData)
    assert result.pareto_k is not None


def test_reloo_missing_pointwise_data(mock_wrapper_reloo):
    loo_data_no_pointwise = loo(mock_wrapper_reloo.data, pointwise=False, var_name="obs")
    with pytest.raises(ValueError, match="reloo requires pointwise LOO results"):
        reloo(mock_wrapper_reloo, loo_orig=loo_data_no_pointwise)


def test_reloo_invalid_loo_orig_type(mock_wrapper_reloo):
    with pytest.raises(TypeError, match="loo_orig must be an ELPDData object"):
        reloo(mock_wrapper_reloo, loo_orig="not_elpd_data")


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_reloo_no_bad_observations(mock_wrapper_reloo, high_k_loo_data):
    result = reloo(mock_wrapper_reloo, loo_orig=high_k_loo_data, k_threshold=1.5)
    assert isinstance(result, ELPDData)
    assert_array_almost_equal(result.pareto_k.values, high_k_loo_data.pareto_k.values)
    assert_array_almost_equal(result.elpd_i.values, high_k_loo_data.elpd_i.values)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_reloo_no_bad_observations_pointwise_false(mock_wrapper_reloo, high_k_loo_data):
    result = reloo(mock_wrapper_reloo, loo_orig=high_k_loo_data, k_threshold=1.5, pointwise=False)
    assert isinstance(result, ELPDData)
    assert result.elpd_i is None
    assert result.pareto_k is None


@pytest.mark.parametrize("k_threshold", [0.5, 0.7, 0.9, 1.0])
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_reloo_k_threshold_values(mock_wrapper_reloo, high_k_loo_data, k_threshold):
    result = reloo(mock_wrapper_reloo, loo_orig=high_k_loo_data, k_threshold=k_threshold)
    assert isinstance(result, ELPDData)
    if k_threshold < 0.8:
        bad_k_count = np.sum(high_k_loo_data.pareto_k.values > k_threshold)
        refitted_count = np.sum(result.pareto_k.values == 0.0)
        assert refitted_count == bad_k_count


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_reloo_k_threshold_none(mock_wrapper_reloo, high_k_loo_data):
    result = reloo(mock_wrapper_reloo, loo_orig=high_k_loo_data, k_threshold=None)
    assert isinstance(result, ELPDData)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_reloo_var_name(mock_wrapper_reloo):
    result = reloo(mock_wrapper_reloo, loo_orig=None, var_name="obs", k_threshold=0.7)
    assert isinstance(result, ELPDData)
    assert result.kind == "loo"


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_reloo_pointwise_false(mock_wrapper_reloo, high_k_loo_data):
    result = reloo(mock_wrapper_reloo, loo_orig=high_k_loo_data, k_threshold=0.7, pointwise=False)
    assert isinstance(result, ELPDData)
    assert result.elpd_i is None
    assert result.pareto_k is None


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_reloo_updates_summary_statistics(mock_wrapper_reloo, high_k_loo_data):
    result = reloo(mock_wrapper_reloo, loo_orig=high_k_loo_data, k_threshold=0.7)
    assert result.elpd != high_k_loo_data.elpd
    assert result.se != high_k_loo_data.se
    assert result.p != high_k_loo_data.p


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_reloo_warning_flag(mock_wrapper_reloo, high_k_loo_data):
    result_high_threshold = reloo(mock_wrapper_reloo, loo_orig=high_k_loo_data, k_threshold=1.5)
    assert result_high_threshold.warning is True

    result_low_threshold = reloo(mock_wrapper_reloo, loo_orig=high_k_loo_data, k_threshold=0.5)
    max_k = np.max(result_low_threshold.pareto_k.values)
    if max_k <= result_low_threshold.good_k:
        assert not result_low_threshold.warning


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_reloo_p_loo_i_computation(mock_wrapper_reloo, high_k_loo_data):
    loo_data_no_p_loo = deepcopy(high_k_loo_data)
    if hasattr(loo_data_no_p_loo, "p_loo_i"):
        delattr(loo_data_no_p_loo, "p_loo_i")
    result = reloo(mock_wrapper_reloo, loo_orig=loo_data_no_p_loo, k_threshold=0.7)
    assert hasattr(result, "p_loo_i")
    assert result.p_loo_i is not None


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_reloo_preserves_good_k_values(mock_wrapper_reloo, high_k_loo_data):
    result = reloo(mock_wrapper_reloo, loo_orig=high_k_loo_data, k_threshold=0.7)
    good_k_mask = high_k_loo_data.pareto_k <= 0.7
    good_k_indices = np.where(good_k_mask.values)
    for idx in zip(*good_k_indices):
        assert result.pareto_k.values[idx] == high_k_loo_data.pareto_k.values[idx]


def test_reloo_elpd_data_without_log_weights(mock_wrapper_reloo):
    loo_result = loo(mock_wrapper_reloo.data, pointwise=True, var_name="obs")
    loo_result_no_weights = ELPDData(
        elpd=loo_result.elpd,
        se=loo_result.se,
        p=loo_result.p,
        good_k=loo_result.good_k,
        n_samples=loo_result.n_samples,
        n_data_points=loo_result.n_data_points,
        warning=loo_result.warning,
        kind=loo_result.kind,
        scale=loo_result.scale,
        elpd_i=loo_result.elpd_i,
        pareto_k=loo_result.pareto_k,
        log_weights=None,
    )
    with pytest.raises(ValueError, match="ELPDData object does not contain log_weights"):
        reloo(mock_wrapper_reloo, log_weights=loo_result_no_weights)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_reloo_dataset_log_weights(mock_2d_data, mock_wrapper_2d):
    loo_orig = loo(mock_2d_data, pointwise=True, var_name="log_lik")
    loo_modified = ELPDData(
        elpd=loo_orig.elpd,
        se=loo_orig.se,
        p=loo_orig.p,
        good_k=loo_orig.good_k,
        n_samples=loo_orig.n_samples,
        n_data_points=loo_orig.n_data_points,
        warning=True,
        kind=loo_orig.kind,
        scale=loo_orig.scale,
        elpd_i=loo_orig.elpd_i.copy(),
        pareto_k=loo_orig.pareto_k.copy(),
    )
    loo_modified.pareto_k.loc[{"school": "school_0", "measurement": "meas_0"}] = 0.85
    result = reloo(mock_wrapper_2d, loo_orig=loo_modified, k_threshold=0.7)
    assert result.pareto_k.loc[{"school": "school_0", "measurement": "meas_0"}] == 0.0
