# pylint: disable=redefined-outer-name, unused-import
# ruff: noqa: F811
"""Test helper functions for k-fold cross-validation."""

import numpy as np
import pytest

from .helpers import datatree, importorskip  # noqa: F401

azb = importorskip("arviz_base")
xr = importorskip("xarray")

from arviz_stats.loo import SamplingWrapper
from arviz_stats.loo.helper_loo_kfold import (
    _combine_fold_elpds,
    _extract_fold_data,
    _get_fold_indices,
    _kfold_split_grouped,
    _kfold_split_random,
    _kfold_split_stratified,
    _prepare_kfold_inputs,
    _validate_array_length,
    _validate_fold_parameters,
    _validate_k_value,
)


@pytest.fixture(name="simple_data", scope="session")
def fixture_simple_data():
    return azb.load_arviz_data("centered_eight")


@pytest.fixture
def simple_wrapper():
    class SimpleWrapper(SamplingWrapper):
        def __init__(self):
            super().__init__(model=None, idata_orig=None)

        def sel_observations(self, idx):
            return {"train": idx}, {"test": idx}

        def sample(self, modified_observed_data):
            return {"data": modified_observed_data}

        def get_inference_data(self, fitted_model):
            return fitted_model

        def log_likelihood__i(self, excluded_obs, idata__i):
            return np.random.randn(10)

    return SimpleWrapper()


@pytest.fixture
def sample_array():
    return np.array([1, 2, 3, 4, 5])


@pytest.fixture
def sample_dataarray():
    return xr.DataArray([1, 2, 3, 4, 5], dims=["obs"])


@pytest.fixture
def fold_assignments():
    return np.array([1, 1, 2, 2, 3, 3, 4, 4])


@pytest.fixture
def stratified_data():
    return np.array([0, 0, 0, 0, 1, 1, 1, 1])


@pytest.fixture
def grouped_data():
    return np.array([1, 1, 2, 2, 3, 3, 4, 4])


@pytest.mark.parametrize(
    "k,n,expected",
    [
        (5, 10, 5),
        (2, 100, 2),
        (10, 10, 10),
    ],
)
def test_validate_k_value_valid(k, n, expected):
    assert _validate_k_value(k, n) == expected


@pytest.mark.parametrize(
    "k,n,error_msg",
    [
        ("not_int", 10, "k must be an integer"),
        (2.5, 10, "k must be an integer"),
        (1, 10, "k must be greater than 1"),
        (11, 10, "k must not be greater than n"),
    ],
)
def test_validate_k_value_invalid(k, n, error_msg):
    with pytest.raises(ValueError, match=error_msg):
        _validate_k_value(k, n)


def test_validate_array_length_numpy(sample_array):
    result = _validate_array_length(sample_array, 5, "test_array")
    np.testing.assert_array_equal(result, sample_array)


def test_validate_array_length_xarray(sample_dataarray):
    result = _validate_array_length(sample_dataarray, 5, "test_da")
    np.testing.assert_array_equal(result, sample_dataarray.values)


@pytest.mark.parametrize(
    "array,length,param_name,error_pattern",
    [
        (None, 5, "test_array", "test_array must be provided"),
        ([1, 2, 3], 5, "test_array", "Length of test_array"),
    ],
)
def test_validate_array_length_errors(array, length, param_name, error_pattern):
    with pytest.raises(ValueError, match=error_pattern):
        _validate_array_length(array, length, param_name)


@pytest.mark.parametrize(
    "folds,stratify_by,group_by",
    [
        (None, None, None),
        ([1, 2, 3], None, None),
        (None, [1, 2], None),
        (None, None, [1, 2]),
    ],
)
def test_validate_fold_parameters_valid(folds, stratify_by, group_by):
    _validate_fold_parameters(folds, stratify_by, group_by)


@pytest.mark.parametrize(
    "folds,stratify_by,group_by,error_msg",
    [
        (
            [1, 2],
            [1, 2],
            None,
            "Cannot use stratify_by or group_by when folds are explicitly provided",
        ),
        (
            [1, 2],
            None,
            [1, 2],
            "Cannot use stratify_by or group_by when folds are explicitly provided",
        ),
        (None, [1, 2], [1, 2], "Cannot use both stratify_by and group_by"),
    ],
)
def test_validate_fold_parameters_invalid(folds, stratify_by, group_by, error_msg):
    with pytest.raises(ValueError, match=error_msg):
        _validate_fold_parameters(folds, stratify_by, group_by)


@pytest.mark.parametrize(
    "n,k",
    [
        (20, 4),
        (10, 5),
        (8, 2),
    ],
)
def test_kfold_split_random(n, k):
    folds = _kfold_split_random(k=k, n=n)

    assert len(folds) == n
    assert set(folds) == set(range(1, k + 1))

    fold_sizes = [np.sum(folds == i) for i in range(1, k + 1)]
    assert all(n // k <= size <= n // k + 1 for size in fold_sizes)


def test_kfold_split_random_errors():
    with pytest.raises(ValueError, match="n must be provided"):
        _kfold_split_random(k=5)

    with pytest.raises(ValueError, match="n must be an integer"):
        _kfold_split_random(k=5, n="not_int")

    with pytest.raises(ValueError, match="k must not be greater than n"):
        _kfold_split_random(k=10, n=5)


@pytest.mark.parametrize("k", [2, 4])
def test_kfold_split_stratified(stratified_data, k):
    folds = _kfold_split_stratified(k=k, x=stratified_data)

    assert len(folds) == len(stratified_data)
    assert set(folds) == set(range(1, k + 1))

    for fold in range(1, k + 1):
        fold_mask = folds == fold
        assert np.sum(stratified_data[fold_mask] == 0) >= 1
        assert np.sum(stratified_data[fold_mask] == 1) >= 1


def test_kfold_split_stratified_errors():
    with pytest.raises(ValueError, match="x must be provided"):
        _kfold_split_stratified(k=5)


def test_kfold_split_grouped(grouped_data):
    k = 2
    folds = _kfold_split_grouped(k=k, x=grouped_data)

    assert len(folds) == len(grouped_data)
    assert set(folds) == {1, 2}

    for group in [1, 2, 3, 4]:
        group_mask = grouped_data == group
        group_folds = folds[group_mask]
        assert len(set(group_folds)) == 1


def test_kfold_split_grouped_many_groups():
    x_many_groups = np.repeat(np.arange(10), 3)
    k = 3
    folds = _kfold_split_grouped(k=k, x=x_many_groups)
    assert len(set(folds)) <= k
    assert len(set(folds)) >= 1
    assert min(folds) >= 1
    assert max(folds) <= k

    for group_id in np.unique(x_many_groups):
        group_mask = x_many_groups == group_id
        group_folds = folds[group_mask]
        assert len(set(group_folds)) == 1


def test_kfold_split_grouped_errors():
    with pytest.raises(ValueError, match="x must be provided"):
        _kfold_split_grouped(k=5)

    with pytest.raises(ValueError, match="k must not be bigger than the number of levels"):
        _kfold_split_grouped(k=3, x=np.array([1, 1, 2, 2]))


@pytest.mark.parametrize("train", [True, False])
def test_extract_fold_data(sample_dataarray, train):
    fold_indices = np.array([2, 3, 4])

    result = _extract_fold_data(sample_dataarray, fold_indices, train=train)

    if train:
        assert result.shape == (2,)
        np.testing.assert_array_equal(result.values, [1, 2])
    else:
        assert result.shape == (3,)
        np.testing.assert_array_equal(result.values, [3, 4, 5])


def test_get_fold_indices(fold_assignments):
    k = 4
    fold_indices = _get_fold_indices(fold_assignments, k)

    assert len(fold_indices) == k

    expected_indices = {
        1: [0, 1],
        2: [2, 3],
        3: [4, 5],
        4: [6, 7],
    }

    for i in range(1, k + 1):
        assert fold_indices[i]["test_indices"].tolist() == expected_indices[i]
        assert fold_indices[i]["n_test"] == 2


def test_combine_fold_elpds():
    fold_elpds = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    n_data_points = 4

    result = _combine_fold_elpds(fold_elpds, n_data_points)

    assert result["elpd_kfold"] == 10.0
    np.testing.assert_array_equal(result["pointwise"], [1.0, 2.0, 3.0, 4.0])

    expected_se = np.sqrt(4 * np.var([1.0, 2.0, 3.0, 4.0]))
    assert np.isclose(result["se_elpd_kfold"], expected_se)


def test_prepare_kfold_inputs(simple_data, simple_wrapper):
    result = _prepare_kfold_inputs(
        data=simple_data,
        var_name=None,
        wrapper=simple_wrapper,
        k=4,
        folds=None,
        stratify_by=None,
        group_by=None,
    )

    assert result.log_likelihood is not None
    assert result.n_data_points == 8
    assert result.k == 4
    assert len(result.folds) == 8
    assert set(result.folds) == {1, 2, 3, 4}


def test_prepare_kfold_inputs_invalid_wrapper(simple_data):
    with pytest.raises(TypeError, match="wrapper must be an instance of SamplingWrapper"):
        _prepare_kfold_inputs(
            data=simple_data,
            var_name=None,
            wrapper="not_a_wrapper",
            k=4,
            folds=None,
            stratify_by=None,
            group_by=None,
        )


@pytest.mark.parametrize(
    "custom_folds",
    [
        np.array([1, 1, 2, 2, 3, 3, 4, 4]),
        np.array([1, 2, 1, 2, 1, 2, 1, 2]),
    ],
)
def test_prepare_kfold_inputs_with_custom_folds(simple_data, simple_wrapper, custom_folds):
    result = _prepare_kfold_inputs(
        data=simple_data,
        var_name=None,
        wrapper=simple_wrapper,
        k=None,
        folds=custom_folds,
        stratify_by=None,
        group_by=None,
    )

    assert result.k == len(np.unique(custom_folds))
    np.testing.assert_array_equal(result.folds, custom_folds)


def test_prepare_kfold_inputs_with_stratify(simple_data, simple_wrapper):
    strata = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    result = _prepare_kfold_inputs(
        data=simple_data,
        var_name=None,
        wrapper=simple_wrapper,
        k=4,
        folds=None,
        stratify_by=strata,
        group_by=None,
    )

    assert result.k == 4
    assert len(result.folds) == 8


def test_prepare_kfold_inputs_with_groups(simple_data, simple_wrapper):
    groups = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    result = _prepare_kfold_inputs(
        data=simple_data,
        var_name=None,
        wrapper=simple_wrapper,
        k=2,
        folds=None,
        stratify_by=None,
        group_by=groups,
    )

    assert result.k == 2
    assert len(result.folds) == 8
