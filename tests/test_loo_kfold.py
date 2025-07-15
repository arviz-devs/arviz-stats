# pylint: disable=redefined-outer-name, unused-import
# ruff: noqa: F811
"""Test k-fold cross-validation."""

import numpy as np
import pytest

from .helpers import datatree, importorskip  # noqa: F401

azb = importorskip("arviz_base")
xr = importorskip("xarray")

from arviz_stats import loo_kfold
from arviz_stats.loo import SamplingWrapper
from arviz_stats.utils import ELPDData


@pytest.fixture(name="centered_eight", scope="session")
def fixture_centered_eight():
    return azb.load_arviz_data("centered_eight")


@pytest.fixture
def mock_wrapper(centered_eight):
    class MockSamplingWrapper(SamplingWrapper):
        def __init__(self, idata):
            super().__init__(model=None, idata_orig=idata)
            self.fit_count = 0
            self.test_indices_history = []

        def sel_observations(self, idx):
            train_data = {"n": len(idx)}
            test_data = {"indices": idx}
            self.test_indices_history.append(idx)
            return train_data, test_data

        def sample(self, modified_observed_data):
            self.fit_count += 1
            return {"model_id": self.fit_count, "data": modified_observed_data}

        def get_inference_data(self, fitted_model):
            return self.idata_orig

        def log_likelihood__i(self, excluded_obs, idata__i):
            log_lik = idata__i.log_likelihood["obs"]
            test_idx = excluded_obs["indices"]
            return log_lik.isel(school=test_idx)

    return MockSamplingWrapper(centered_eight)


@pytest.mark.parametrize("pointwise", [True, False])
def test_loo_kfold(centered_eight, mock_wrapper, pointwise):
    mock_wrapper.fit_count = 0
    mock_wrapper.test_indices_history = []

    kfold_data = loo_kfold(data=centered_eight, pointwise=pointwise, wrapper=mock_wrapper, k=4)

    assert isinstance(kfold_data, ELPDData)
    assert kfold_data.kind == "loo_kfold"
    assert kfold_data.n_data_points == 8
    assert kfold_data.n_samples == 2000
    assert kfold_data.scale == "log"
    assert kfold_data.warning is False
    assert kfold_data.good_k is None
    assert kfold_data.pareto_k is None

    assert mock_wrapper.fit_count == 4

    if pointwise:
        assert kfold_data.elpd_i is not None
        assert kfold_data.elpd_i.shape == (8, 2)
        assert list(kfold_data.elpd_i.coords["metric"].values) == [
            "elpd_loo_kfold",
            "p_loo_kfold",
        ]
    else:
        assert kfold_data.elpd_i is None


@pytest.mark.parametrize("k", [2, 5, 8])
def test_loo_kfold_different_k(centered_eight, mock_wrapper, k):
    mock_wrapper.fit_count = 0
    mock_wrapper.test_indices_history = []

    kfold_data = loo_kfold(data=centered_eight, pointwise=False, wrapper=mock_wrapper, k=k)

    assert mock_wrapper.fit_count == k
    assert isinstance(kfold_data.elpd, float)
    assert isinstance(kfold_data.se, float)
    assert isinstance(kfold_data.p, float)


def test_loo_kfold_custom_folds(centered_eight, mock_wrapper):
    mock_wrapper.fit_count = 0
    mock_wrapper.test_indices_history = []

    folds = np.array([1, 1, 2, 2, 3, 3, 4, 4])

    kfold_data = loo_kfold(data=centered_eight, pointwise=True, wrapper=mock_wrapper, folds=folds)

    assert kfold_data.kind == "loo_kfold"
    assert mock_wrapper.fit_count == 4


def test_loo_kfold_stratified(centered_eight, mock_wrapper):
    mock_wrapper.fit_count = 0
    mock_wrapper.test_indices_history = []

    strata = np.array([0, 0, 1, 1, 0, 0, 1, 1])

    kfold_data = loo_kfold(
        data=centered_eight, pointwise=False, wrapper=mock_wrapper, k=4, stratify_by=strata
    )

    assert kfold_data.kind == "loo_kfold"
    assert mock_wrapper.fit_count == 4


def test_loo_kfold_grouped(centered_eight, mock_wrapper):
    mock_wrapper.fit_count = 0
    mock_wrapper.test_indices_history = []

    groups = np.array([1, 1, 2, 2, 3, 3, 4, 4])

    kfold_data = loo_kfold(
        data=centered_eight, pointwise=False, wrapper=mock_wrapper, k=2, group_by=groups
    )

    assert kfold_data.kind == "loo_kfold"
    assert mock_wrapper.fit_count == 2


def test_loo_kfold_save_fits(centered_eight, mock_wrapper):
    mock_wrapper.fit_count = 0
    mock_wrapper.test_indices_history = []

    kfold_data = loo_kfold(
        data=centered_eight, pointwise=False, wrapper=mock_wrapper, k=4, save_fits=True
    )

    assert hasattr(kfold_data, "fold_fits")
    assert len(kfold_data.fold_fits) == 4

    for k in range(1, 5):
        assert k in kfold_data.fold_fits
        assert "fit" in kfold_data.fold_fits[k]
        assert "test_indices" in kfold_data.fold_fits[k]


def test_loo_kfold_errors(centered_eight, mock_wrapper):
    with pytest.raises(TypeError, match="wrapper must be an instance of SamplingWrapper"):
        loo_kfold(data=centered_eight, wrapper="not_a_wrapper")

    match_msg = "Cannot use stratify_by or group_by when folds are explicitly provided"
    with pytest.raises(ValueError, match=match_msg):
        loo_kfold(
            data=centered_eight,
            wrapper=mock_wrapper,
            folds=[1, 2, 1, 2, 1, 2, 1, 2],
            stratify_by=[0, 0, 1, 1, 0, 0, 1, 1],
        )

    with pytest.raises(ValueError, match="Cannot use both stratify_by and group_by"):
        loo_kfold(
            data=centered_eight,
            wrapper=mock_wrapper,
            stratify_by=[0, 0, 1, 1, 0, 0, 1, 1],
            group_by=[1, 1, 2, 2, 3, 3, 4, 4],
        )

    with pytest.raises(ValueError, match="Length of stratify_by"):
        loo_kfold(
            data=centered_eight,
            wrapper=mock_wrapper,
            stratify_by=[0, 1, 0],
        )
