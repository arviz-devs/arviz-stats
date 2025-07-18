# pylint: disable=redefined-outer-name, unused-import, too-many-function-args
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
    class CenteredEightWrapper(SamplingWrapper):
        def __init__(self, idata):
            super().__init__(model=None, idata_orig=idata)
            self.fit_count = 0
            self.test_indices_history = []
            self.rng = np.random.default_rng(42)

            self.original_posterior = idata.posterior
            self.original_obs_data = idata.observed_data["obs"].values
            self.n_schools = len(self.original_obs_data)

            self.sigma_values = np.array([15, 10, 16, 11, 9, 11, 10, 18])

        def sel_observations(self, idx):
            all_indices = np.arange(self.n_schools)
            train_indices = np.setdiff1d(all_indices, idx)

            train_y = self.original_obs_data[train_indices]
            test_y = self.original_obs_data[idx]

            train_data = {"indices": train_indices, "y": train_y, "n_schools": len(train_indices)}
            test_data = {"indices": idx, "y": test_y, "n_schools": len(idx)}

            self.test_indices_history.append(idx)
            return train_data, test_data

        def sample(self, modified_observed_data):
            self.fit_count += 1

            train_y = modified_observed_data["y"]
            n_train = len(train_y)

            y_mean = np.mean(train_y)
            y_std = np.std(train_y, ddof=1)

            n_samples = 2000
            mu_samples = self.rng.normal(y_mean, y_std / np.sqrt(n_train), n_samples)

            tau_samples = np.abs(self.rng.normal(y_std, y_std / 2, n_samples))
            tau_samples = np.maximum(tau_samples, 0.1)

            theta_samples = np.zeros((n_samples, n_train))
            for i in range(n_samples):
                theta_samples[i] = self.rng.normal(mu_samples[i], tau_samples[i], n_train)

            return {
                "mu": mu_samples,
                "tau": tau_samples,
                "theta": theta_samples,
                "train_indices": modified_observed_data["indices"],
                "n_train": n_train,
            }

        def get_inference_data(self, fitted_model):
            posterior_dict = {
                "mu": xr.DataArray(
                    fitted_model["mu"].reshape(1, 2000),
                    dims=["chain", "draw"],
                    coords={"chain": [0], "draw": np.arange(2000)},
                ),
                "tau": xr.DataArray(
                    fitted_model["tau"].reshape(1, 2000),
                    dims=["chain", "draw"],
                    coords={"chain": [0], "draw": np.arange(2000)},
                ),
                "theta": xr.DataArray(
                    fitted_model["theta"].T.reshape(1, 2000, fitted_model["n_train"]),
                    dims=["chain", "draw", "school"],
                    coords={
                        "chain": [0],
                        "draw": np.arange(2000),
                        "school": fitted_model["train_indices"],
                    },
                ),
            }

            idata_new = azb.from_dict(
                {"posterior": posterior_dict},
                dims={"school": ["school"]},
                coords={"school": fitted_model["train_indices"]},
            )

            return idata_new

        def log_likelihood__i(self, excluded_obs, idata__i):
            test_y = excluded_obs["y"]
            test_indices = excluded_obs["indices"]
            n_test = len(test_indices)

            mu_samples = idata__i.posterior["mu"].values.flatten()
            tau_samples = idata__i.posterior["tau"].values.flatten()
            n_samples = len(mu_samples)

            sigma_values = self.sigma_values[test_indices]
            log_lik = np.zeros((n_samples, n_test))

            for i in range(n_samples):
                for j, (y_j, sigma_j) in enumerate(zip(test_y, sigma_values)):
                    total_var = tau_samples[i] ** 2 + sigma_j**2
                    log_lik[i, j] = (
                        -0.5 * np.log(2 * np.pi * total_var)
                        - 0.5 * (y_j - mu_samples[i]) ** 2 / total_var
                    )

            return xr.DataArray(
                log_lik.T.reshape(1, n_test, n_samples),
                dims=["chain", "school", "draw"],
                coords={"chain": [0], "school": test_indices, "draw": np.arange(n_samples)},
            )

    return CenteredEightWrapper(centered_eight)


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


def test_loo_kfold_dataarray_inputs(centered_eight, mock_wrapper):
    mock_wrapper.fit_count = 0
    mock_wrapper.test_indices_history = []

    folds_da = xr.DataArray([1, 1, 2, 2, 3, 3, 4, 4], dims=["school"])
    kfold_data = loo_kfold(
        data=centered_eight, pointwise=True, wrapper=mock_wrapper, folds=folds_da
    )
    assert kfold_data.kind == "loo_kfold"
    assert mock_wrapper.fit_count == 4

    mock_wrapper.fit_count = 0
    strata_da = xr.DataArray([0, 0, 1, 1, 0, 0, 1, 1], dims=["school"])
    kfold_data = loo_kfold(
        data=centered_eight, pointwise=False, wrapper=mock_wrapper, k=4, stratify_by=strata_da
    )
    assert kfold_data.kind == "loo_kfold"
    assert mock_wrapper.fit_count == 4

    mock_wrapper.fit_count = 0
    groups_da = xr.DataArray([1, 1, 2, 2, 3, 3, 4, 4], dims=["school"])
    kfold_data = loo_kfold(
        data=centered_eight, pointwise=False, wrapper=mock_wrapper, k=2, group_by=groups_da
    )
    assert kfold_data.kind == "loo_kfold"
    assert mock_wrapper.fit_count == 2

    mock_wrapper.fit_count = 0
    folds_2d = xr.DataArray([[1, 2], [1, 2], [3, 4], [3, 4]], dims=["school", "county"])
    kfold_data = loo_kfold(
        data=centered_eight, pointwise=False, wrapper=mock_wrapper, k=4, group_by=folds_2d
    )
    assert kfold_data.kind == "loo_kfold"
    assert mock_wrapper.fit_count == 4


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
