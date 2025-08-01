# pylint: disable=redefined-outer-name, unused-import, too-many-function-args
# ruff: noqa: F811
"""Test k-fold cross-validation."""

import numpy as np
import pytest
from scipy import stats

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
def custom_folds():
    return np.array([1, 1, 2, 2, 3, 3, 4, 4])


@pytest.fixture
def strata():
    return np.array([0, 0, 1, 1, 0, 0, 1, 1])


@pytest.fixture
def groups():
    return np.array([1, 1, 2, 2, 3, 3, 4, 4])


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

            train_data = {
                "indices": train_indices,
                "y": train_y,
                "sigma": self.sigma_values[train_indices],
                "n_schools": len(train_indices),
            }
            test_data = {
                "indices": idx,
                "y": test_y,
                "sigma": self.sigma_values[idx],
                "n_schools": len(idx),
            }

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
            test_sigma = excluded_obs["sigma"]
            mu = idata__i.posterior["mu"].values.flatten()
            tau = idata__i.posterior["tau"].values.flatten()

            var_total = tau[:, np.newaxis] ** 2 + test_sigma**2
            log_lik = stats.norm.logpdf(test_y, loc=mu[:, np.newaxis], scale=np.sqrt(var_total))

            dims = ["chain", "school", "draw"]
            coords = {"school": excluded_obs["indices"]}
            return xr.DataArray(log_lik.T[np.newaxis, :, :], dims=dims, coords=coords)

    return CenteredEightWrapper(centered_eight)


@pytest.fixture
def fresh_wrapper(mock_wrapper):
    mock_wrapper.fit_count = 0
    mock_wrapper.test_indices_history = []
    return mock_wrapper


@pytest.mark.parametrize("pointwise", [True, False])
def test_loo_kfold(centered_eight, fresh_wrapper, pointwise):
    kfold_data = loo_kfold(data=centered_eight, pointwise=pointwise, wrapper=fresh_wrapper, k=4)

    assert isinstance(kfold_data, ELPDData)
    assert kfold_data.kind == "loo_kfold"
    assert kfold_data.n_data_points == 8
    assert kfold_data.n_samples == 2000
    assert kfold_data.scale == "log"
    assert kfold_data.warning is False
    assert kfold_data.good_k is None
    assert kfold_data.pareto_k is None
    assert fresh_wrapper.fit_count == 4

    if pointwise:
        assert kfold_data.elpd_i is not None
        assert kfold_data.elpd_i.shape == (8,)
        assert hasattr(kfold_data, "p_kfold_i")
        assert kfold_data.p_kfold_i is not None
        assert kfold_data.p_kfold_i.shape == (8,)
    else:
        assert kfold_data.elpd_i is None
        assert not hasattr(kfold_data, "p_kfold_i")


@pytest.mark.parametrize("k", [2, 5, 8])
def test_loo_kfold_different_k(centered_eight, fresh_wrapper, k):
    kfold_data = loo_kfold(data=centered_eight, pointwise=False, wrapper=fresh_wrapper, k=k)

    assert fresh_wrapper.fit_count == k
    assert isinstance(kfold_data.elpd, float)
    assert isinstance(kfold_data.se, float)
    assert isinstance(kfold_data.p, float)


@pytest.mark.parametrize("pointwise", [True, False])
def test_loo_kfold_custom_folds(centered_eight, fresh_wrapper, custom_folds, pointwise):
    kfold_data = loo_kfold(
        data=centered_eight, pointwise=pointwise, wrapper=fresh_wrapper, folds=custom_folds
    )

    assert kfold_data.kind == "loo_kfold"
    assert fresh_wrapper.fit_count == 4


@pytest.mark.parametrize("k,pointwise", [(4, False), (2, True)])
def test_loo_kfold_stratified(centered_eight, fresh_wrapper, strata, k, pointwise):
    kfold_data = loo_kfold(
        data=centered_eight, pointwise=pointwise, wrapper=fresh_wrapper, k=k, stratify_by=strata
    )

    assert kfold_data.kind == "loo_kfold"
    assert fresh_wrapper.fit_count == k


def test_loo_kfold_grouped(centered_eight, fresh_wrapper, groups):
    kfold_data = loo_kfold(
        data=centered_eight, pointwise=False, wrapper=fresh_wrapper, k=2, group_by=groups
    )

    assert kfold_data.kind == "loo_kfold"
    assert fresh_wrapper.fit_count == 2


@pytest.mark.parametrize("save_fits", [True, False])
def test_loo_kfold_save_fits(centered_eight, fresh_wrapper, save_fits):
    kfold_data = loo_kfold(
        data=centered_eight, pointwise=False, wrapper=fresh_wrapper, k=4, save_fits=save_fits
    )

    if save_fits:
        assert hasattr(kfold_data, "fold_fits")
        assert len(kfold_data.fold_fits) == 4

        for k in range(1, 5):
            assert k in kfold_data.fold_fits
            assert "fit" in kfold_data.fold_fits[k]
            assert "test_indices" in kfold_data.fold_fits[k]
    else:
        assert not hasattr(kfold_data, "fold_fits") or kfold_data.fold_fits is None


@pytest.mark.parametrize(
    "input_type,dims",
    [
        ("folds", ["school"]),
        ("stratify_by", ["school"]),
        ("group_by", ["school"]),
        ("group_by_2d", ["school", "county"]),
    ],
)
def test_loo_kfold_dataarray_inputs(centered_eight, fresh_wrapper, input_type, dims):
    if input_type == "folds":
        da = xr.DataArray([1, 1, 2, 2, 3, 3, 4, 4], dims=dims)
        kfold_data = loo_kfold(data=centered_eight, pointwise=True, wrapper=fresh_wrapper, folds=da)
        expected_fits = 4
    elif input_type == "stratify_by":
        da = xr.DataArray([0, 0, 1, 1, 0, 0, 1, 1], dims=dims)
        kfold_data = loo_kfold(
            data=centered_eight, pointwise=False, wrapper=fresh_wrapper, k=4, stratify_by=da
        )
        expected_fits = 4
    elif input_type == "group_by":
        da = xr.DataArray([1, 1, 2, 2, 3, 3, 4, 4], dims=dims)
        kfold_data = loo_kfold(
            data=centered_eight, pointwise=False, wrapper=fresh_wrapper, k=2, group_by=da
        )
        expected_fits = 2
    else:
        da = xr.DataArray([[1, 2], [1, 2], [3, 4], [3, 4]], dims=dims)
        kfold_data = loo_kfold(
            data=centered_eight, pointwise=False, wrapper=fresh_wrapper, k=4, group_by=da
        )
        expected_fits = 4

    assert kfold_data.kind == "loo_kfold"
    assert fresh_wrapper.fit_count == expected_fits


@pytest.mark.parametrize(
    "error_case,expected_error,expected_msg",
    [
        ("invalid_wrapper", TypeError, "wrapper must be an instance of SamplingWrapper"),
        (
            "stratify_with_folds",
            ValueError,
            "Cannot use stratify_by or group_by when folds are explicitly provided",
        ),
        (
            "group_with_folds",
            ValueError,
            "Cannot use stratify_by or group_by when folds are explicitly provided",
        ),
        ("both_stratify_group", ValueError, "Cannot use both stratify_by and group_by"),
        ("short_stratify", ValueError, "Length of stratify_by"),
    ],
)
def test_loo_kfold_errors(centered_eight, mock_wrapper, error_case, expected_error, expected_msg):
    if error_case == "invalid_wrapper":
        with pytest.raises(expected_error, match=expected_msg):
            loo_kfold(data=centered_eight, wrapper="not_a_wrapper")
    elif error_case == "stratify_with_folds":
        with pytest.raises(expected_error, match=expected_msg):
            loo_kfold(
                data=centered_eight,
                wrapper=mock_wrapper,
                folds=[1, 2, 1, 2, 1, 2, 1, 2],
                stratify_by=[0, 0, 1, 1, 0, 0, 1, 1],
            )
    elif error_case == "group_with_folds":
        with pytest.raises(expected_error, match=expected_msg):
            loo_kfold(
                data=centered_eight,
                wrapper=mock_wrapper,
                folds=[1, 2, 1, 2, 1, 2, 1, 2],
                group_by=[1, 1, 2, 2, 3, 3, 4, 4],
            )
    elif error_case == "both_stratify_group":
        with pytest.raises(expected_error, match=expected_msg):
            loo_kfold(
                data=centered_eight,
                wrapper=mock_wrapper,
                stratify_by=[0, 0, 1, 1, 0, 0, 1, 1],
                group_by=[1, 1, 2, 2, 3, 3, 4, 4],
            )
    elif error_case == "short_stratify":
        with pytest.raises(expected_error, match=expected_msg):
            loo_kfold(
                data=centered_eight,
                wrapper=mock_wrapper,
                stratify_by=[0, 1, 0],
            )
