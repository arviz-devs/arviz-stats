# pylint: disable=redefined-outer-name
import numpy as np
import pytest

from .helpers import importorskip

azb = importorskip("arviz_base")
xr = importorskip("xarray")

from arviz_stats import loo, reloo
from arviz_stats.loo.wrapper import SamplingWrapper
from arviz_stats.utils import ELPDData


@pytest.fixture(name="non_centered_eight", scope="session")
def fixture_non_centered_eight():
    return azb.load_arviz_data("non_centered_eight")


class MockSamplingWrapper(SamplingWrapper):
    def __init__(self, model, idata_orig=None, **kwargs):
        super().__init__(model, idata_orig, **kwargs)
        self.data = idata_orig
        self.log_lik_var_name = "obs"
        self.refit_count = 0
        self.failed_refits = []
        self.actual_refits = 0

    def sel_observations(self, idx):
        obs_data = self.data.observed_data["obs"]
        all_indices = list(range(len(obs_data)))
        all_indices.remove(idx)
        modified_obs = obs_data.isel(obs_dim_0=all_indices)
        excluded_obs = obs_data.isel(obs_dim_0=idx)
        return modified_obs, excluded_obs

    def sample(self, modified_observed_data):
        self.refit_count += 1
        return {"fit": "mock_fit", "data": modified_observed_data}

    def get_inference_data(self, fitted_model):
        return self.data

    def log_likelihood__i(self, excluded_obs, idata__i):
        posterior = idata__i.posterior
        theta = posterior["theta"]

        obs_idx = int(excluded_obs.obs_dim_0.values)
        theta_i = theta.isel(school=obs_idx)

        sigma = 12.5
        log_lik_values = (
            -0.5 * np.log(2 * np.pi * sigma**2)
            - 0.5 * ((excluded_obs.values - theta_i) / sigma) ** 2
        )

        log_lik = log_lik_values
        return log_lik


@pytest.fixture
def mock_wrapper(non_centered_eight):
    return MockSamplingWrapper(model=None, idata_orig=non_centered_eight)


@pytest.fixture
def high_k_loo_data(non_centered_eight):
    loo_data = loo(non_centered_eight, pointwise=True, var_name="obs")

    loo_data_modified = ELPDData(
        elpd=loo_data.elpd,
        se=loo_data.se,
        p=loo_data.p,
        good_k=loo_data.good_k,
        n_samples=loo_data.n_samples,
        n_data_points=loo_data.n_data_points,
        warning=True,
        kind=loo_data.kind,
        scale=loo_data.scale,
        elpd_i=loo_data.elpd_i.copy(),
        pareto_k=loo_data.pareto_k.copy(),
    )

    loo_data_modified.pareto_k.values[[0, 2, 5]] = [0.8, 0.9, 1.1]
    return loo_data_modified


def test_reloo(mock_wrapper, high_k_loo_data):
    high_k_indices = np.where(high_k_loo_data.pareto_k.values > 0.7)[0]

    result = reloo(
        mock_wrapper,
        loo_orig=high_k_loo_data,
        k_threshold=0.7,
    )

    assert mock_wrapper.refit_count == len(high_k_indices) + 1

    assert isinstance(result, ELPDData)
    assert result.kind == "loo"

    assert result.pareto_k.values[0] == 0.0
    assert result.pareto_k.values[1] == 0.0
    assert result.pareto_k.values[2] == 0.0
    assert result.pareto_k.values[5] == 0.0

    assert result.pareto_k.values[3] == high_k_loo_data.pareto_k.values[3]
    assert result.pareto_k.values[4] == high_k_loo_data.pareto_k.values[4]


def test_reloo_wrapper_validation():
    with pytest.raises(TypeError, match="wrapper must be an instance of SamplingWrapper"):
        reloo("not_a_wrapper")

    class IncompleteSamplingWrapper(SamplingWrapper):  # pylint: disable=abstract-method
        def sel_observations(self, idx):
            return None, None

    incomplete_wrapper = IncompleteSamplingWrapper(model=None)
    with pytest.raises(TypeError, match="does not implement all methods required"):
        reloo(incomplete_wrapper)
