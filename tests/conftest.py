"""Common fixtures for tests."""

# pylint: disable=redefined-outer-name
import numpy as np
import pytest

from .helpers import (
    create_data_random,
    create_model,
    create_multidimensional_model,
    importorskip,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture()
def data_random():
    idata = create_data_random()
    return idata


@pytest.fixture(scope="module")
def models():
    class Models:
        model_1 = create_model(seed=10)
        model_2 = create_model(seed=11, transpose=True)

    return Models()


@pytest.fixture(scope="module")
def multidim_models():
    class Models:
        model_1 = create_multidimensional_model(seed=10)
        model_2 = create_multidimensional_model(seed=11, transpose=True)

    return Models()


@pytest.fixture(scope="session")
def datatree():
    azb = importorskip("arviz_base")
    return azb.testing.datatree()


@pytest.fixture(scope="session")
def datatree_binary():
    azb = importorskip("arviz_base")
    return azb.testing.datatree_binary()


@pytest.fixture(scope="session")
def datatree_regression():
    azb = importorskip("arviz_base")
    return azb.testing.datatree_regression()


@pytest.fixture(scope="session")
def fake_dt():
    azb = importorskip("arviz_base")
    return azb.testing.fake_dt()


@pytest.fixture(scope="session")
def centered_eight():
    azb = importorskip("arviz_base")
    return azb.load_arviz_data("centered_eight")


@pytest.fixture(scope="session")
def non_centered_eight():
    azb = importorskip("arviz_base")
    return azb.load_arviz_data("non_centered_eight")


@pytest.fixture(scope="module")
def centered_eight_with_sigma(centered_eight):
    xr = importorskip("xarray")
    sigma_values = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])
    sigma_da = xr.DataArray(
        sigma_values, dims=["school"], coords={"school": centered_eight.observed_data.school.values}
    )
    data = centered_eight.copy()
    data["constant_data"] = data["constant_data"].to_dataset().assign(sigma=sigma_da)
    return data


@pytest.fixture(scope="module")
def multivariable_log_likelihood(centered_eight):
    xr = importorskip("xarray")
    centered_eight = centered_eight.copy()

    new_arr = xr.DataArray(
        np.zeros(centered_eight.log_likelihood["obs"].values.shape),
        dims=["chain", "draw", "school"],
        coords=centered_eight.log_likelihood.coords,
    )
    centered_eight.log_likelihood["decoy"] = new_arr
    return centered_eight


@pytest.fixture(name="simple_data", scope="session")
def fixture_simple_data():
    azb = importorskip("arviz_base")
    return azb.load_arviz_data("centered_eight")


@pytest.fixture
def simple_wrapper():
    from arviz_stats.loo.wrapper import SamplingWrapper

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
            rng = np.random.default_rng(42)
            return rng.normal(size=10)

    return SimpleWrapper()


@pytest.fixture
def sample_array():
    return np.array([1, 2, 3, 4, 5])


@pytest.fixture
def sample_dataarray():
    xr = importorskip("xarray")
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


@pytest.fixture(scope="module")
def log_likelihood_dataset(centered_eight):
    from arviz_stats.utils import get_log_likelihood_dataset

    return get_log_likelihood_dataset(centered_eight, var_names="obs")


@pytest.fixture(scope="module")
def elpd_data(centered_eight):
    from arviz_stats.utils import ELPDData, get_log_likelihood_dataset

    xr = importorskip("xarray")
    log_likelihood = get_log_likelihood_dataset(centered_eight, var_names="obs")["obs"]
    n_samples = log_likelihood.chain.size * log_likelihood.draw.size
    n_data_points = log_likelihood.school.size

    rng = np.random.default_rng(42)
    elpd_values = rng.normal(size=n_data_points)
    pareto_k_values = rng.uniform(0, 0.7, size=n_data_points)

    elpd_i = xr.DataArray(elpd_values, dims=["school"], coords={"school": log_likelihood.school})

    pareto_k = xr.DataArray(
        pareto_k_values, dims=["school"], coords={"school": log_likelihood.school}
    )

    mock_elpd = ELPDData(
        elpd=float(elpd_values.sum()),
        se=1.0,
        p=2.0,
        good_k=0.7,
        n_samples=n_samples,
        n_data_points=n_data_points,
        warning=False,
        elpd_i=elpd_i,
        pareto_k=pareto_k,
        scale="log",
        kind="loo",
    )
    return mock_elpd


@pytest.fixture(scope="module")
def log_densities(centered_eight):
    from arviz_stats.utils import get_log_likelihood_dataset

    xr = importorskip("xarray")
    log_lik = get_log_likelihood_dataset(centered_eight, var_names="obs")
    rng = np.random.default_rng(seed=42)

    p_values = rng.normal(size=(log_lik.chain.size, log_lik.draw.size))
    q_values = rng.normal(loc=-1.0, size=(log_lik.chain.size, log_lik.draw.size))

    log_p_da = xr.DataArray(
        p_values,
        dims=["chain", "draw"],
        coords={"chain": log_lik.chain, "draw": log_lik.draw},
    )
    log_q_da = xr.DataArray(
        q_values,
        dims=["chain", "draw"],
        coords={"chain": log_lik.chain, "draw": log_lik.draw},
    )

    log_p_np = p_values.ravel()
    log_q_np = q_values.ravel()

    return {
        "log_lik": log_lik,
        "dataarray": (log_p_da, log_q_da),
        "numpy": (log_p_np, log_q_np),
    }


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
    from arviz_stats.loo.wrapper import SamplingWrapper

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
            xr = importorskip("xarray")
            azb = importorskip("arviz_base")

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
            sp = importorskip("scipy")
            xr = importorskip("xarray")

            test_y = excluded_obs["y"]
            test_sigma = excluded_obs["sigma"]
            mu = idata__i.posterior["mu"].values.flatten()
            tau = idata__i.posterior["tau"].values.flatten()

            var_total = tau[:, np.newaxis] ** 2 + test_sigma**2
            log_lik = sp.stats.norm.logpdf(test_y, loc=mu[:, np.newaxis], scale=np.sqrt(var_total))

            dims = ["chain", "school", "draw"]
            coords = {"school": excluded_obs["indices"]}
            return xr.DataArray(log_lik.T[np.newaxis, :, :], dims=dims, coords=coords)

    return CenteredEightWrapper(centered_eight)


@pytest.fixture
def fresh_wrapper(mock_wrapper):
    mock_wrapper.fit_count = 0
    mock_wrapper.test_indices_history = []
    return mock_wrapper


@pytest.fixture
def mock_wrapper_reloo(non_centered_eight):
    from arviz_stats.loo.wrapper import SamplingWrapper

    class MockSamplingWrapper(SamplingWrapper):
        def __init__(self, model, idata_orig=None, **kwargs):
            super().__init__(model, idata_orig, **kwargs)
            self.data = idata_orig
            self.log_lik_var_name = "obs"

        def sel_observations(self, idx):
            obs_data = self.data.observed_data["obs"]
            all_indices = list(range(len(obs_data)))
            all_indices.remove(idx)
            modified_obs = obs_data.isel(school=all_indices)
            excluded_obs = obs_data.isel(school=idx)
            return modified_obs, excluded_obs

        def sample(self, modified_observed_data):
            return {"fit": "mock_fit", "data": modified_observed_data}

        def get_inference_data(self, fitted_model):
            return self.data

        def log_likelihood__i(self, excluded_obs, idata__i):
            posterior = idata__i.posterior
            theta = posterior["theta"]
            theta_i = theta.sel(school=excluded_obs.school.item())
            sigma = 12.5
            log_lik_values = (
                -0.5 * np.log(2 * np.pi * sigma**2)
                - 0.5 * ((excluded_obs.values - theta_i) / sigma) ** 2
            )
            log_lik = log_lik_values
            return log_lik

    return MockSamplingWrapper(model=None, idata_orig=non_centered_eight)


@pytest.fixture
def high_k_loo_data(non_centered_eight):
    from arviz_stats import loo
    from arviz_stats.utils import ELPDData

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
        log_weights=loo_data.log_weights.copy() if loo_data.log_weights is not None else None,
    )
    loo_data_modified.pareto_k.values[loo_data_modified.pareto_k.values > 0.7] = 0.6
    loo_data_modified.pareto_k.values[[0, 2, 5]] = [0.8, 0.9, 1.1]
    return loo_data_modified


@pytest.fixture
def mock_2d_data():
    azb = importorskip("arviz_base")
    rng = np.random.default_rng(42)

    n_chains = 4
    n_draws = 100
    n_schools = 3
    n_measurements = 4

    posterior = {
        "theta": rng.normal(0, 1, (n_chains, n_draws, n_schools, n_measurements)),
    }

    log_likelihood = {
        "log_lik": rng.normal(-1, 0.5, (n_chains, n_draws, n_schools, n_measurements)),
    }

    observed_data = {
        "y_obs": rng.normal(0, 1, (n_schools, n_measurements)),
    }

    return azb.from_dict(
        {
            "posterior": posterior,
            "log_likelihood": log_likelihood,
            "observed_data": observed_data,
        },
        dims={
            "theta": ["school", "measurement"],
            "log_lik": ["school", "measurement"],
            "y_obs": ["school", "measurement"],
        },
        coords={
            "school": [f"school_{i}" for i in range(n_schools)],
            "measurement": [f"meas_{i}" for i in range(n_measurements)],
        },
    )


@pytest.fixture
def mock_wrapper_2d(mock_2d_data):
    from arviz_stats.loo.wrapper import SamplingWrapper

    class MockSamplingWrapper2D(SamplingWrapper):
        def __init__(self, model, idata_orig=None, **kwargs):
            super().__init__(model, idata_orig, **kwargs)
            self.data = idata_orig
            self.log_lik_var_name = "log_lik"

        def sel_observations(self, idx):
            obs_data = self.data.observed_data["y_obs"]

            if isinstance(idx, dict):
                xr = importorskip("xarray")
                mask = xr.ones_like(obs_data, dtype=bool)
                mask.loc[idx] = False
                modified_obs = obs_data.where(mask, drop=True)
                excluded_obs = obs_data.loc[idx]
            else:
                modified_obs = obs_data
                excluded_obs = obs_data

            return modified_obs, excluded_obs

        def sample(self, modified_observed_data):
            return {"fit": "mock_fit_2d", "data": modified_observed_data}

        def get_inference_data(self, fitted_model):
            return self.data

        def log_likelihood__i(self, excluded_obs, idata__i):
            posterior = idata__i.posterior
            xr = importorskip("xarray")
            rng = np.random.default_rng(42)
            return xr.DataArray(
                rng.normal(-1, 0.1, size=(posterior.dims["chain"], posterior.dims["draw"])),
                dims=["chain", "draw"],
                coords={"chain": posterior.chain, "draw": posterior.draw},
            )

    return MockSamplingWrapper2D(model=None, idata_orig=mock_2d_data)
