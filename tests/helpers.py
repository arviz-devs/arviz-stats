import numpy as np
import pytest

from arviz_base import from_dict

def create_model(seed=10, transpose=False):
    """Create model with fake data."""
    rng = np.random.default_rng(seed)
    nchains = 4
    ndraws = 500
    data = {
        "J": 8,
        "y": np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]),
        "sigma": np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]),
    }
    posterior = {
        "mu": rng.random((nchains, ndraws)),
        "tau": abs(rng.random((nchains, ndraws))),
        "eta": rng.random((nchains, ndraws, data["J"])),
        "theta": rng.random((nchains, ndraws, data["J"])),
    }
    posterior_predictive = {"y": rng.random((nchains, ndraws, len(data["y"])))}
    sample_stats = {
        "energy": rng.random((nchains, ndraws)),
        "diverging": rng.random((nchains, ndraws)) > 0.90,
        "max_depth": rng.random((nchains, ndraws)) > 0.90,
    }
    log_likelihood = {
        "y": rng.random((nchains, ndraws, data["J"])),
    }
    prior = {
        "mu": rng.random((nchains, ndraws)) / 2,
        "tau": abs(rng.random((nchains, ndraws))) / 2,
        "eta": rng.random((nchains, ndraws, data["J"])) / 2,
        "theta": rng.random((nchains, ndraws, data["J"])) / 2,
    }
    prior_predictive = {"y": rng.random((nchains, ndraws, len(data["y"]))) / 2}
    sample_stats_prior = {
        "energy": rng.random((nchains, ndraws)),
        "diverging": (rng.random((nchains, ndraws)) > 0.90).astype(int),
    }
    model = from_dict(
        {
            "posterior": posterior,
            "posterior_predictive": posterior_predictive,
            "sample_stats": sample_stats,
            "log_likelihood": log_likelihood,
            "prior": prior,
            "prior_predictive": prior_predictive,
            "sample_stats_prior": sample_stats_prior,
            "observed_data": {"y": data["y"]},
        },
        dims={
            "y": ["obs_dim"],
            "log_likelihood": ["obs_dim"],
            "theta": ["school"],
            "eta": ["school"],
        },
        coords={"obs_dim": range(data["J"])},
    )
    if transpose:
        for group, group_dataset in model.children.items():
            if all(dim in group_dataset.dims for dim in ("draw", "chain")):
                model[group] = group_dataset.transpose("draw", "chain", ...)
    return model


def create_multidimensional_model(seed=10, transpose=False):
    """Create model with fake data."""
    rng = np.random.default_rng(seed)
    nchains = 4
    ndraws = 500
    ndim1 = 5
    ndim2 = 7
    y = rng.normal(size=(ndim1, ndim2))
    posterior = {
        "mu": rng.standard_normal((nchains, ndraws)),
        "tau": abs(rng.standard_normal((nchains, ndraws))),
        "eta": rng.standard_normal((nchains, ndraws, ndim1, ndim2)),
        "theta": rng.standard_normal((nchains, ndraws, ndim1, ndim2)),
    }
    posterior_predictive = {"y": rng.standard_normal((nchains, ndraws, ndim1, ndim2))}
    sample_stats = {
        "energy": rng.standard_normal((nchains, ndraws)),
        "diverging": rng.standard_normal((nchains, ndraws)) > 0.90,
    }
    log_likelihood = {
        "y": rng.standard_normal((nchains, ndraws, ndim1, ndim2)),
    }
    prior = {
        "mu": rng.standard_normal((nchains, ndraws)) / 2,
        "tau": abs(rng.standard_normal((nchains, ndraws))) / 2,
        "eta": rng.standard_normal((nchains, ndraws, ndim1, ndim2)) / 2,
        "theta": rng.standard_normal((nchains, ndraws, ndim1, ndim2)) / 2,
    }
    prior_predictive = {"y": rng.standard_normal((nchains, ndraws, ndim1, ndim2)) / 2}
    sample_stats_prior = {
        "energy": rng.standard_normal((nchains, ndraws)),
        "diverging": (rng.standard_normal((nchains, ndraws)) > 0.95).astype(int),
    }
    model = from_dict(
        {
            "posterior": posterior,
            "posterior_predictive": posterior_predictive,
            "sample_stats": sample_stats,
            "log_likelihood": log_likelihood,
            "prior": prior,
            "prior_predictive": prior_predictive,
            "sample_stats_prior": sample_stats_prior,
            "observed_data": {"y": y},
        },
        dims={"y": ["dim1", "dim2"], "log_likelihood": ["dim1", "dim2"]},
        coords={"dim1": range(ndim1), "dim2": range(ndim2)},
    )
    if transpose:
        for group, group_dataset in model.children.items():
            if all(dim in group_dataset.dims for dim in ("draw", "chain")):
                model[group] = group_dataset.transpose("draw", "chain", ...)
    return model


def create_data_random(groups=None, seed=10):
    """Create InferenceData object using random data."""
    if groups is None:
        groups = ["posterior", "sample_stats", "observed_data", "posterior_predictive"]
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(4, 500, 8))
    idata_dict = {
        "posterior": {"a": data[..., 0], "b": data},
        "sample_stats": {"a": data[..., 0], "b": data},
        "observed_data": {"b": data[0, 0, :]},
        "posterior_predictive": {"a": data[..., 0], "b": data},
        "prior": {"a": data[..., 0], "b": data},
        "prior_predictive": {"a": data[..., 0], "b": data},
        "warmup_posterior": {"a": data[..., 0], "b": data},
        "warmup_posterior_predictive": {"a": data[..., 0], "b": data},
        "warmup_prior": {"a": data[..., 0], "b": data},
    }
    idata = from_dict(
        {group: ary for group, ary in idata_dict.items() if group in groups}, save_warmup=True
    )
    return idata


@pytest.fixture()
def data_random():
    """Fixture containing InferenceData object using random data."""
    idata = create_data_random()
    return idata


@pytest.fixture(scope="module")
def models():
    """Fixture containing 2 mock inference data instances for testing."""
    # blank line to keep black and pydocstyle happy

    class Models:
        model_1 = create_model(seed=10)
        model_2 = create_model(seed=11, transpose=True)

    return Models()


@pytest.fixture(scope="module")
def multidim_models():
    """Fixture containing 2 mock inference data instances with multidimensional data for testing."""
    # blank line to keep black and pydocstyle happy

    class Models:
        model_1 = create_multidimensional_model(seed=10)
        model_2 = create_multidimensional_model(seed=11, transpose=True)

    return Models()
