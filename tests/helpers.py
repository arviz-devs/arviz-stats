"""Test related helper functions."""

import os
import sys
import warnings
from typing import Any

import numpy as np
import pytest


def importorskip(modname: str, reason: str | None = None) -> Any:
    """Import and return the requested module ``modname``.

    Doesn't allow skips when ``ARVIZ_REQUIRE_ALL_DEPS`` env var is defined.
    Borrowed and modified from ``pytest.importorskip``.

    Parameters
    ----------
    modname : str
        the name of the module to import
    reason : str, optional
        this reason is shown as skip message when the module cannot be imported.
    """
    __tracebackhide__ = True  # pylint: disable=unused-variable
    compile(modname, "", "eval")  # to catch syntaxerrors

    with warnings.catch_warnings():
        # Make sure to ignore ImportWarnings that might happen because
        # of existing directories with the same name we're trying to
        # import but without a __init__.py file.
        warnings.simplefilter("ignore")
        try:
            __import__(modname)
        except ImportError as exc:
            if "ARVIZ_REQUIRE_ALL_DEPS" in os.environ:
                raise exc
            if reason is None:
                reason = f"could not import {modname!r}: {exc}"
            pytest.skip(reason, allow_module_level=True)

    mod = sys.modules[modname]
    return mod


def create_model(seed=10, transpose=False):
    """Create model with fake data."""
    from arviz_base import from_dict

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
    from arviz_base import from_dict

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
    from arviz_base import from_dict

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


def create_binary_data():
    from arviz_base import from_dict

    y_true = np.array([1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0])
    y_pred = np.array(
        [
            [
                [
                    1,
                    0,
                    1,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                ],
                [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0],
            ]
        ]
    )

    log_likelihood = np.array(
        [
            [
                -0.07947292,
                -0.00673689,
                -0.03548443,
                -0.0734388,
                -0.01053072,
                -1.92504819,
                -0.04810741,
                -0.85600871,
                -0.04810741,
                -0.14321535,
                -0.05885898,
                -0.10014122,
                -0.04810741,
                -0.05321948,
                -0.06293666,
                -0.64476814,
                -0.13867731,
                -0.00251631,
                -0.16804357,
                -0.0734388,
                -0.6068221,
                -0.01797324,
            ],
            [
                -0.07947292,
                -0.00673689,
                -0.03548443,
                -0.0734388,
                -0.01053072,
                -1.92504819,
                -0.04810741,
                -0.85600871,
                -0.04810741,
                -0.14321535,
                -0.05885898,
                -0.10014122,
                -0.04810741,
                -0.05321948,
                -0.06293666,
                -0.64476814,
                -0.13867731,
                -0.00251631,
                -0.16804357,
                -0.0734388,
                -0.6068221,
                -0.01797324,
            ],
            [
                -0.04076453,
                -0.02625018,
                -0.01322834,
                -0.09253384,
                -0.03323738,
                -2.28945093,
                -0.02020827,
                -1.02274493,
                -0.02020827,
                -0.09329038,
                -0.0267789,
                -0.05638643,
                -0.02020827,
                -0.02326542,
                -0.02940704,
                -0.54261699,
                -0.08915241,
                -0.01558169,
                -0.11687097,
                -0.09253384,
                -0.68570614,
                -0.04406399,
            ],
        ]
    )
    return from_dict(
        {
            "observed_data": {"y": y_true},
            "posterior_predictive": {"y": y_pred},
            "log_likelihood": {"y": log_likelihood},
        }
    )


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
