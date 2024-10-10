# pylint: disable=redefined-outer-name, no-member
import numpy as np
import pytest
from arviz_base import load_arviz_data, ndarray_to_dataarray, rc_context
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy.stats import bernoulli, norm, poisson
from xarray import DataArray, Dataset


@pytest.fixture(scope="module")
def centered_eight():
    centered_eight = load_arviz_data("centered_eight")
    return centered_eight


@pytest.fixture(scope="module")
def non_centered_eight():
    non_centered_eight = load_arviz_data("non_centered_eight")
    return non_centered_eight


@pytest.fixture(scope="module")
def multivariable_log_likelihood(centered_eight):
    centered_eight = centered_eight.copy()
    new_arr = DataArray(
        np.zeros(centered_eight.log_likelihood["obs"].values.shape),
        dims=["chain", "draw", "school"],
        coords=centered_eight.log_likelihood.coords,
    )
    centered_eight.log_likelihood["decoy"] = new_arr
    return centered_eight


def test_hdi_idata(centered_eight):
    accessor = centered_eight.posterior.ds.azstats
    result = accessor.hdi()
    assert isinstance(result, Dataset)
    assert result.sizes == {"school": 8, "hdi": 2}

    result = accessor.hdi(dims="chain")
    assert isinstance(result, Dataset)
    assert result.sizes == {"draw": 500, "hdi": 2, "school": 8}


def test_hdi_idata_varnames(centered_eight):
    accessor = centered_eight.posterior.ds.azstats
    result = accessor.filter_vars(var_names=["mu", "theta"]).hdi()
    assert isinstance(result, Dataset)
    assert result.sizes == {"hdi": 2, "school": 8}
    assert list(result.data_vars.keys()) == ["mu", "theta"]


@pytest.mark.skip(reason="AttributeError: DatasetView objects are not to be initialized directly")
def test_hdi_idata_group(centered_eight):
    result_posterior = centered_eight.azstats.hdi(group="posterior")
    result_prior = centered_eight.azstats.hdi(group="prior")
    assert "hdi" in result_prior.mu.dims
    range_posterior = result_posterior.mu.values[1] - result_posterior.mu.values[0]
    range_prior = result_prior.mu.values[1] - result_prior.mu.values[0]
    assert range_posterior < range_prior


def test_hdi_coords(centered_eight):
    data = centered_eight.posterior.sel({"chain": [0, 1, 3]}).ds
    result = data.azstats.hdi(dims="draw")
    assert_array_equal(result.coords["chain"], [0, 1, 3])


@pytest.mark.parametrize("prob", [0.56, 0.83])
def test_hdi_multimodal_continuous(prob):
    rng = np.random.default_rng(43)
    dist1 = norm(loc=-30, scale=0.5)
    dist2 = norm(loc=30, scale=0.5)
    normal_sample = ndarray_to_dataarray(
        np.concatenate(
            (dist1.rvs(2500000, random_state=rng), dist2.rvs(2500000, random_state=rng))
        ),
        "x",
        sample_dims=["sample"],
    )
    exact_hdis = np.concatenate(
        [
            np.array(dist1.interval(prob))[np.newaxis, :],
            np.array(dist2.interval(prob))[np.newaxis, :],
        ],
        axis=0,
    )
    intervals = normal_sample.azstats.hdi(dims="sample", method="multimodal", prob=prob)
    assert_array_almost_equal(intervals, exact_hdis, 1)


@pytest.mark.parametrize("prob", [0.56, 0.83])
@pytest.mark.parametrize("dist", [poisson(10), bernoulli(0.7)])
def test_hdi_multimodal_unimodal_discrete_consistent(dist, prob):
    rng = np.random.default_rng(43)
    x = dist.rvs(size=1_000, random_state=rng)
    sample = ndarray_to_dataarray(x, "x", sample_dims=["sample"])
    intervals = sample.azstats.hdi(dims="sample", method="multimodal", prob=prob)
    intervals_unimodal = sample.azstats.hdi(dims="sample", method="nearest", prob=prob)
    intervals = intervals.squeeze("mode")
    assert_array_equal(intervals, intervals_unimodal)


def test_hdi_multimodal_multivars():
    size = 2500000
    var1 = np.concatenate((np.random.normal(-4, 1, size), np.random.normal(2, 0.5, size)))
    var2 = np.random.normal(8, 1, size * 2)
    sample = Dataset(
        {
            "var1": (("chain", "draw"), var1[np.newaxis, :]),
            "var2": (("chain", "draw"), var2[np.newaxis, :]),
        },
        coords={"chain": [0], "draw": np.arange(size * 2)},
    )
    intervals = sample.azstats.hdi(method="multimodal", prob=0.83)
    assert_array_almost_equal(intervals.var1, [[-5.2, -2.8], [1.2, 2.8]], 1)
    assert_array_almost_equal(intervals.var2, [[6.6, 9.4]], 1)
    assert "var1_mode" in intervals.var1.dims
    assert "var2_mode" in intervals.var2.dims


def test_hdi_multimodal_max_modes():
    rng = np.random.default_rng(42)
    x = np.concatenate([rng.normal(0, 1, 250_000), rng.normal(30, 1, 2_500_000)])
    sample = ndarray_to_dataarray(x, "x", sample_dims=["sample"])
    intervals = sample.azstats.hdi(dims="sample", method="multimodal", prob=0.9)
    assert intervals.sizes["mode"] == 2
    intervals2 = sample.azstats.hdi(dims="sample", method="multimodal", prob=0.9, max_modes=1)
    assert intervals2.sizes["mode"] == 1
    assert intervals2.equals(intervals.isel(mode=[1]))


def test_hdi_circular():
    rng = np.random.default_rng(43)
    normal_sample = ndarray_to_dataarray(
        rng.vonmises(np.pi, 1, 5000000), "x", sample_dims=["sample"]
    )
    interval = normal_sample.azstats.hdi(circular=True, prob=0.83, dims="sample")
    assert_array_almost_equal(interval, [1.3, -1.4], 1)
    interval_multi = normal_sample.azstats.hdi(
        circular=True, prob=0.83, dims="sample", method="multimodal"
    )
    assert_array_almost_equal(interval_multi, [[1.3, -1.4]], 1)


def test_hdi_bad_ci():
    rng = np.random.default_rng(43)
    normal_sample = ndarray_to_dataarray(rng.normal(size=50), "x", sample_dims=["sample"])
    with pytest.raises(ValueError):
        normal_sample.azstats.hdi(prob=2, dims="sample")


def test_hdi_skipna():
    rng = np.random.default_rng(43)
    with rc_context(rc={"data.sample_dims": ["sample"]}):
        normal_sample = ndarray_to_dataarray(rng.normal(size=500), "x")
        interval = normal_sample.sel(sample=slice(10, None)).azstats.hdi()
        normal_sample.loc[{"sample": slice(None, 10)}] = np.nan
        interval_ = normal_sample.azstats.hdi(skipna=True)
    assert_array_almost_equal(interval, interval_)


def test_ecdf(centered_eight):
    accessor = centered_eight.posterior.ds.azstats
    result = accessor.ecdf()
    assert isinstance(result, Dataset)
    assert result.sizes == {"plot_axis": 2, "quantile": 200}

    result = accessor.hdi(dims="chain")
    assert isinstance(result, Dataset)
    assert result.sizes == {"draw": 500, "hdi": 2, "school": 8}


def test_ecdf_idata_varnames(centered_eight):
    accessor = centered_eight.posterior.ds.azstats
    result = accessor.filter_vars(var_names=["mu", "theta"]).ecdf()
    assert isinstance(result, Dataset)
    assert result.sizes == {"plot_axis": 2, "quantile": 200}
    assert list(result.data_vars.keys()) == ["mu", "theta"]


def test_ecdf_coords(centered_eight):
    data = centered_eight.posterior.sel({"chain": [0, 1, 3]}).ds
    result = data.azstats.ecdf(dims="draw")
    assert_array_equal(result.coords["chain"], [0, 1, 3])
