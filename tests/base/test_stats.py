# pylint: disable=redefined-outer-name, no-member
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy.stats import bernoulli, norm, poisson

from ..helpers import importorskip

azb = importorskip("arviz_base")
xr = importorskip("xarray")


@pytest.fixture(scope="module")
def centered_eight():
    centered_eight = azb.load_arviz_data("centered_eight")
    return centered_eight


@pytest.fixture(scope="module")
def non_centered_eight():
    non_centered_eight = azb.load_arviz_data("non_centered_eight")
    return non_centered_eight


@pytest.fixture(scope="module")
def multivariable_log_likelihood(centered_eight):
    centered_eight = centered_eight.copy()
    new_arr = xr.DataArray(
        np.zeros(centered_eight.log_likelihood["obs"].values.shape),
        dims=["chain", "draw", "school"],
        coords=centered_eight.log_likelihood.coords,
    )
    centered_eight.log_likelihood["decoy"] = new_arr
    return centered_eight


@pytest.mark.parametrize(
    "kde_kwargs",
    [
        {},
        {"adaptive": True},
        {"circular": True},
    ],
    ids=["default", "adaptive", "circular"],
)
@pytest.mark.parametrize("bound_correction", [True, False])
def test_kde_is_normalized(bound_correction, kde_kwargs):
    if "circular" in kde_kwargs and bound_correction:
        pytest.skip("Bound correction not implemented for circular kde")
    rng = np.random.default_rng(43)
    if kde_kwargs.get("circular", False):
        data = rng.vonmises(np.pi, 1, (1_000, 100))
    else:
        data = rng.normal(size=(1_000, 100))
        kde_kwargs["bound_correction"] = bound_correction
    sample = azb.ndarray_to_dataarray(data, "x", sample_dims=["sample"])
    kde = sample.azstats.kde(dim="sample", **kde_kwargs)
    dx = kde.sel(plot_axis="x").diff(dim="kde_dim")
    density_norm = kde.sel(plot_axis="y").sum(dim="kde_dim") * dx
    assert_array_almost_equal(density_norm, 1, 6)


@pytest.mark.parametrize("func", ("eti", "hdi", "kde", "histogram"))
def test_extra_kwargs_raise(centered_eight, func):
    accessor = centered_eight.posterior.ds.azstats
    with pytest.raises(TypeError, match=".*unexpected keyword argument.*"):
        # 'dims' is not valid, to match xarray behaviour we use 'dim' only
        getattr(accessor, func)(dims="draw")


def test_hdi_idata(centered_eight):
    accessor = centered_eight.posterior.ds.azstats
    result = accessor.hdi()
    assert isinstance(result, xr.Dataset)
    assert result.sizes == {"school": 8, "ci_bound": 2}

    result = accessor.hdi(dim="chain")
    assert isinstance(result, xr.Dataset)
    assert result.sizes == {"draw": 500, "ci_bound": 2, "school": 8}


def test_hdi_idata_varnames(centered_eight):
    accessor = centered_eight.posterior.ds.azstats
    result = accessor.filter_vars(var_names=["mu", "theta"]).hdi()
    assert isinstance(result, xr.Dataset)
    assert result.sizes == {"ci_bound": 2, "school": 8}
    assert list(result.data_vars.keys()) == ["mu", "theta"]


# @pytest.mark.skip(reason="AttributeError: DatasetView objects are not to be initialized directly")
def test_hdi_idata_group(centered_eight):
    result_posterior = centered_eight.azstats.hdi(group="posterior")
    result_prior = centered_eight.azstats.hdi(group="prior")
    assert "ci_bound" in result_prior.mu.dims
    range_posterior = result_posterior.mu.values[1] - result_posterior.mu.values[0]
    range_prior = result_prior.mu.values[1] - result_prior.mu.values[0]
    assert range_posterior < range_prior


def test_hdi_coords(centered_eight):
    data = centered_eight.posterior.sel({"chain": [0, 1, 3]}).ds
    result = data.azstats.hdi(dim="draw")
    assert_array_equal(result.coords["chain"], [0, 1, 3])


@pytest.mark.parametrize("prob", [0.56, 0.83])
@pytest.mark.parametrize("nearest", [True, False])
def test_hdi_multimodal_continuous(prob, nearest):
    method = "multimodal_sample" if nearest else "multimodal"
    rng = np.random.default_rng(43)
    dist1 = norm(loc=-30, scale=0.5)
    dist2 = norm(loc=30, scale=0.5)
    normal_sample = azb.ndarray_to_dataarray(
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
    intervals = normal_sample.azstats.hdi(dim="sample", method=method, prob=prob)
    assert_array_almost_equal(intervals, exact_hdis, 1)

    if nearest:
        assert np.all(np.isin(intervals, normal_sample))
    else:
        assert not np.any(np.isin(intervals, normal_sample))


@pytest.mark.parametrize("prob", [0.56, 0.83])
def test_hdi_multimodal_discrete(prob):
    rng = np.random.default_rng(43)
    dist1 = poisson(10)
    dist2 = poisson(100)
    x = np.concatenate([dist1.rvs(2500000, random_state=rng), dist2.rvs(2500000, random_state=rng)])
    sample = azb.ndarray_to_dataarray(x, "x", sample_dims=["sample"])
    intervals = sample.azstats.hdi(dim="sample", method="multimodal", prob=prob)
    assert intervals.sizes["mode"] == 2
    lower_mode = intervals.sel(mode=0)
    higher_mode = intervals.sel(mode=1)
    assert lower_mode[0] <= 10 <= lower_mode[1]
    assert higher_mode[0] <= 100 <= higher_mode[1]

    # restrict the bins to a range in which only a single mode will appear
    bins = np.arange(0, 20) - 0.5
    intervals = sample.azstats.hdi(dim="sample", method="multimodal", prob=prob, bins=bins)
    assert intervals.sizes["mode"] == 1
    assert intervals.sel(mode=0)[0] <= 10 <= intervals.sel(mode=0)[1]


@pytest.mark.parametrize("prob", [0.56, 0.83])
@pytest.mark.parametrize("dist", [poisson(10), bernoulli(0.7)])
def test_hdi_multimodal_unimodal_discrete_consistent(dist, prob):
    rng = np.random.default_rng(43)
    x = dist.rvs(size=1_000, random_state=rng)
    sample = azb.ndarray_to_dataarray(x, "x", sample_dims=["sample"])
    intervals = sample.azstats.hdi(dim="sample", method="multimodal", prob=prob)
    intervals_unimodal = sample.azstats.hdi(dim="sample", method="nearest", prob=prob)
    intervals = intervals.squeeze("mode")
    assert_array_equal(intervals, intervals_unimodal)


def test_hdi_multimodal_multivars():
    size = 2500000
    var1 = np.concatenate((np.random.normal(-4, 1, size), np.random.normal(2, 0.5, size)))
    var2 = np.random.normal(8, 1, size * 2)
    sample = xr.Dataset(
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
    sample = azb.ndarray_to_dataarray(x, "x", sample_dims=["sample"])
    intervals = sample.azstats.hdi(dim="sample", method="multimodal", prob=0.9)
    assert intervals.sizes["mode"] == 2
    with pytest.warns(UserWarning, match="found more modes"):
        intervals2 = sample.azstats.hdi(dim="sample", method="multimodal", prob=0.9, max_modes=1)
    assert intervals2.sizes["mode"] == 1
    assert intervals2.equals(intervals.isel(mode=[1]))


@pytest.mark.parametrize("nearest", [True, False])
def test_hdi_multimodal_circular(nearest):
    rng = np.random.default_rng(43)
    normal_sample = azb.ndarray_to_dataarray(
        np.concatenate(
            [
                rng.vonmises(np.pi, 10, 2500000),
                rng.vonmises(np.pi / 2, 10, 2500000),
            ]
        ),
        "x",
        sample_dims=["sample"],
    )
    method = "multimodal_sample" if nearest else "multimodal"
    interval = normal_sample.azstats.hdi(circular=True, method=method, prob=0.83, dim="sample")
    assert interval.sizes["mode"] == 2
    assert interval.sel(mode=0)[0] <= np.pi / 2 <= interval.sel(mode=0)[1]
    assert interval.sel(mode=1)[0] <= np.pi and interval.sel(mode=1)[1] >= -np.pi


def test_hdi_circular():
    rng = np.random.default_rng(43)
    normal_sample = azb.ndarray_to_dataarray(
        rng.vonmises(np.pi, 1, 5000000), "x", sample_dims=["sample"]
    )
    interval = normal_sample.azstats.hdi(circular=True, prob=0.83, dim="sample")
    assert_array_almost_equal(interval, [1.3, -1.4], 1)
    interval_multi = normal_sample.azstats.hdi(
        circular=True, prob=0.83, dim="sample", method="multimodal"
    )
    assert_array_almost_equal(interval_multi, [[1.3, -1.4]], 1)


def test_hdi_bad_ci():
    rng = np.random.default_rng(43)
    normal_sample = azb.ndarray_to_dataarray(rng.normal(size=50), "x", sample_dims=["sample"])
    with pytest.raises(ValueError):
        normal_sample.azstats.hdi(prob=2, dim="sample")


def test_hdi_skipna():
    rng = np.random.default_rng(43)
    with azb.rc_context(rc={"data.sample_dims": ["sample"]}):
        normal_sample = azb.ndarray_to_dataarray(rng.normal(size=500), "x")
        interval = normal_sample.sel(sample=slice(10, None)).azstats.hdi()
        normal_sample.loc[{"sample": slice(None, 10)}] = np.nan
        interval_ = normal_sample.azstats.hdi(skipna=True)
    assert_array_almost_equal(interval, interval_)


def test_ecdf(centered_eight):
    accessor = centered_eight.posterior.ds.azstats
    result = accessor.ecdf()
    assert isinstance(result, xr.Dataset)
    assert result.sizes == {"plot_axis": 2, "quantile": 200}


def test_ecdf_idata_varnames(centered_eight):
    accessor = centered_eight.posterior.ds.azstats
    result = accessor.filter_vars(var_names=["mu", "theta"]).ecdf()
    assert isinstance(result, xr.Dataset)
    assert result.sizes == {"plot_axis": 2, "quantile": 200}
    assert list(result.data_vars.keys()) == ["mu", "theta"]


def test_ecdf_coords(centered_eight):
    data = centered_eight.posterior.sel({"chain": [0, 1, 3]}).ds
    result = data.azstats.ecdf(dim="draw")
    assert_array_equal(result.coords["chain"], [0, 1, 3])
