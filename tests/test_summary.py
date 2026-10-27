"""Test summary."""

# pylint: disable=redefined-outer-name,protected-access
from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from .helpers import importorskip

azb = importorskip("arviz_base")
pd = importorskip("pandas")
xr = importorskip("xarray")

from arviz_stats import ci_in_rope, eti, hdi, iqr, mad, mean, median, mode, qds, std, summary, var
from arviz_stats.base.stats_utils import round_num
from arviz_stats.summary import SummaryDataFrame, _build_fmt_map

POINT_STATS = {
    "mean": mean,
    "median": median,
    "mode": mode,
    "std": std,
    "var": var,
    "mad": mad,
    "iqr": iqr,
}

METRICS_NAMES = [
    "mean",
    "sd",
    "eti89_lb",
    "eti89_ub",
    "ess_bulk",
    "ess_tail",
    "r_hat",
    "mcse_mean",
    "mcse_sd",
    "median",
    "mad",
    "eti89_lb",
    "eti89_ub",
    "ess_median",
    "ess_tail",
    "r_hat",
    "mcse_median",
    "mcse_mean",
    "ess_mean",
    "min_ss",
]


@pytest.fixture(scope="module")
def mixed_datatree():
    rng = np.random.default_rng(42)
    return azb.from_dict(
        {
            "posterior": {
                "mu": rng.normal(loc=1.5, scale=0.5, size=(4, 100)),
                "theta": rng.normal(loc=0.0, scale=1.0, size=(4, 100, 3)),
                "k": rng.integers(0, 6, size=(4, 100)),
                "z": rng.choice([True, False], size=(4, 100), p=[0.7, 0.3]),
            },
            "prior": {
                "mu": rng.normal(loc=0.0, scale=2.0, size=(4, 100)),
                "theta": rng.normal(loc=0.0, scale=2.0, size=(4, 100, 3)),
                "k": rng.integers(0, 10, size=(4, 100)),
                "z": rng.choice([True, False], size=(4, 100)),
            },
        },
        dims={"theta": ["hierarchy"]},
    )


def _flat_values(obj):
    if isinstance(obj, xr.Dataset):
        obj = obj.to_dataarray()
    return np.asarray(obj.values, dtype=float).ravel()


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "kind,expected_cols",
    (
        ("all", METRICS_NAMES[:9]),
        ("stats", METRICS_NAMES[:4]),
        ("diagnostics", METRICS_NAMES[4:9]),
        ("all_median", METRICS_NAMES[9:17]),
        ("stats_median", METRICS_NAMES[9:13]),
        ("diagnostics_median", METRICS_NAMES[13:17]),
        ("mc_diagnostics", METRICS_NAMES[17:]),
    ),
)
def test_summary_mixed_datatree_kinds(mixed_datatree, kind, expected_cols):
    result = summary(mixed_datatree, kind=kind)
    assert isinstance(result, SummaryDataFrame)
    assert_array_equal(result.columns, expected_cols)
    expected_rows = ["mu", "theta[0]", "theta[1]", "theta[2]", "k", "z"]
    assert list(result.index) == expected_rows
    display = result._display_df()
    for row in expected_rows:
        assert all(isinstance(v, str) for v in display.loc[row])


@pytest.mark.parametrize(
    "kind",
    [
        "all",
        "stats",
        "diagnostics",
        "all_median",
        "stats_median",
        "diagnostics_median",
        "mc_diagnostics",
    ],
)
def test_summary_bool(kind):
    rng = np.random.default_rng(0)
    indicator = rng.uniform(size=(4, 100)) < 0.3
    summary_bool = summary(
        azb.from_dict({"posterior": {"x": indicator}}), kind=kind, round_to="none"
    )
    summary_float = summary(
        azb.from_dict({"posterior": {"x": indicator.astype(float)}}), kind=kind, round_to="none"
    )
    assert_allclose(summary_bool.to_numpy(dtype=float), summary_float.to_numpy(dtype=float))


@pytest.mark.parametrize("fmt", ["wide", "long", "xarray"])
def test_summary_mixed_datatree_formats(mixed_datatree, fmt):
    result = summary(mixed_datatree, fmt=fmt)
    if fmt == "wide":
        assert isinstance(result, SummaryDataFrame)
        assert {"mu", "theta[0]", "k", "z"}.issubset(result.index)
        assert "mean" in result.columns
        assert 0.0 <= result.loc["z", "mean"] <= 1.0
    elif fmt == "long":
        assert isinstance(result, SummaryDataFrame)
        assert "mean" in result.index
        assert {"mu", "theta", "k", "z"}.issubset(result.columns)
    else:
        assert isinstance(result, xr.Dataset)
        assert "summary" in result.dims
        assert set(result.data_vars) == {"mu", "theta", "k", "z"}


@pytest.mark.parametrize("stat_name", list(POINT_STATS))
def test_point_stats_mixed_datatree_and_inputs(mixed_datatree, stat_name):
    stat = POINT_STATS[stat_name]
    res_dt = stat(mixed_datatree)
    assert set(res_dt.data_vars) == {"mu", "theta", "k", "z"}
    assert res_dt["mu"].shape == ()
    assert res_dt["theta"].shape == (3,)
    assert res_dt["k"].shape == ()
    assert res_dt["z"].shape == ()

    res_ds = stat(mixed_datatree.posterior.ds, round_to=None)
    assert set(res_ds.data_vars) == {"mu", "theta", "k", "z"}

    res_da = stat(mixed_datatree.posterior["z"], round_to=2)
    assert res_da.shape == ()

    res_arr = stat(mixed_datatree.posterior["z"].values)
    assert np.shape(res_arr) == ()


def test_intervals_and_rope_mixed_datatree(mixed_datatree):
    hdi_res = hdi(mixed_datatree, prob=0.89)
    eti_res = eti(mixed_datatree, prob=0.89)
    qds_res = qds(mixed_datatree, nquantiles=20)
    rope_res = ci_in_rope(mixed_datatree, rope=(-0.5, 0.5))

    for res in (hdi_res, eti_res):
        assert set(res.data_vars) == {"mu", "theta", "k", "z"}
        assert res["mu"].shape == (2,)
        assert res["theta"].shape == (3, 2)
        assert res["k"].shape == (2,)
        assert res["z"].shape == (2,)
        assert "ci_bound" in res.dims

    assert set(qds_res.data_vars) == {"mu", "theta", "k", "z"}
    assert qds_res["mu"].shape == (2, 20)
    assert qds_res["theta"].shape == (2, 3, 20)
    assert "plot_axis" in qds_res.dims

    assert set(rope_res.data_vars) == {"mu", "theta", "k", "z"}
    for var_name in rope_res.data_vars:
        vals = np.asarray(rope_res[var_name].values, dtype=float)
        assert np.all((vals >= 0) & (vals <= 100))


def test_point_stats_options(mixed_datatree):
    res_prior = mean(mixed_datatree, var_names=["mu"], group="prior")
    res_post = mean(mixed_datatree, var_names=["mu"], group="posterior")
    assert float(res_prior["mu"]) != float(res_post["mu"])

    res_dim = median(mixed_datatree, var_names=["mu"], dim="chain")
    assert dict(res_dim["mu"].sizes) == {"draw": 100}

    res_coords = std(mixed_datatree, var_names=["theta"], coords={"hierarchy": [0, 2]})
    assert res_coords["theta"].shape == (2,)

    res_like = var(mixed_datatree, var_names=["th"], filter_vars="like")
    assert set(res_like.data_vars) == {"theta"}

    res_regex = mad(mixed_datatree, var_names=["^m"], filter_vars="regex")
    assert set(res_regex.data_vars) == {"mu"}

    grouped = mixed_datatree.posterior.ds.groupby("chain")
    res_group = mean(grouped, var_names=["mu"])
    assert dict(res_group.sizes) == {"chain": 4}

    assert np.shape(mean([1.0, 2.0, 3.0])) == ()


@pytest.mark.parametrize("round_to", [0, 2, "2g", "none", None])
def test_point_stats_round_to(mixed_datatree, round_to):
    da = mixed_datatree.posterior["mu"]
    raw = mean(da, round_to="none")
    if round_to == "none":
        expected = raw
    else:
        precision = azb.rcParams["stats.round_to"] if round_to is None else round_to
        expected = round_num(raw.item(), precision)
    assert_allclose(mean(da, round_to=round_to), expected)


def test_point_stats_skipna(mixed_datatree):
    da = mixed_datatree.posterior["mu"].copy()
    da.values[0, 0] = np.nan
    for stat in (mean, mode, iqr):
        assert np.isfinite(stat(da, round_to="none", skipna=True).item())
        assert np.isnan(stat(da, round_to="none", skipna=False).item())


def test_iqr_quantiles(mixed_datatree):
    da = mixed_datatree.posterior["mu"]
    default = iqr(da, round_to="none")
    wider = iqr(da, quantiles=(0.1, 0.9), round_to="none")
    narrower = iqr(da, quantiles=(0.4, 0.6), round_to="none")
    assert bool(wider >= default >= narrower)


def test_hdi_and_eti_inputs(datatree):
    assert hdi(datatree.posterior, method="multimodal")["mu"].shape == (1, 2)
    assert hdi(datatree.posterior["mu"]).shape == (2,)
    assert hdi(datatree.posterior["mu"].values).shape == (4, 2)

    assert eti(datatree.posterior, skipna=True)["mu"].shape == (2,)
    assert eti(datatree.posterior["mu"]).shape == (2,)
    assert eti(datatree.posterior["mu"].values).shape == (4, 2)

    narrow = eti(datatree, prob=0.5, var_names=["mu"])
    wide = eti(datatree, prob=0.95, var_names=["mu"])
    narrow_w = narrow["mu"].sel(ci_bound="upper") - narrow["mu"].sel(ci_bound="lower")
    wide_w = wide["mu"].sel(ci_bound="upper") - wide["mu"].sel(ci_bound="lower")
    assert bool((wide_w >= narrow_w).all())


def test_qds_inputs_and_quantiles(datatree):
    assert qds(datatree, nquantiles=25)["mu"].shape == (2, 25)
    assert qds(datatree.posterior)["mu"].shape == (2, 100)
    assert qds(datatree.posterior["mu"]).shape == (2, 100)
    res_arr = qds(datatree.posterior["mu"].values)
    assert res_arr[0].shape == (4, 100)
    assert res_arr[2].shape == (4,)


def test_summary_inputs_and_filtering(datatree, rng):
    assert summary(rng.normal(size=(4, 100, 2))).shape == (2, 9)
    assert len(summary(datatree.posterior["mu"]).index) == 1
    assert len(summary(datatree.posterior).index) == 9
    assert len(summary(datatree, var_names="mu").index) == 1
    assert len(summary(datatree, var_names=["mu", "tau"]).index) == 2
    assert "mu" in summary(datatree, var_names=["mu"], filter_vars="like").index
    assert "mu" in summary(datatree, var_names=["^mu"], filter_vars="regex").index
    assert len(summary(datatree, coords={"hierarchy": [0, 1, 2]}, var_names=["theta"]).index) == 3

    post = summary(datatree, group="posterior", kind="stats")
    prior = summary(datatree, group="prior", kind="stats")
    assert list(post["mean"]) != list(prior["mean"])


def test_summary_sd_uses_sample_std(centered_eight):
    summary_df = summary(centered_eight, var_names=["mu"], kind="stats", round_to="none")
    expected = centered_eight.posterior["mu"].std(dim=["chain", "draw"], ddof=1)
    assert_allclose(summary_df.loc["mu", "sd"], expected)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_summary_nan_and_skipna(datatree):
    dt_all_nan = deepcopy(datatree)
    dt_all_nan.posterior["theta"].loc[{"hierarchy": 0}] = np.nan
    res_all_nan = summary(dt_all_nan, skipna=True)
    assert res_all_nan.loc["theta[0]"].isnull().all()
    assert res_all_nan.drop(index="theta[0]").notnull().all().all()

    dt_part_nan = deepcopy(datatree)
    dt_part_nan.posterior["theta"].loc[{"draw": slice(10), "hierarchy": 0}] = np.nan
    res_part_nan = summary(dt_part_nan, skipna=True)
    theta_null = res_part_nan.loc["theta[0]"].isnull()
    assert ~theta_null[:4].all()
    assert theta_null[4:].all()


@pytest.mark.parametrize("ci_kind", ["hdi", "eti"])
def test_summary_ci_kind_and_prob(datatree, ci_kind):
    summary_df = summary(datatree, ci_kind=ci_kind, ci_prob=0.89, var_names=["mu"], round_to="none")
    posterior = datatree.posterior.ds[["mu"]]
    expected = getattr(posterior.azstats, ci_kind)(prob=0.89, dim=["chain", "draw"])

    assert_allclose(summary_df[f"{ci_kind}89_lb"].to_numpy(), np.asarray(expected["mu"]).ravel()[0])
    assert_allclose(summary_df[f"{ci_kind}89_ub"].to_numpy(), np.asarray(expected["mu"]).ravel()[1])

    summary_med = summary(datatree, kind="stats_median", ci_kind="hdi", var_names=["mu"])
    assert "eti89_lb" in summary_med.columns
    assert "median" in summary_med.columns


@pytest.mark.parametrize("fmt", ["wide", "long", "xarray"])
@pytest.mark.parametrize("round_to", [0, 2, "3g", None])
def test_summary_round_to(datatree, fmt, round_to):
    raw = summary(datatree, var_names=["mu"], fmt=fmt, round_to="none")
    result = summary(datatree, var_names=["mu"], fmt=fmt, round_to=round_to)
    precision = azb.rcParams["stats.round_to"] if round_to is None else round_to
    expected = np.array([round_num(v, precision) for v in _flat_values(raw)])
    assert_allclose(_flat_values(result), expected)


def test_summary_rc_context(datatree):
    with azb.rc_context({"stats.round_to": 0}):
        default = summary(datatree, var_names=["mu"], round_to="auto", fmt="xarray")
    explicit = summary(datatree, var_names=["mu"], round_to=0, fmt="xarray")
    assert_allclose(_flat_values(default), _flat_values(explicit))

    with azb.rc_context({"stats.ci_prob": 0.5, "stats.ci_kind": "hdi"}):
        res = summary(datatree, var_names=["mu"])
    assert "hdi50_lb" in res.columns


def test_summary_fmt_wide_and_long_auto_display(datatree):
    wide = summary(datatree, var_names=["mu"], round_to="auto")
    raw_wide = summary(datatree, var_names=["mu"], round_to="none")
    assert isinstance(wide, SummaryDataFrame)
    assert not isinstance(raw_wide, SummaryDataFrame)
    assert_allclose(_flat_values(wide), _flat_values(raw_wide))
    wide_disp = wide._display_df()
    assert wide_disp.loc["mu", "r_hat"] == f"{raw_wide.loc['mu', 'r_hat']:.2f}"
    assert wide_disp.loc["mu", "ess_bulk"].isdigit()
    assert f"{raw_wide.loc['mu', 'r_hat']:.2f}" in str(wide)

    col_sub = wide[["mean", "r_hat"]]._display_df()
    assert isinstance(col_sub.loc["mu", "mean"], str)

    transposed = wide.T
    assert transposed._fmt_map is not None
    assert transposed._fmt_hash is not None

    long_df = summary(datatree, fmt="long", round_to="auto")
    raw_long = summary(datatree, fmt="long", round_to="none")
    long_disp = long_df._display_df()
    expected_ess = [str(int(np.floor(v))) for v in raw_long.loc["ess_bulk", "theta"]]
    assert list(long_disp.loc["ess_bulk", "theta"]) == expected_ess
    expected_rhat = [f"{v:.2f}" for v in raw_long.loc["r_hat", "theta"]]
    assert list(long_disp.loc["r_hat", "theta"]) == expected_rhat
    assert all(isinstance(v, (int, np.integer)) for v in long_disp["hierarchy"])

    row_sub = long_df.loc[["mean", "sd"]]._display_df()
    assert all(isinstance(v, float) for v in row_sub["mu"])

    long_stats = summary(datatree, var_names=["mu"], kind="stats", fmt="long", round_to="auto")
    assert isinstance(long_stats._display_df().loc["mean", "mu"], str)

    long_diag = summary(datatree, var_names=["mu"], kind="diagnostics", fmt="long", round_to="auto")
    assert isinstance(long_diag._display_df().loc["mcse_mean", "mu"], str)


def test_summary_validations(rng):
    array = rng.normal(size=(4, 100, 2))
    with pytest.raises(ValueError, match="ci_kind must be either 'hdi' or 'eti'"):
        summary(array, ci_kind="invalid")
    with pytest.raises(ValueError, match="valid options for kind are"):
        summary(array, kind="invalid")
    with pytest.raises(ValueError, match="fmt must be one of"):
        summary(array, fmt="bogus")


def test_build_fmt_map_and_edge_cases():
    labels = ["a", "bb", "ccc", "d", "e"]
    data = {
        "mean": [111.11, 1.2345e-6, 5.4321e8, np.inf, np.nan],
        "mcse_mean": [0.0012345, 5.432e-5, 2.1234e5, np.inf, np.nan],
        "sd": [0.0012345, 5.432e-5, 2.1234e5, np.inf, np.nan],
        "r_hat": [1.009, 1.011, 0.99, np.inf, np.nan],
        "ess_bulk": [312.45, 23.32, 1011.98, np.inf, np.nan],
        "ess_tail": [9.2345, 876.321, 999.99, np.inf, np.nan],
        "count": [1, 2, 3, 4, 5],
    }
    df = pd.DataFrame(data, index=labels)
    result = _build_fmt_map(df, round_val=2)

    assert result["ess_bulk"] == {"a": "312", "bb": "23", "ccc": "1011", "d": "inf", "e": "nan"}
    assert result["r_hat"] == {"a": "1.01", "bb": "1.01", "ccc": "0.99", "d": "inf", "e": "nan"}
    assert result["mcse_mean"] == {"a": "0", "bb": "0", "ccc": "212340", "d": "inf", "e": "nan"}
    assert result["mean"] == {
        "a": "111.11",
        "bb": "0",
        "ccc": "5.43e+08",
        "d": "inf",
        "e": "nan",
    }
    assert result["sd"] == {"a": "0", "bb": "0", "ccc": "212340", "d": "inf", "e": "nan"}
    assert result["count"] == {}

    df_nan_zero = pd.DataFrame(
        {"mean": [3.14159, 2.7], "mcse_mean": [np.nan, 0.0]}, index=["a", "b"]
    )
    res_nz = _build_fmt_map(df_nan_zero, round_val=2)
    assert res_nz["mcse_mean"] == {"a": "nan", "b": "0"}
    assert res_nz["mean"] == {"a": "3.14", "b": "3"}


def test_summary_data_frame_methods():
    data = {"mean": [0.123456, -1.987654], "r_hat": [1.001234, 1.056789]}
    fmt_map = {
        "mean": {"mu_1": "0.123", "tau": "-1.988", "missing_row": "0.0"},
        "r_hat": {"mu_1": "1.00", "tau": "1.06"},
        "missing_col": {"mu_1": "x"},
    }
    sdf = SummaryDataFrame(data, index=["mu_1", "tau"], fmt_map=fmt_map)

    for text in (repr(sdf), str(sdf), sdf.to_string(), sdf._repr_html_(), sdf.to_html()):
        assert "0.123" in text
        assert "1.06" in text

    latex = sdf.to_latex()
    assert "0.123" in latex
    assert "r\\_hat" in latex
    assert "mu\\_1" in latex

    plain = SummaryDataFrame(data, index=["mu_1", "tau"])
    assert plain._fmt_map is None
    assert plain._fmt_hash is None
    assert plain.T._fmt_map is None
    for text in (
        repr(plain),
        str(plain),
        plain.to_string(),
        plain._repr_html_(),
        plain.to_html(),
        plain.to_latex(),
    ):
        assert "0.123456" in text


def test_ci_in_rope_workflows_and_validations(fake_dt, centered_eight):
    samples = np.tile(np.arange(100, dtype=float), (2, 1))
    idata = azb.from_dict({"posterior": {"a": samples}})
    assert float(ci_in_rope(idata, var_names=["a"], rope=(-1000, 1000), ci_prob=0.9)["a"]) == 100
    assert float(ci_in_rope(idata, var_names=["a"], rope=(1000, 1001), ci_prob=0.9)["a"]) == 0
    assert float(ci_in_rope(idata, var_names=["a"], rope=(-100, 49.5), ci_prob=0.9)["a"]) == 50

    res_dict = ci_in_rope(fake_dt, rope={"a": (-100, -99), "b": (-2, 2)}, ci_kind="hdi")
    assert float(res_dict["a"]) == 0
    assert float(res_dict["b"]) > 90

    rope_ds = xr.Dataset({"a": (["rope_dim"], [-1, 1]), "b": (["rope_dim"], [-2, 2])})
    res_ds = ci_in_rope(fake_dt, rope=rope_ds, var_names=["a", "b"], group="prior")
    assert set(res_ds.data_vars) == {"a", "b"}

    schools = ["Choate", "Deerfield"]
    full = ci_in_rope(centered_eight, var_names=["theta"], rope=(-0.5, 0.5))
    subset = ci_in_rope(
        centered_eight, var_names=["theta"], rope=(-0.5, 0.5), coords={"school": schools}
    )
    assert_allclose(subset["theta"].values, full["theta"].sel(school=schools).values)

    with pytest.raises(ValueError, match="ci_kind must be either"):
        ci_in_rope(fake_dt, var_names=["a"], rope=(-1, 1), ci_kind="bogus")
    with pytest.raises(ValueError, match="rope` must be a subset of the variables"):
        ci_in_rope(fake_dt, rope={"invalid_var": (-1, 1)})
    with pytest.raises(ValueError, match="rope` must be a dict of tuples of length 2"):
        ci_in_rope(fake_dt, rope={"a": (-1,)})
    with pytest.raises(ValueError, match="rope_dim is not a dimension"):
        ci_in_rope(fake_dt, rope=xr.Dataset({"a": (["wrong_dim"], [-1, 1])}), rope_dim="rope_dim")
    with pytest.raises(ValueError, match="Length of rope_dim dim must be 2"):
        ci_in_rope(fake_dt, rope=xr.Dataset({"a": (["rope_dim"], [-1, 0, 1])}))
    with pytest.raises(ValueError, match="tuple of length 2"):
        ci_in_rope(fake_dt, var_names=["a"], rope=(1, 2, 3))
    with pytest.raises(ValueError, match="failed to convert"):
        ci_in_rope(fake_dt, var_names=["a"], rope=[1, [2, 3]])


def test_summary_data_frame_display_after_changes():
    data = {"mean": [0.123456, -1.987654], "r_hat": [1.001234, 1.056789]}
    fmt_map = {
        "mean": {"mu_1": "0.123", "tau": "-1.988"},
        "r_hat": {"mu_1": "1.00", "tau": "1.06"},
    }
    sdf = SummaryDataFrame(data, index=["mu_1", "tau"], fmt_map=fmt_map)

    # adding columns keeps the formatting of the existing ones
    added = sdf.assign(new=[0.111111, 0.222222])
    sdf_inplace = sdf.copy()
    sdf_inplace["new"] = [0.111111, 0.222222]
    for frame in (added, sdf_inplace):
        out = frame.to_string()
        assert "0.123" in out
        assert "0.123456" not in out
        assert "1.06" in out
        assert "0.111111" in out
        assert "1.06" in frame.T.to_string()
        assert "0.111111" in frame.T.to_string()

    # row filtering and transposition keep the formatting
    assert "-1.988" in sdf.loc[["tau"]].to_string()
    assert "-1.987654" not in sdf.loc[["tau"]].to_string()
    assert "-1.988" in sdf.T.to_string()
    assert "-1.988" in sdf.T.T.to_string()

    # changing values reverts to pandas' default display
    scaled = sdf * 100
    edited = sdf.copy()
    edited.loc["tau", "mean"] = 75.0
    edited_t = sdf.T
    edited_t.loc["mean", "tau"] = 75.0
    for frame in (scaled, edited, edited.T, edited_t, edited_t.T):
        plain = pd.DataFrame(frame)
        assert repr(frame) == repr(plain)
        assert frame.to_string() == plain.to_string()
        assert frame._repr_html_() == plain._repr_html_()
        assert frame.to_html() == plain.to_html()
        assert frame.to_latex() == plain.to_latex()
    assert "12.3456" in scaled.to_string()
    assert "75.0" in edited.to_string()

    # new rows also revert to pandas' default display
    extended = sdf.copy()
    extended.loc["sigma"] = [0.5, 1.0]
    assert extended.to_string() == pd.DataFrame(extended).to_string()


def test_summary_display_after_scaling():
    posterior = xr.Dataset(
        {"rate": (("chain", "draw"), [[0.125, 0.375, 0.625, 0.875], [0.25, 0.5, 0.5, 0.75]])}
    )
    table = summary(posterior, kind="stats")[["mean"]]
    assert "50.0" in (table * 100).to_string()


def test_summary_data_frame_display_after_column_selection():
    data = {"mean": [0.123456, -1.987654], "r_hat": [1.001234, 1.056789]}
    fmt_map = {
        "mean": {"mu_1": "0.123", "tau": "-1.988"},
        "r_hat": {"mu_1": "1.00", "tau": "1.06"},
    }
    sdf = SummaryDataFrame(data, index=["mu_1", "tau"], fmt_map=fmt_map)

    # dropping formatted columns keeps the formatting of the remaining ones
    selected = sdf[["r_hat"]]
    out = selected.to_string()
    assert "1.06" in out
    assert "1.056789" not in out
    assert "mean" not in out

    # changing the values of a remaining column still reverts to pandas' default display
    edited = selected.copy()
    edited.loc["tau", "r_hat"] = 1.5
    assert edited.to_string() == pd.DataFrame(edited).to_string()
    assert "1.5" in edited.to_string()


def test_summary_data_frame_display_unhashable_values():
    data = {"mean": [0.123456, -1.987654], "r_hat": [1.001234, 1.056789]}
    fmt_map = {
        "mean": {"mu_1": "0.123", "tau": "-1.988"},
        "r_hat": {"mu_1": "1.00", "tau": "1.06"},
    }
    sdf = SummaryDataFrame(data, index=["mu_1", "tau"], fmt_map=fmt_map)
    sdf["mean"] = pd.Series([np.array([1, 2]), np.array([3, 4])], index=sdf.index, dtype=object)

    # values that can't be hashed fall back to pandas' default display
    plain = pd.DataFrame(sdf)
    assert repr(sdf) == repr(plain)
    assert sdf.to_string() == plain.to_string()
    assert sdf.to_html() == plain.to_html()
    assert "1.056789" in sdf.to_string()
