"""Test moment matching for PSIS-LOO-CV."""

# pylint: disable=redefined-outer-name
import pytest

from ..helpers import importorskip

importorskip("arviz_base")
js = importorskip("json")
np = importorskip("numpy")
xr = importorskip("xarray")
ft = importorskip("functools")
os = importorskip("os")
pt = importorskip("pathlib")
atexit = importorskip("atexit")
lzma = importorskip("lzma")
shutil = importorskip("shutil")
tempfile = importorskip("tempfile")

from arviz_stats.loo import loo, loo_moment_match
from arviz_stats.loo.loo_moment_match import _split_moment_match


@ft.lru_cache(maxsize=1)
def _get_roaches_data_path():
    data = pt.Path(__file__).resolve().parent / "roaches.nc.xz"
    if not data.exists():
        raise FileNotFoundError(f"missing roaches data at {data}")

    with tempfile.NamedTemporaryFile(
        prefix="arviz_stats_roaches_", suffix=".nc", delete=False
    ) as handle:
        with lzma.open(data, "rb") as src:
            shutil.copyfileobj(src, handle)
        path = pt.Path(handle.name)

    atexit.register(lambda target=path: target.unlink(missing_ok=True))
    return path


ROACHES_DATA_PATH = _get_roaches_data_path()


def _safe_exp(da):
    with np.errstate(over="ignore"):
        data = np.exp(da.data)
    return xr.DataArray(data, dims=da.dims, coords=da.coords)


def _as_beta(upars, beta_param_names, coef_names):
    return upars.sel(uparam=beta_param_names).rename(uparam="coef").assign_coords(coef=coef_names)


def log_prob_upars(
    upars,
    design_matrix,
    y_da,
    offset_da,
    factorial_term,
    beta_param_names,
    coef_names,
    beta_scale,
    alpha_scale,
    const_log_prior_beta,
    const_log_prior_intercept,
    sample_dims,
):
    upars = upars.transpose(*sample_dims, ...)
    beta = _as_beta(upars, beta_param_names, coef_names)
    intercept = upars.sel(uparam="intercept")
    lin = xr.dot(beta, design_matrix, dim="coef") + intercept + offset_da
    exp_term = _safe_exp(lin)

    log_lik = y_da * lin - exp_term - factorial_term
    log_prior_beta = const_log_prior_beta - 0.5 * (beta / beta_scale) ** 2
    log_prior_beta = log_prior_beta.sum("coef")
    log_prior_intercept = const_log_prior_intercept - 0.5 * (intercept / alpha_scale) ** 2
    return log_lik.sum("obs") + log_prior_beta + log_prior_intercept


def log_lik_i_upars(
    upars,
    index,
    design_matrix,
    y_da,
    offset_da,
    factorial_term,
    beta_param_names,
    coef_names,
    sample_dims,
):
    upars = upars.transpose(*sample_dims, ...)
    beta = _as_beta(upars, beta_param_names, coef_names)
    intercept = upars.sel(uparam="intercept")
    features_i = design_matrix.isel(obs=index)
    lin = (beta * features_i).sum("coef") + intercept + offset_da.isel(obs=index)
    return y_da.isel(obs=index) * lin - _safe_exp(lin) - factorial_term.isel(obs=index)


def load_roaches_r_example():
    root_ds = xr.load_dataset(ROACHES_DATA_PATH)
    posterior_ds = xr.load_dataset(ROACHES_DATA_PATH, group="posterior")
    log_likelihood_ds = xr.load_dataset(ROACHES_DATA_PATH, group="log_likelihood")
    observed_ds = xr.load_dataset(ROACHES_DATA_PATH, group="observed_data")
    upars_store = xr.load_dataset(ROACHES_DATA_PATH, group="upars")

    coef_names = posterior_ds["beta"].coords["coef"].values.tolist()
    beta_param_names = [f"beta_{name}" for name in coef_names]

    data_tree = xr.DataTree()
    data_tree["posterior"] = xr.DataTree(dataset=posterior_ds)
    data_tree["log_likelihood"] = xr.DataTree(dataset=log_likelihood_ds)

    design_matrix = observed_ds["design_matrix"].rename("design_matrix")
    y_da = observed_ds["y"].rename("y")
    offset_da = observed_ds["offset"].rename("offset")
    factorial_term = observed_ds["factorial_term"].rename("factorial_term")
    upars_da = upars_store["upars"].rename("upars")

    beta_scale = root_ds.attrs["beta_prior_scale"]
    alpha_scale = root_ds.attrs["alpha_prior_scale"]
    const_log_prior_beta = -0.5 * np.log(2 * np.pi * beta_scale**2)
    const_log_prior_intercept = -0.5 * np.log(2 * np.pi * alpha_scale**2)
    sample_dims = ["chain", "draw"]

    log_prob_fn = ft.partial(
        log_prob_upars,
        design_matrix=design_matrix,
        y_da=y_da,
        offset_da=offset_da,
        factorial_term=factorial_term,
        beta_param_names=beta_param_names,
        coef_names=coef_names,
        beta_scale=beta_scale,
        alpha_scale=alpha_scale,
        const_log_prior_beta=const_log_prior_beta,
        const_log_prior_intercept=const_log_prior_intercept,
        sample_dims=sample_dims,
    )

    log_lik_i_fn = ft.partial(
        log_lik_i_upars,
        design_matrix=design_matrix,
        y_da=y_da,
        offset_da=offset_da,
        factorial_term=factorial_term,
        beta_param_names=beta_param_names,
        coef_names=coef_names,
        sample_dims=sample_dims,
    )

    r_reference = js.loads(root_ds.attrs["r_reference_json"])
    root_ds.close()

    return {
        "data_tree": data_tree,
        "upars": upars_da,
        "log_prob_fn": log_prob_fn,
        "log_lik_i_fn": log_lik_i_fn,
        "r_reference": r_reference,
    }


@pytest.fixture(scope="module")
def roaches_r_example():
    return load_roaches_r_example()


def moment_match_debug_payload(parity, log_fn=None):
    payload = {
        "log_lik_ref": parity.get("log_lik"),
        "log_weights_ref": parity.get("log_weights_mm"),
        "mm_debug": parity.get("mm_debug") or {},
        "metadata": parity.get("metadata", {}),
        "targets": parity.get("elpd", {}).get("moment_match", {}) if parity.get("elpd") else {},
    }
    if log_fn is not None:
        payload["log_fn"] = log_fn
    return payload


def reshape_draw_major(matrix, n_chains, n_draws, n_params):
    return matrix.reshape(n_chains, n_draws, n_params).transpose(1, 0, 2).reshape(-1, n_params)


def transform_forward_upars(upars_matrix, total_shift, total_scaling, total_mapping):
    mean_original = upars_matrix.mean(axis=0)
    centered = upars_matrix - mean_original
    scaled = centered * total_scaling
    if total_mapping.size:
        scaled = scaled @ total_mapping.T
    return scaled + (total_shift + mean_original)


def transform_inverse_upars(upars_matrix, total_shift, total_scaling, total_mapping):
    mean_original = upars_matrix.mean(axis=0)
    centered = upars_matrix - mean_original
    mapped = centered
    if total_mapping.size:
        mapped = mapped @ np.linalg.inv(total_mapping.T)
    scaled = mapped / total_scaling
    return scaled + (mean_original - total_shift)


def load_r_parity():
    parity_ds = xr.load_dataset(ROACHES_DATA_PATH, group="parity")
    try:
        log_lik = parity_ds["log_lik"].load().rename("log_lik")
        log_weights = parity_ds["log_weights"].load().rename("log_weights")

        log_weights_mm = parity_ds.get("log_weights_mm")
        if log_weights_mm is not None:
            log_weights_mm = log_weights_mm.load().rename("log_weights_mm")

        pareto_k = parity_ds["pareto_k"].load().rename("pareto_k")
        pareto_k_mm = parity_ds.get("pareto_k_mm")
        if pareto_k_mm is not None:
            pareto_k_mm = pareto_k_mm.load().rename("pareto_k_mm")

        r_eff = parity_ds["r_eff"].load().rename("r_eff")
        r_eff_mm = parity_ds.get("r_eff_mm")
        if r_eff_mm is not None:
            r_eff_mm = r_eff_mm.load().rename("r_eff_mm")

        elpd = js.loads(parity_ds.attrs.get("elpd_json", "{}"))
        metadata = js.loads(parity_ds.attrs.get("metadata_json", "{}"))
        mm_debug_raw = js.loads(parity_ds.attrs.get("mm_debug_json", "{}"))
        mm_debug = (
            {int(key): value for key, value in mm_debug_raw.items()} if mm_debug_raw else None
        )

        return {
            "log_lik": log_lik,
            "log_weights": log_weights,
            "pareto_k": pareto_k,
            "r_eff": r_eff,
            "elpd": elpd,
            "metadata": metadata,
            "log_weights_mm": log_weights_mm,
            "pareto_k_mm": pareto_k_mm,
            "r_eff_mm": r_eff_mm,
            "mm_debug": mm_debug,
        }
    finally:
        parity_ds.close()


def capture_log_prob_call(container, upars_da):
    container.append(upars_da.copy(deep=True))
    return xr.zeros_like(upars_da.isel(uparam=0, drop=True))


def capture_log_lik_call(container, upars_da, _):
    container.append(upars_da.copy(deep=True))
    return xr.zeros_like(upars_da.isel(uparam=0, drop=True))


def test_psis_matches_r_reference(roaches_r_example):
    example = roaches_r_example
    loo_result = loo(example["data_tree"], pointwise=True, var_name="log_lik")
    ref = example["r_reference"]["loo1"]

    assert loo_result.elpd == pytest.approx(ref["elpd"][0], abs=1.0)
    assert loo_result.p == pytest.approx(ref["p"][0], abs=1.0)

    np.testing.assert_allclose(
        loo_result.pareto_k.values,
        np.asarray(ref["pareto_k"]),
        atol=0.2,
        rtol=0,
    )


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_moment_match_matches_r_reference(roaches_r_example):
    example = roaches_r_example
    parity = load_r_parity()
    loo_orig = loo(
        example["data_tree"],
        pointwise=True,
        var_name="log_lik",
        reff=parity["r_eff"],
        log_weights=parity["log_weights"],
        pareto_k=parity["pareto_k"],
    )

    if os.environ.get("ARVIZ_STATS_MM_DEBUG"):
        setattr(loo_orig, "_moment_match_debug", moment_match_debug_payload(parity))

    loo_mm = loo_moment_match(
        example["data_tree"],
        loo_orig,
        log_prob_upars_fn=example["log_prob_fn"],
        log_lik_i_upars_fn=example["log_lik_i_fn"],
        upars=example["upars"],
        var_name="log_lik",
        cov=True,
    )

    mm_ref = parity["elpd"].get("moment_match", {})
    if mm_ref:
        assert loo_mm.elpd == pytest.approx(mm_ref.get("elpd"), abs=1.0)
        assert loo_mm.p == pytest.approx(mm_ref.get("p"), abs=1.0)

        if "pareto_k" in mm_ref:
            np.testing.assert_allclose(
                loo_mm.pareto_k.values,
                np.asarray(mm_ref["pareto_k"]),
                atol=1.0,
                rtol=0,
            )

    xr.testing.assert_allclose(
        loo_mm.pareto_k,
        parity["pareto_k_mm"],
        atol=1.1,
        rtol=0,
    )

    if loo_mm.log_weights is not None and parity["log_weights_mm"] is not None:
        xr.testing.assert_allclose(
            loo_mm.log_weights.transpose(*parity["log_weights_mm"].dims),
            parity["log_weights_mm"],
            atol=1e-3,
            rtol=0,
        )


def test_split_moment_match_matches_r_snapshot(roaches_r_example):
    split_case_ds = xr.load_dataset(ROACHES_DATA_PATH, group="split_case")
    split_snapshot_ds = xr.load_dataset(ROACHES_DATA_PATH, group="split_snapshot")

    upars_matrix = split_case_ds["upars"].values.astype(np.float64, copy=False)
    total_shift = split_case_ds["total_shift"].values.astype(np.float64, copy=False)
    total_scaling = split_case_ds["total_scaling"].values.astype(np.float64, copy=False)
    total_mapping = split_case_ds["total_mapping"].values.astype(np.float64, copy=False)

    n_samples, n_params = upars_matrix.shape
    n_chains = roaches_r_example["upars"].sizes["chain"]
    n_draws = n_samples // n_chains

    upars = xr.DataArray(
        upars_matrix.reshape(n_chains, n_draws, n_params),
        dims=("chain", "draw", "uparam"),
        coords={
            "chain": roaches_r_example["upars"].coords["chain"],
            "draw": roaches_r_example["upars"].coords["draw"],
            "uparam": roaches_r_example["upars"].coords["uparam"],
        },
    )

    log_prob_calls = []
    log_lik_calls = []

    log_prob_fn = ft.partial(capture_log_prob_call, log_prob_calls)
    log_lik_i_fn = ft.partial(capture_log_lik_call, log_lik_calls)

    _split_moment_match(
        upars=upars,
        cov=True,
        total_shift=total_shift,
        total_scaling=total_scaling,
        total_mapping=total_mapping,
        i=int(split_case_ds.attrs["observation_index"]),
        reff=1.0,
        log_prob_upars_fn=log_prob_fn,
        log_lik_i_upars_fn=log_lik_i_fn,
    )

    assert len(log_prob_calls) >= 2
    upars_trans_half = log_prob_calls[0]
    upars_trans_half_inv = log_prob_calls[1]
    param_dim = next(dim for dim in upars_trans_half.dims if dim not in {"chain", "draw"})

    stacked_trans_half = (
        upars_trans_half.stack(__sample__=("draw", "chain"))
        .transpose("__sample__", param_dim)
        .values
    )

    upars_matrix_draw_major = reshape_draw_major(upars_matrix, n_chains, n_draws, n_params)

    upars_trans = transform_forward_upars(
        upars_matrix,
        total_shift,
        total_scaling,
        total_mapping,
    )
    upars_trans_draw_major = reshape_draw_major(upars_trans, n_chains, n_draws, n_params)

    expected_half = upars_matrix_draw_major.copy()
    expected_half[: n_samples // 2, :] = upars_trans_draw_major[: n_samples // 2, :]

    r_snapshot = split_snapshot_ds["upars_snapshot"].values
    r_snapshot_draw_major = reshape_draw_major(r_snapshot, n_chains, n_draws, n_params)

    np.testing.assert_allclose(stacked_trans_half, r_snapshot_draw_major, rtol=0, atol=1e-12)
    np.testing.assert_allclose(stacked_trans_half, expected_half, rtol=0, atol=1e-10)

    stacked_trans_half_inv = (
        upars_trans_half_inv.stack(__sample__=("draw", "chain"))
        .transpose("__sample__", param_dim)
        .values
    )

    upars_trans_inv = transform_inverse_upars(
        upars_matrix,
        total_shift,
        total_scaling,
        total_mapping,
    )
    upars_trans_inv_draw_major = reshape_draw_major(upars_trans_inv, n_chains, n_draws, n_params)

    expected_half_inv = upars_matrix_draw_major.copy()
    expected_half_inv[n_samples // 2 :, :] = upars_trans_inv_draw_major[n_samples // 2 :, :]

    np.testing.assert_allclose(stacked_trans_half_inv, expected_half_inv, rtol=0, atol=1e-10)

    split_case_ds.close()
    split_snapshot_ds.close()
