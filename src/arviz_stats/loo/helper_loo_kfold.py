"""Helper functions for K-fold cross-validation."""

from collections import namedtuple

import numpy as np
import xarray as xr
from arviz_base import convert_to_datatree
from xarray_einstats.stats import logsumexp

from arviz_stats.loo.wrapper import SamplingWrapper
from arviz_stats.utils import get_log_likelihood

__all__ = [
    "_prepare_kfold_inputs",
    "_compute_kfold_results",
    "_kfold_split_random",
    "_kfold_split_stratified",
    "_kfold_split_grouped",
    "_extract_fold_data",
    "_get_fold_indices",
    "_combine_fold_elpds",
    "_validate_k_value",
    "_validate_array_length",
    "_validate_fold_parameters",
]

FoldData = namedtuple("FoldData", ["train_indices", "test_indices", "fold_id"])

KfoldResults = namedtuple("KfoldResults", ["elpds", "ps", "fold_fits"])

KfoldInputs = namedtuple(
    "KfoldInputs",
    [
        "log_likelihood",
        "var_name",
        "sample_dims",
        "obs_dims",
        "n_data_points",
        "n_samples",
        "folds",
        "k",
    ],
)


def _prepare_kfold_inputs(data, var_name, wrapper, k, folds, stratify_by, group_by):
    """Prepare inputs for k-fold cross-validation."""
    data = convert_to_datatree(data)

    if not isinstance(wrapper, SamplingWrapper):
        raise TypeError("wrapper must be an instance of SamplingWrapper")

    required_methods = ["sel_observations", "sample", "get_inference_data", "log_likelihood__i"]
    not_implemented = wrapper.check_implemented_methods(required_methods)

    if not_implemented:
        raise ValueError(
            f"The following methods must be implemented in the SamplingWrapper: {not_implemented}"
        )

    log_likelihood = get_log_likelihood(data, var_name)
    var_name = log_likelihood.name if var_name is None else var_name
    sample_dims = ["chain", "draw"]

    obs_dims = [dim for dim in log_likelihood.dims if dim not in sample_dims]
    n_data_points = int(np.prod([log_likelihood.sizes[dim] for dim in obs_dims]))
    n_samples = int(np.prod([log_likelihood.sizes[dim] for dim in sample_dims]))

    _validate_fold_parameters(folds, stratify_by, group_by)

    if folds is not None:
        folds = _validate_array_length(folds, n_data_points, "folds")
        k = len(np.unique(folds))
    elif stratify_by is not None:
        stratify_by = _validate_array_length(stratify_by, n_data_points, "stratify_by")
        folds = _kfold_split_stratified(k=k, x=stratify_by)
    elif group_by is not None:
        group_by = _validate_array_length(group_by, n_data_points, "group_by")
        folds = _kfold_split_grouped(k=k, x=group_by)
    else:
        folds = _kfold_split_random(k=k, n=n_data_points)

    return KfoldInputs(
        log_likelihood=log_likelihood,
        var_name=var_name,
        sample_dims=sample_dims,
        obs_dims=obs_dims,
        n_data_points=n_data_points,
        n_samples=n_samples,
        folds=folds,
        k=k,
    )


def _compute_kfold_results(kfold_inputs, wrapper, save_fits):
    """Compute k-fold cross-validation results."""
    ll_full = kfold_inputs.log_likelihood
    lpds_full = logsumexp(ll_full, dims=kfold_inputs.sample_dims, b=1 / kfold_inputs.n_samples)

    fold_indices = _get_fold_indices(kfold_inputs.folds, kfold_inputs.k)

    elpd_unordered = []
    fold_results = {} if save_fits else None

    for fold_num in range(1, kfold_inputs.k + 1):
        test_idx = fold_indices[fold_num]["test_indices"]
        train_data, test_data = wrapper.sel_observations(test_idx)
        fitted_model = wrapper.sample(train_data)
        idata_k = wrapper.get_inference_data(fitted_model)
        log_lik_k = wrapper.log_likelihood__i(test_data, idata_k)

        if "chain" in log_lik_k.dims and "draw" in log_lik_k.dims:
            sample_dims_k = ["chain", "draw"]
        else:
            sample_dims_k = [dim for dim in log_lik_k.dims if dim not in kfold_inputs.obs_dims]

        n_samples_k = int(np.prod([log_lik_k.sizes[dim] for dim in sample_dims_k]))
        elpd_k = logsumexp(log_lik_k, dims=sample_dims_k, b=1 / n_samples_k)
        elpd_values = elpd_k.values if hasattr(elpd_k, "values") else elpd_k

        if np.isscalar(elpd_values):
            elpd_unordered.append((test_idx[0], float(elpd_values)))
        else:
            for i, idx in enumerate(test_idx):
                elpd_unordered.append((idx, float(elpd_values[i])))

        if save_fits:
            fold_results[fold_num] = {"fit": idata_k, "test_indices": test_idx}

    elpd_unordered.sort(key=lambda x: x[0])
    elpds = np.array([x[1] for x in elpd_unordered])

    lpds_ordered = lpds_full.values.flatten()
    ps = lpds_ordered - elpds

    return KfoldResults(elpds=elpds, ps=ps, fold_fits=fold_results)


def _kfold_split_random(k=10, n=None):
    """Split the data into K groups of equal size (or roughly equal size)."""
    if n is None:
        raise ValueError("n must be provided")
    if not isinstance(n, int | np.integer) or np.isnan(n):
        raise ValueError("n must be an integer")
    n = int(n)
    k = _validate_k_value(k, n)

    fold_size = n // k
    remainder = n % k

    folds = np.zeros(n, dtype=int)
    start = 0
    for i in range(k):
        end = start + fold_size + (1 if i < remainder else 0)
        folds[start:end] = i + 1
        start = end

    rng = np.random.default_rng()
    perm = rng.permutation(n)
    return folds[perm]


def _kfold_split_stratified(k=10, x=None):
    """Split observations into k groups ensuring relative category frequencies are preserved."""
    if x is None:
        raise ValueError("x must be provided")
    x = np.asarray(x)
    k = _validate_k_value(k, len(x))

    unique_vals = np.unique(x)
    x_int = np.zeros(len(x), dtype=int)
    for i, val in enumerate(unique_vals):
        x_int[x == val] = i

    n_lev = len(unique_vals)
    n = len(x)
    xids = []

    rng = np.random.default_rng()
    for level in range(n_lev):
        idx = np.where(x_int == level)[0]
        if len(idx) > 1:
            xids.extend(rng.permutation(idx))
        else:
            xids.extend(idx)

    xids = np.array(xids)
    bins = np.zeros(n, dtype=int)
    bins[xids] = np.tile(np.arange(1, k + 1), int(np.ceil(n / k)))[:n]

    return bins


def _kfold_split_grouped(k=10, x=None):
    """Split observations ensuring all observations from the same group stay together."""
    if x is None:
        raise ValueError("x must be provided")
    x = np.asarray(x)
    k = _validate_k_value(k, len(x))

    unique_vals = np.unique(x)
    x_int = np.zeros(len(x), dtype=int)
    for i, val in enumerate(unique_vals):
        x_int[x == val] = i + 1

    n_levels = len(unique_vals)

    if n_levels < k:
        raise ValueError("k must not be bigger than the number of levels/groups in x")

    if n_levels == k:
        return x_int

    s1 = int(np.ceil(n_levels / k))
    n_s2 = s1 * k - n_levels
    n_s1 = k - n_s2

    rng = np.random.default_rng()
    perm = rng.permutation(n_levels) + 1

    breaks = []
    if n_s1 > 0:
        breaks.extend(np.arange(s1 + 0.5, s1 * n_s1 + 0.5, s1))
    if n_s2 > 0:
        start = breaks[-1] if breaks else 0.5
        breaks.extend(np.arange(start + s1 - 1, start + (s1 - 1) * n_s2, s1 - 1))

    breaks = np.array(breaks)
    groups = np.searchsorted(breaks, perm, side="right") + 1

    bins = np.zeros(len(x), dtype=int)
    for j in range(1, n_levels + 1):
        bins[x_int == j] = groups[j - 1]

    return bins


def _extract_fold_data(data, fold_indices, train=True):
    """Extract data for a specific fold."""
    if train:
        mask = np.ones(data.shape[-1], dtype=bool)
        mask[fold_indices] = False
        return data.isel({data.dims[-1]: mask})
    return data.isel({data.dims[-1]: fold_indices})


def _get_fold_indices(fold_assignments, k):
    """Get test indices for each fold."""
    results = {}

    for i in range(1, k + 1):
        test_indices = np.where(fold_assignments == i)[0]
        results[i] = {"test_indices": test_indices, "n_test": len(test_indices)}

    return results


def _combine_fold_elpds(fold_elpds, n_data_points):
    """Combine ELPD values from all folds into final estimates."""
    elpds = np.concatenate(fold_elpds)
    elpd_kfold = np.sum(elpds)
    se_elpd_kfold = np.sqrt(n_data_points * np.var(elpds))

    return {"elpd_kfold": elpd_kfold, "se_elpd_kfold": se_elpd_kfold, "pointwise": elpds}


def _validate_k_value(k, n, param_name="k"):
    """Validate k parameter for k-fold splitting."""
    if not isinstance(k, int | np.integer) or np.isnan(k):
        raise ValueError(f"{param_name} must be an integer")
    k = int(k)
    if k <= 1:
        raise ValueError(f"{param_name} must be greater than 1")
    if k > n:
        raise ValueError(f"{param_name} must not be greater than n ({n})")
    return k


def _validate_array_length(array, expected_length, param_name):
    """Validate array length matches expected number of observations."""
    if array is None:
        raise ValueError(f"{param_name} must be provided")

    if isinstance(array, xr.DataArray):
        array_length = array.size
        array_values = array.values.flatten()
    else:
        array = np.asarray(array)
        array_length = len(array)
        array_values = array

    if array_length != expected_length:
        raise ValueError(
            f"Length of {param_name} ({array_length}) must match number of "
            f"observations ({expected_length})"
        )
    return array_values


def _validate_fold_parameters(folds, stratify_by, group_by):
    """Validate fold parameter combinations."""
    if folds is not None:
        if stratify_by is not None or group_by is not None:
            raise ValueError(
                "Cannot use stratify_by or group_by when folds are explicitly provided"
            )
    elif stratify_by is not None and group_by is not None:
        raise ValueError("Cannot use both stratify_by and group_by")
