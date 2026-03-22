"""Compare PSIS-LOO-CV results."""

import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
from arviz_base import rcParams
from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.stats import dirichlet, norm

from arviz_stats.base.stats_utils import get_decimal_places_from_se, round_num
from arviz_stats.loo import loo
from arviz_stats.loo.helper_loo import _diff_srs_estimator
from arviz_stats.utils import ELPDData


def compare(
    compare_dict,
    method="stacking",
    var_name=None,
    reference=None,
    round_to="auto",
):
    r"""Compare models based on their expected log pointwise predictive density (ELPD).

    The ELPD is estimated by Pareto smoothed importance sampling leave-one-out
    cross-validation, the same method used by :func:`arviz_stats.loo`.
    The method is described in [2]_ and [3]_.
    By default, the weights are estimated using ``"stacking"`` as described in [4]_.

    If more than 11 models are compared, a diagnostic check for selection bias
    is performed. If detected, avoid LOO-based selection and use model averaging
    or `projection predictive inference <https://kulprit.readthedocs.io/en/latest/index.html>`_.

    See the EABM chapters on `Model Comparison <https://arviz-devs.github.io/EABM/Chapters/Model_comparison.html>`_,
    `Model Comparison (Case Study) <https://arviz-devs.github.io/EABM/Chapters/Case_study_model_comparison.html>`_,
    and `Model Comparison for Large Data <https://arviz-devs.github.io/EABM/Chapters/Model_comparison_large_data.html>`_
    for more details.

    Parameters
    ----------
    compare_dict: dict of {str: DataTree or ELPDData}
        A dictionary of model names and :class:`xr.DataTree` or ``ELPDData``.
    method: str, optional
        Method used to estimate the weights for each model. Available options are:

        - 'stacking' : stacking of predictive distributions.
        - 'BB-pseudo-BMA' : pseudo-Bayesian Model averaging using Akaike-type
          weighting. The weights are stabilized using the Bayesian bootstrap.
        - 'pseudo-BMA': pseudo-Bayesian Model averaging using Akaike-type
          weighting, without Bootstrap stabilization (not recommended).

        For more information read https://arxiv.org/abs/1704.02030
    var_name: str, optional
        If there is more than a single observed variable in the ``InferenceData``, which
        should be used as the basis for comparison.
    reference: str, optional
        Name of the reference model used for computing ``elpd_diff``. If ``None`` (default),
        the best-performing model (highest ELPD) is used as the reference. When specified,
        all ``elpd_diff`` values are computed relative to this model, which will have
        ``elpd_diff = 0``. This is useful for comparing against a baseline model, null model,
        or a specific model of interest rather than the top-ranked model.
    round_to : int or {"auto", "none"}, optional
        Rounding specification. Defaults to "auto". If integer, number of decimal places to
        round to. Use the string "None" or "none" to return raw numbers. If None use
        ``rcParams["stats.round_to"]``.
        If ``"auto"``, applies custom rounding rules to columns in the returned DataFrame:

        * ``elpd`` and ``elpd_diff`` are rounded based on ``se`` and ``dse`` respectively,
            using the same rule as ``summary`` stat/se pairs.
        * ``se`` and ``dse`` are rounded based on ``rcParams["stats.round_to"]``.
        * ``p`` is rounded to 1 decimal place.
        * ``weight`` uses precision based on the largest weight, showing approximately 2
          significant digits for that maximum value.

    Returns
    -------
    DataFrame
        A DataFrame, ordered from best to worst model (measured by the ELPD).
        The index reflects the key with which the models are passed to this function.
        The columns are:

        - **rank**: The rank-order of the models. 0 is the best.
        - **elpd**: ELPD estimated using PSIS-LOO-CV (`elpd_loo`).
          Higher ELPD indicates higher out-of-sample predictive fit ("better" model).
        - **pIC**: Estimated effective number of parameters.
        - **elpd_diff**: The difference in ELPD between two models.
          If more than two models are compared, the difference is computed relative to the
          top-ranked model, that always has an `elpd_diff` of 0.
        - **weight**: Relative weight for each model.
          This can be loosely interpreted as the probability of each model
          (among the compared models)
          given the data. By default the uncertainty in the weights estimation is considered using
          Bayesian bootstrap.
        - **SE**: Standard error of the ELPD estimate.
          If method = BB-pseudo-BMA these values are estimated using Bayesian bootstrap.
        - **dSE**: Standard error of the difference in ELPD between each model
          and the top-ranked model. It's always 0 for the top-ranked model.
        - **subsampling_dSE**: (Only when subsampling is used) The subsampling component
          of the standard error of the ELPD difference. This quantifies the uncertainty due to
          using a subsample rather than all observations.
        - **warning**: A value of 1 indicates that the computation of the ELPD may not be reliable.
          This could be indication of LOO starting to fail see
          http://arxiv.org/abs/1507.04544 for details.

    Examples
    --------
    Compare the centered and non centered models of the eight school problem:

    .. ipython::  python
        :okwarning:

        In [1]: from arviz_stats import compare
           ...: from arviz_base import load_arviz_data
           ...: data1 = load_arviz_data("non_centered_eight")
           ...: data2 = load_arviz_data("centered_eight")
           ...: compare_dict = {"non centered": data1, "centered": data2}
           ...: compare(compare_dict)

    Compare models using subsampled LOO:

    .. ipython::  python
        :okwarning:

        In [1]: from arviz_stats import loo_subsample
           ...: from arviz_base import load_arviz_data
           ...: data1 = load_arviz_data("non_centered_eight")
           ...: data2 = load_arviz_data("centered_eight")
           ...: loo_sub1 = loo_subsample(data1, observations=6, pointwise=True, seed=42)
           ...: loo_sub2 = loo_subsample(data2, observations=6, pointwise=True, seed=42)
           ...: compare({"non_centered": loo_sub1, "centered": loo_sub2})

    When using subsampled LOO, the ``subsampling_dse`` column quantifies the additional
    uncertainty from using subsamples instead of all observations. The ``elpd_diff`` values
    are computed using a difference-of-estimators approach on overlapping observations, which
    can differ from simple subtraction of ELPD values. Using the same seed across models
    ensures overlapping observations for more accurate paired comparisons with smaller
    standard errors.

    See Also
    --------
    :func:`loo` : Compute the ELPD using the Pareto smoothed importance sampling Leave-one-out
        cross-validation method.
    :func:`arviz_plots.plot_compare`: Summary plot for model comparison.

    References
    ----------

    .. [1] McLatchie, Y., Vehtari, A. *Efficient estimation and correction of selection-induced
        bias with order statistics*. Statistics and Computing, 34, 132 (2024).
        https://doi.org/10.1007/s11222-024-10442-4
        arXiv preprint https://arxiv.org/abs/2309.03742

    .. [2] Vehtari et al. *Practical Bayesian model evaluation using leave-one-out cross-validation
        and WAIC*. Statistics and Computing. 27(5) (2017) https://doi.org/10.1007/s11222-016-9696-4
        arXiv preprint https://arxiv.org/abs/1507.04544.

    .. [3] Vehtari et al. *Pareto Smoothed Importance Sampling*.
        Journal of Machine Learning Research, 25(72) (2024) https://jmlr.org/papers/v25/19-556.html
        arXiv preprint https://arxiv.org/abs/1507.02646

    .. [4] Yao et al. *Using stacking to average Bayesian predictive distributions*
        Bayesian Analysis, 13, 3 (2018). https://doi.org/10.1214/17-BA1091
        arXiv preprint https://arxiv.org/abs/1704.02030.
    """
    if round_to == "auto" or round_to is None:
        round_val = rcParams["stats.round_to"]
    else:
        round_val = round_to

    ics_dict = _calculate_ics(compare_dict, var_name=var_name)
    names = list(ics_dict.keys())

    has_subsampling = any(
        getattr(elpd, "subsample_size", None) is not None for elpd in ics_dict.values()
    )

    df_cols = {
        "rank": pd.Series(index=names, dtype="int"),
        "elpd": pd.Series(index=names, dtype="float"),
        "p": pd.Series(index=names, dtype="float"),
        "elpd_diff": pd.Series(index=names, dtype="float"),
        "weight": pd.Series(index=names, dtype="float"),
        "se": pd.Series(index=names, dtype="float"),
        "dse": pd.Series(index=names, dtype="float"),
        "warning": pd.Series(index=names, dtype="boolean"),
    }
    if has_subsampling:
        df_cols["subsampling_dse"] = pd.Series(index=names, dtype="float")

    df_comp = pd.DataFrame(df_cols)

    method = rcParams["stats.ic_compare_method"] if method is None else method
    available_methods = ["stacking", "bb-pseudo-bma", "pseudo-bma"]
    if method.lower() not in available_methods:
        raise ValueError(
            f"Invalid method '{method}'. "
            f"Available methods: {', '.join(available_methods)}. "
            f"Use 'stacking' for robust model averaging as recommended in the original paper "
            f"https://doi.org/10.1214/17-BA1091."
        )

    if reference is not None and reference not in names:
        raise ValueError(
            f"Reference model '{reference}' not found in compare_dict. "
            f"Available models: {', '.join(names)}"
        )

    ics = pd.DataFrame.from_dict(ics_dict, orient="index")
    ics.sort_values(by="elpd", inplace=True, ascending=False)
    ics["elpd_i"] = ics["elpd_i"].apply(lambda x: x.values.flatten())
    ses = ics["se"]

    if method.lower() == "stacking":
        rows, cols, ic_i_val = _ic_matrix(ics)
        exp_ic_i = np.exp(ic_i_val)

        def log_score(weights):
            return -np.sum(np.log(np.dot(exp_ic_i, weights)))

        def gradient(weights):
            denominator = np.dot(exp_ic_i, weights)
            return -np.sum(exp_ic_i / denominator[:, np.newaxis], axis=0)

        theta = np.full(cols, 1.0 / cols)
        bounds = Bounds(lb=np.zeros(cols), ub=np.ones(cols))
        constraints = LinearConstraint(np.ones(cols), lb=1.0, ub=1.0)

        minimize_result = minimize(
            fun=log_score, x0=theta, jac=gradient, bounds=bounds, constraints=constraints
        )
        weights = minimize_result["x"]

    elif method.lower() == "bb-pseudo-bma":
        b_samples = 1000
        rows, cols, ic_i_val = _ic_matrix(ics)
        ic_i_val = ic_i_val * rows

        b_weighting = dirichlet.rvs(alpha=[1] * rows, size=b_samples, random_state=124)
        weights = np.zeros((b_samples, cols))
        z_bs = np.zeros_like(weights)
        for i in range(b_samples):
            z_b = np.dot(b_weighting[i], ic_i_val)
            u_weights = np.exp(z_b - np.max(z_b))
            z_bs[i] = z_b  # pylint: disable=unsupported-assignment-operation
            weights[i] = u_weights / np.sum(u_weights)

        weights = weights.mean(axis=0)
        ses = pd.Series(z_bs.std(axis=0), index=ics.index)  # pylint: disable=no-member

    elif method.lower() == "pseudo-bma":
        min_ic = ics.iloc[0]["elpd"]
        z_rv = np.exp(ics["elpd"] - min_ic)
        weights = (z_rv / np.sum(z_rv)).to_numpy()

    if np.any(weights):
        ref_name = reference if reference is not None else ics.index[0]
        ref_elpd_data = ics_dict[ref_name]
        n_models = len(ics.index)
        mismatched_pairs = []

        for idx, val in enumerate(ics.index):
            res = ics.loc[val]
            current_elpd_data = ics_dict[val]

            if val == ref_name:
                d_ic = 0.0
                d_std_err = 0.0
                subsampling_d_std_err = 0.0 if has_subsampling else None
            else:
                diff_result = _compute_elpd_diff_subsampled(
                    ref_elpd_data,
                    current_elpd_data,
                )

                d_ic = diff_result["elpd_diff"]
                d_std_err = diff_result["se_diff"]
                subsampling_d_std_err = diff_result.get("subsampling_dse")

                if diff_result.get("subsample_mismatch"):
                    mismatched_pairs.append((ref_name, val))

            std_err = ses.loc[val]
            weight = weights[idx]

            row_data = [
                idx,
                res["elpd"],
                res["p"],
                d_ic,
                weight,
                std_err,
                d_std_err,
                res["warning"],
            ]
            if has_subsampling:
                row_data.append(subsampling_d_std_err)

            df_comp.loc[val] = row_data

        if mismatched_pairs:
            if n_models > 2:
                pairs_str = ", ".join([f"'{a}' and '{b}'" for a, b in mismatched_pairs])
                warnings.warn(
                    f"Different subsamples used in {pairs_str}. Naive diff SE is used.",
                    UserWarning,
                )
            else:
                for name_a, name_b in mismatched_pairs:
                    warnings.warn(
                        f"Different subsamples used in '{name_a}' and '{name_b}'. "
                        "Naive diff SE is used.",
                        UserWarning,
                    )

    df_comp["rank"] = df_comp["rank"].astype(int)
    df_comp["warning"] = df_comp["warning"].astype(bool)

    result = df_comp.sort_values(by="elpd", ascending=False)

    if round_to == "auto":
        result = _round_compare(result, round_val)
    else:
        if round_to not in ("None", "none"):
            cols_to_round = ["elpd", "p", "elpd_diff", "weight", "se", "dse"]
            result[cols_to_round] = result[cols_to_round].map(lambda x: round_num(x, round_val))

    model_order = list(ics.index)
    _order_stat_check(ics_dict, model_order, has_subsampling)

    return result


def _compute_elpd_diff_subsampled(elpd_a, elpd_b):
    """Compute ELPD differences for subsampled models."""
    subsample_a = getattr(elpd_a, "loo_subsample_observations", None)
    subsample_b = getattr(elpd_b, "loo_subsample_observations", None)
    mixed_subsample = (subsample_a is None) != (subsample_b is None)

    if subsample_a is None and subsample_b is None:
        pointwise_a = elpd_a.elpd_i
        pointwise_b = elpd_b.elpd_i

        if pointwise_a is None or pointwise_b is None:
            return _compute_naive_diff(elpd_a, elpd_b)

        diff = (np.asarray(pointwise_a) - np.asarray(pointwise_b)).reshape(-1)
        valid = diff[np.isfinite(diff)]

        if valid.size == 0:
            return _compute_naive_diff(elpd_a, elpd_b)

        elpd_diff = np.nansum(valid)
        se_diff = np.sqrt(valid.size * np.nanvar(valid))
        result = {"elpd_diff": elpd_diff, "se_diff": se_diff}

        subsampling_a = getattr(elpd_a, "subsampling_se", None) or 0.0
        subsampling_b = getattr(elpd_b, "subsampling_se", None) or 0.0
        combined = np.sqrt(subsampling_a**2 + subsampling_b**2)

        if combined:
            result["subsampling_dse"] = combined
        return result

    if mixed_subsample:
        warnings.warn(
            "Estimated elpd_diff using observations included in loo calculations for all models.",
            UserWarning,
        )

    indices_a = (
        np.unique(np.asarray(subsample_a, dtype=int))
        if subsample_a is not None
        else np.arange(elpd_a.n_data_points, dtype=int)
    )
    indices_b = (
        np.unique(np.asarray(subsample_b, dtype=int))
        if subsample_b is not None
        else np.arange(elpd_b.n_data_points, dtype=int)
    )

    shared = np.intersect1d(indices_a, indices_b)
    subsample_mismatch = (
        subsample_a is not None
        and subsample_b is not None
        and not np.array_equal(indices_a, indices_b)
    )

    if shared.size >= 2:
        result = _difference_estimator(elpd_a, elpd_b, shared, subsample_a, subsample_b)
        if result is not None:
            result["subsample_mismatch"] = subsample_mismatch
            return result

    result = _compute_naive_diff(elpd_a, elpd_b)
    result["subsample_mismatch"] = subsample_mismatch
    return result


def _difference_estimator(elpd_a, elpd_b, shared_indices, subsample_a=None, subsample_b=None):
    """Compute ELPD difference using the difference-of-estimators approach."""
    shared = np.asarray(shared_indices, dtype=int)
    ordered_shared = next(
        (
            candidate_arr[mask]
            for candidate in (subsample_a, subsample_b)
            if candidate is not None
            for candidate_arr in (np.asarray(candidate, dtype=int),)
            for mask in (np.isin(candidate_arr, shared),)
            if np.any(mask)
        ),
        shared,
    )

    if ordered_shared.size < 2:
        return None

    elpd_a_values = elpd_a.elpd_i
    elpd_b_values = elpd_b.elpd_i
    elpd_a_full = (
        None if elpd_a_values is None else np.asarray(elpd_a_values, dtype=float).reshape(-1)
    )
    elpd_b_full = (
        None if elpd_b_values is None else np.asarray(elpd_b_values, dtype=float).reshape(-1)
    )

    approx_a_values = getattr(elpd_a, "elpd_loo_approx", None)
    if approx_a_values is None:
        approx_a_values = elpd_a_values
    approx_a_full = (
        None if approx_a_values is None else np.asarray(approx_a_values, dtype=float).reshape(-1)
    )

    approx_b_values = getattr(elpd_b, "elpd_loo_approx", None)
    if approx_b_values is None:
        approx_b_values = elpd_b_values
    approx_b_full = (
        None if approx_b_values is None else np.asarray(approx_b_values, dtype=float).reshape(-1)
    )

    if any(
        component is None for component in (elpd_a_full, elpd_b_full, approx_a_full, approx_b_full)
    ):
        return None

    diff_sample = elpd_a_full[ordered_shared] - elpd_b_full[ordered_shared]
    diff_approx_sample = approx_a_full[ordered_shared] - approx_b_full[ordered_shared]
    diff_approx_all = approx_a_full - approx_b_full

    valid = np.isfinite(diff_sample) & np.isfinite(diff_approx_sample)
    if valid.sum() < 2:
        return None

    diff_sample = diff_sample[valid]
    diff_approx_sample = diff_approx_sample[valid]

    elpd_diff, subsampling_dse, se_diff = _diff_srs_estimator(
        diff_sample,
        diff_approx_sample,
        diff_approx_all,
        elpd_a.n_data_points,
    )

    return {
        "elpd_diff": elpd_diff,
        "se_diff": se_diff,
        "subsampling_dse": subsampling_dse,
    }


def _compute_naive_diff(elpd_a, elpd_b):
    """Compute naive ELPD difference using paired observations."""
    elpd_diff = elpd_a.elpd - elpd_b.elpd
    se_a = getattr(elpd_a, "se", 0.0)
    se_b = getattr(elpd_b, "se", 0.0)
    se_diff = np.sqrt(se_a**2 + se_b**2)

    result = {"elpd_diff": elpd_diff, "se_diff": se_diff}
    subsampling_a = getattr(elpd_a, "subsampling_se", None) or 0.0
    subsampling_b = getattr(elpd_b, "subsampling_se", None) or 0.0
    combined = np.sqrt(subsampling_a**2 + subsampling_b**2)

    if combined:
        result["subsampling_dse"] = combined
    return result


def _ic_matrix(ics):
    """Store the previously computed pointwise predictive accuracy values (ics) in a 2D matrix."""
    cols, _ = ics.shape
    rows = len(ics["elpd_i"].iloc[0])
    ic_i_val = np.zeros((rows, cols))

    mismatches = []
    for val in ics.index:
        ic_len = len(ics.loc[val]["elpd_i"])
        if ic_len != rows:
            mismatches.append((val, ic_len))

    if mismatches:
        obs_counts = {name: len(ics.loc[name]["elpd_i"]) for name in ics.index}
        sorted_counts = sorted(obs_counts.items(), key=lambda item: (item[1], item[0]))
        mismatch_details = ", ".join([f"'{name}' ({count})" for name, count in sorted_counts])
        raise ValueError(
            "All models must have the same number of observations, but models have inconsistent "
            f"observation counts: {mismatch_details}"
        )

    for idx, val in enumerate(ics.index):
        ic = ics.loc[val]["elpd_i"]
        ic_i_val[:, idx] = ic

    return rows, cols, ic_i_val


def _calculate_ics(
    compare_dict,
    var_name=None,
):
    """Calculate LOO only if necessary.

    It always calls LOO with ``pointwise=True``.

    Parameters
    ----------
    compare_dict :  dict of {str : DataTree or ELPDData}
        A dictionary of model names and DataTree or ELPDData objects.
    var_name : str, optional
        Name of the variable storing pointwise log likelihood values in ``log_likelihood`` group.


    Returns
    -------
    compare_dict : dict of ELPDData
    """
    precomputed_elpds = {
        name: elpd_data
        for name, elpd_data in compare_dict.items()
        if isinstance(elpd_data, ELPDData)
    }
    if precomputed_elpds:
        for name, elpd_data in precomputed_elpds.items():
            if elpd_data.elpd_i is None:
                raise ValueError(
                    f"Model '{name}' is missing pointwise ELPD values. "
                    f"Recalculate with pointwise=True."
                )

        methods_used = {}
        for name, elpd_data in precomputed_elpds.items():
            kind = elpd_data.kind
            if kind not in methods_used:
                methods_used[kind] = []
            methods_used[kind].append(name)

        if len(methods_used) > 1:
            has_loo = "loo" in methods_used
            has_kfold = "loo_kfold" in methods_used
            has_lfo = "lfo_cv" in methods_used

            # LFO-CV should only be compared with other LFO-CV results
            if has_lfo and (has_loo or has_kfold):
                method_list = sorted(methods_used.keys())
                raise ValueError(
                    f"Cannot compare LFO-CV results with other cross-validation methods: "
                    f"{method_list}. LFO-CV evaluates time series predictive accuracy and "
                    f"should only be compared with other LFO-CV results."
                )
            elif has_loo and has_kfold and len(methods_used) == 2:
                warnings.warn(
                    "Comparing LOO-CV to K-fold-CV. "
                    "For a more accurate comparison use the same number of folds "
                    "or loo for all models compared.",
                    UserWarning,
                )
            elif not (has_loo or has_kfold or has_lfo):
                method_list = sorted(methods_used.keys())
                raise ValueError(
                    f"Cannot compare models with incompatible cross-validation methods: "
                    f"{method_list}. Only 'loo', 'loo_kfold', and 'lfo_cv' methods "
                    f"are supported currently."
                )

    new_compare_dict = deepcopy(compare_dict)
    for name, dataset in compare_dict.items():
        if not isinstance(dataset, ELPDData):
            try:
                new_compare_dict[name] = loo(
                    dataset,
                    pointwise=True,
                    var_name=var_name,
                )
            except Exception as e:
                raise e.__class__(
                    f"Encountered error trying to compute ELPD from model {name}."
                ) from e
    return new_compare_dict


def _order_stat_check(ics_dict, model_order, has_subsampling):
    """Perform order statistics-based checks on models."""
    if has_subsampling or len(ics_dict) <= 11:
        return

    # Use the median model as the baseline model to compute ELPD differences
    baseline_idx = len(model_order) // 2
    baseline_model = model_order[baseline_idx]
    baseline_elpd = ics_dict[baseline_model]

    elpd_diffs = np.zeros(len(model_order))
    for idx, model_name in enumerate(model_order):
        if model_name != baseline_model:
            elpd_a_vals = np.ravel(baseline_elpd.elpd_i)
            elpd_b_vals = np.ravel(ics_dict[model_name].elpd_i)
            elpd_diffs[idx] = np.sum(elpd_b_vals - elpd_a_vals)

    elpd_diffs = np.array(elpd_diffs)
    diff_median = np.median(elpd_diffs)
    elpd_diff_trunc = elpd_diffs[elpd_diffs >= diff_median]
    n_models = np.sum(~np.isnan(elpd_diff_trunc))

    if n_models < 1:
        return

    candidate_sd = np.sqrt(1 / n_models * np.sum(elpd_diff_trunc**2))

    # Defensive check to avoid a runtime error when computing the order statistic
    if candidate_sd == 0 or not np.isfinite(candidate_sd):
        warnings.warn(
            "All models have nearly identical performance.",
            UserWarning,
        )
        return

    # Estimate expected best diff under null hypothesis
    k = len(ics_dict) - 1
    order_stat = norm.ppf(1 - 1 / (k * 2), loc=0, scale=candidate_sd)

    if np.nanmax(elpd_diffs) <= order_stat:
        warnings.warn(
            "Difference in performance potentially due to chance. "
            "See https://doi.org/10.1007/s11222-024-10442-4 for details.",
            UserWarning,
        )


def _round_compare(result, round_val):
    """Apply custom rounding rules to compare.

    Parameters
    ----------
    result : pandas.DataFrame
        The compare result to round
    round_val : int or str
        Number of decimals or significant figures to round to.

    Returns
    -------
    pandas.DataFrame
    """
    stat_se_pairs = [
        ("elpd", "se"),
        ("elpd_diff", "dse"),
    ]

    for stat_col, se_col in stat_se_pairs:
        result[se_col] = result[se_col].apply(lambda x: round_num(x, round_val))

        for idx in result.index:
            stat_val = result.loc[idx, stat_col]
            se_val = result.loc[idx, se_col]

            decimal_places = get_decimal_places_from_se(se_val)
            result.loc[idx, stat_col] = round_num(stat_val, decimal_places)

    if "p" in result.columns:
        result["p"] = result["p"].apply(lambda x: round_num(x, 1))

    if "weight" in result.columns:
        max_weight = result["weight"].to_numpy(dtype=float).max()
        decimals = int(np.ceil(-np.log10(max_weight))) + 1
        result["weight"] = result["weight"].apply(lambda x: round_num(x, decimals))

    return result
