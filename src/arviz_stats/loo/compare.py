"""Compare PSIS-LOO-CV results."""

import itertools
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
from arviz_base import rcParams
from scipy.optimize import minimize
from scipy.stats import dirichlet

from arviz_stats.loo import loo
from arviz_stats.utils import ELPDData


def compare(
    compare_dict,
    method="stacking",
    var_name=None,
):
    r"""Compare models based on their expected log pointwise predictive density (ELPD).

    The ELPD is estimated by Pareto smoothed importance sampling leave-one-out
    cross-validation, the same method used by :func:`arviz_stats.loo`.
    The method is described in [1]_ and [2]_.
    By default, the weights are estimated using ``"stacking"`` as described in [3]_.

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

    Returns
    -------
    DataFrame
        A DataFrame, ordered from best to worst model (measured by the ELPD).
        The index reflects the key with which the models are passed to this function.
        The columns are:

        - **rank**: The rank-order of the models. 0 is the best.
        - **elpd**: ELPD estimated either using (PSIS-LOO-CV `elpd_loo` or WAIC `elpd_waic`).
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

    See Also
    --------
    :func:`loo` : Compute the ELPD using the Pareto smoothed importance sampling Leave-one-out
        cross-validation method.
    :func:`arviz_plots.plot_compare`: Summary plot for model comparison.

    References
    ----------

    .. [1] Vehtari et al. *Practical Bayesian model evaluation using leave-one-out cross-validation
        and WAIC*. Statistics and Computing. 27(5) (2017) https://doi.org/10.1007/s11222-016-9696-4
        arXiv preprint https://arxiv.org/abs/1507.04544.

    .. [2] Vehtari et al. *Pareto Smoothed Importance Sampling*.
        Journal of Machine Learning Research, 25(72) (2024) https://jmlr.org/papers/v25/19-556.html
        arXiv preprint https://arxiv.org/abs/1507.02646

    .. [3] Yao et al. *Using stacking to average Bayesian predictive distributions*
        Bayesian Analysis, 13, 3 (2018). https://doi.org/10.1214/17-BA1091
        arXiv preprint https://arxiv.org/abs/1704.02030.
    """
    ics_dict = _calculate_ics(compare_dict, var_name=var_name)
    names = list(ics_dict.keys())

    df_comp = pd.DataFrame(
        {
            "rank": pd.Series(index=names, dtype="int"),
            "elpd": pd.Series(index=names, dtype="float"),
            "p": pd.Series(index=names, dtype="float"),
            "elpd_diff": pd.Series(index=names, dtype="float"),
            "weight": pd.Series(index=names, dtype="float"),
            "se": pd.Series(index=names, dtype="float"),
            "dse": pd.Series(index=names, dtype="float"),
            "warning": pd.Series(index=names, dtype="boolean"),
        }
    )

    method = rcParams["stats.ic_compare_method"] if method is None else method
    available_methods = ["stacking", "bb-pseudo-bma", "pseudo-bma"]
    if method.lower() not in available_methods:
        raise ValueError(
            f"Invalid method '{method}'. "
            f"Available methods: {', '.join(available_methods)}. "
            f"Use 'stacking' for robust model averaging as recommended in the original paper "
            f"https://doi.org/10.1214/17-BA1091."
        )

    ics = pd.DataFrame.from_dict(ics_dict, orient="index")
    ics.sort_values(by="elpd", inplace=True, ascending=False)
    ics["elpd_i"] = ics["elpd_i"].apply(lambda x: x.values.flatten())
    ses = ics["se"]

    if method.lower() == "stacking":
        rows, cols, ic_i_val = _ic_matrix(ics)
        exp_ic_i = np.exp(ic_i_val)
        km1 = cols - 1

        def w_fuller(weights):
            return np.concatenate((weights, [max(1.0 - np.sum(weights), 0.0)]))

        def log_score(weights):
            w_full = w_fuller(weights)
            score = 0.0
            for i in range(rows):
                score += np.log(np.dot(exp_ic_i[i], w_full))
            return -score

        def gradient(weights):
            w_full = w_fuller(weights)
            grad = np.zeros(km1)
            for k, i in itertools.product(range(km1), range(rows)):
                grad[k] += (exp_ic_i[i, k] - exp_ic_i[i, km1]) / np.dot(exp_ic_i[i], w_full)
            return -grad

        theta = np.full(km1, 1.0 / cols)
        bounds = [(0.0, 1.0) for _ in range(km1)]
        constraints = [
            {"type": "ineq", "fun": lambda x: -np.sum(x) + 1.0},
            {"type": "ineq", "fun": np.sum},
        ]

        minimize_result = minimize(
            fun=log_score, x0=theta, jac=gradient, bounds=bounds, constraints=constraints
        )

        weights = w_fuller(minimize_result["x"])

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
        min_ic_i_val = ics["elpd_i"].iloc[0]
        for idx, val in enumerate(ics.index):
            res = ics.loc[val]
            diff = min_ic_i_val - res["elpd_i"]
            d_ic = np.sum(diff)
            d_std_err = np.sqrt(len(diff) * np.var(diff))
            std_err = ses.loc[val]
            weight = weights[idx]
            df_comp.loc[val] = (
                idx,
                res["elpd"],
                res["p"],
                d_ic,
                weight,
                std_err,
                d_std_err,
                res["warning"],
            )

    df_comp["rank"] = df_comp["rank"].astype(int)
    df_comp["warning"] = df_comp["warning"].astype(bool)
    return df_comp.sort_values(by="elpd", ascending=False)


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

            if has_loo and has_kfold and len(methods_used) == 2:
                warnings.warn(
                    "Comparing LOO-CV to K-fold-CV. "
                    "For a more accurate comparison use the same number of folds "
                    "or loo for all models compared.",
                    UserWarning,
                )
            else:
                method_list = sorted(methods_used.keys())
                raise ValueError(
                    f"Cannot compare models with incompatible cross-validation methods: "
                    f"{method_list}. Only comparisons between 'loo' and 'loo_kfold' methods "
                    f"are supported currently."
                )

    compare_dict = deepcopy(compare_dict)
    for name, dataset in compare_dict.items():
        if not isinstance(dataset, ELPDData):
            try:
                compare_dict[name] = loo(
                    dataset,
                    pointwise=True,
                    var_name=var_name,
                )
            except Exception as e:
                raise e.__class__(
                    f"Encountered error trying to compute ELPD from model {name}."
                ) from e
    return compare_dict
