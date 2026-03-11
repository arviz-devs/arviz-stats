"""Compute influential observations based on leave-one-out expectations."""

import numpy as np
from arviz_base import extract

from arviz_stats.loo import loo_expectations
from arviz_stats.summary import mad, mean, median, std, var
from arviz_stats.validate import validate_dims


def loo_influence(
    data,
    var_names=None,
    group="posterior_predictive",
    sample_dims=None,
    log_likelihood_var_name=None,
    kind="mean",
    standardize=True,
    probs=None,
    log_weights=None,
    pareto_k=None,
):
    """Compute influential observations based on leave-one-out (LOO) expectations.

    Computes observation influence by measuring the change in posterior or posterior predictive
    summaries when leaving out each observation. The function supports various summary statistics.

    Parameters
    ----------
    data: DataTree or InferenceData
        It should contain the selected `group` and `log_likelihood`.
    var_names: str or list of str, optional
        The name(s) of the variable(s) to compute the influence.
    group: str
        Group from which to compute weighted expectations. Defaults to ``posterior_predictive``.
    sample_dims : str or sequence of hashable, optional
        Defaults to ``rcParams["data.sample_dims"]``
    log_likelihood_var_name: str, optional
        The name of the variable in the log_likelihood group to use for loo computation.
        When log_likelihood contains more than one variable and group is ``posterior``,
        this must be provided.
    kind: str, optional
        The kind of expectation to compute. Available options are:

        - 'mean'. Default.
        - 'median'.
        - 'sd'.
        - 'var'.
        - 'quantile'.
        - 'octiles'.
    standardize: bool
        Whether to standardize the computed metric.
        It uses the standard deviation when ``kind=mean`` and MAD when ``kind=median``.
        Ignored for the other values of kind.
    probs: float or list of float, optional
        The quantile(s) to compute when kind is 'quantile'.
    log_weights : DataArray, optional
        Pre-computed smoothed log weights from PSIS. Must be provided together with pareto_k.
        If not provided, PSIS will be computed internally.
    pareto_k : DataArray, optional
        Pre-computed Pareto k-hat diagnostic values. Must be provided together with log_weights.

    Returns
    -------
    shift : DataArray or Dataset
       Influential metric
    khat : DataArray or Dataset
        Function-specific Pareto k-hat diagnostics for each observation.

    Examples
    --------
    Calculate influential observations based on the posterior median for the parameter ``mu``:

    .. ipython::

        In [1]: from arviz_stats import loo_influence
           ...: from arviz_base import load_arviz_data
           ...: dt = load_arviz_data("centered_eight")
           ...: shift, _ = loo_influence(dt, kind="median", var_names="mu", group="posterior")
           ...: shift

    Calculate influential observations based on 3 quantiles of the posterior predictive:

    .. ipython::

        In [2]: shift, khat = loo_influence(dt, kind="quantile", probs=[0.25, 0.5, 0.75])
           ...: shift
    """
    sample_dims = validate_dims(sample_dims)

    if group not in ["posterior_predictive", "posterior"]:
        raise ValueError("group must be either 'posterior_predictive' or 'posterior'")

    _validkinds = (
        "mean",
        "median",
        "sd",
        "var",
        "quantile",
        "octiles",
    )

    if kind not in _validkinds:
        raise ValueError(f"kind must be one of {_validkinds}, got {kind}")

    if kind == "octiles":
        probs = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
    elif kind == "quantile" and probs is None:
        raise ValueError("probs must be provided when kind is 'quantile'")

    if kind in ["quantile", "octiles"]:
        loo_expec, khat = loo_expectations(
            data,
            var_name=var_names,
            group=group,
            sample_dims=sample_dims,
            log_likelihood_var_name=log_likelihood_var_name,
            kind="quantile",
            probs=probs,
            log_weights=log_weights,
            pareto_k=pareto_k,
        )
        group_data = extract(data, var_names=var_names, group=group, combined=False)
        shift = np.abs(loo_expec - group_data.quantile(probs, dim=sample_dims)).mean("quantile")
    else:
        loo_expec, khat = loo_expectations(
            data,
            var_name=var_names,
            group=group,
            sample_dims=sample_dims,
            log_likelihood_var_name=log_likelihood_var_name,
            kind=kind,
            log_weights=log_weights,
            pareto_k=pareto_k,
        )

        func = None
        func_s = None
        if kind == "mean":
            func = mean
            func_s = std
        elif kind == "median":
            func = median
            func_s = mad
        elif kind == "sd":
            func = std
        elif kind == "var":
            func = var

        shift = np.abs(
            loo_expec
            - func(data, group=group, var_names=var_names, dim=sample_dims, round_to="none").dataset
        )

        if standardize and kind in ["mean", "median"]:
            shift /= func_s(
                data, group=group, var_names=var_names, dim=sample_dims, round_to="none"
            ).dataset

    return shift, khat
