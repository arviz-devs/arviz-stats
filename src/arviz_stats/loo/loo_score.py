"""Continuously ranked probability scores with PSIS-LOO-CV weights."""

from collections import namedtuple

import numpy as np
import xarray as xr
from arviz_base import convert_to_datatree, extract
from xarray_einstats.stats import logsumexp

from arviz_stats.loo.helper_loo import (
    _get_r_eff,
    _prepare_loo_inputs,
    _validate_crps_input,
    _warn_pareto_k,
)
from arviz_stats.utils import round_num


def loo_score(
    data,
    var_name=None,
    log_weights=None,
    pareto_k=None,
    kind="crps",
    pointwise=False,
    round_to="2g",
):
    r"""Compute PWM-based CRPS/SCRPS with PSIS-LOO-CV weights.

    Implements the probability-weighted-moment (PWM) identity for the continuous ranked
    probability score (CRPS) with Pareto-smoothed importance sampling leave-one-out (PSIS-LOO-CV)
    weights, but returns its negative as a maximization score (larger is better). This assumes
    that the PSIS-LOO-CV approximation is working well.

    Specifically, the PWM identity used here is

    .. math::

        \operatorname{CRPS}_{\text{loo}}(F, y)
        = E_{\text{loo}}\left[|X - y|\right]
        + E_{\text{loo}}[X]
        - 2\cdot E_{\text{loo}} \left[X\,F_{\text{loo}}(X') \right].

    The PWM identity is described in [3]_, traditional CRPS and SCRPS are described in
    [1]_ and [2]_, and the PSIS-LOO-CV method is described in [4]_ and [5]_.

    Parameters
    ----------
    data : DataTree or InferenceData
        Input data. It should contain the ``posterior_predictive``, ``observed_data`` and
        ``log_likelihood`` groups.
    var_name : str, optional
        The name of the variable in the log_likelihood group to use. If None, the first
        variable in ``observed_data`` is used and assumed to match ``log_likelihood`` and
        ``posterior_predictive`` names.
    log_weights : DataArray, optional
        Smoothed log weights for PSIS-LOO-CV. Must have the same shape as the log-likelihood data.
        Defaults to None. If not provided, they will be computed via PSIS-LOO-CV. Must be provided
        together with ``pareto_k`` or both must be None.
    pareto_k : DataArray, optional
        Pareto tail indices corresponding to the PSIS smoothing. Same shape as the log-likelihood
        data. If not provided, they will be computed via PSIS-LOO-CV. Must be provided together with
        ``log_weights`` or both must be None.
    kind : str, default "crps"
        The kind of score to compute. Available options are:

        - 'crps': continuous ranked probability score. Default.
        - 'scrps': scale-invariant continuous ranked probability score.
    pointwise : bool, default False
        If True, include per-observation score values in the return object.
    round_to : int or str, default "2g"
        If integer, number of decimal places to round the result. If string of the form ``"2g"``,
        number of significant digits to round the result. Use None to return raw numbers.

    Returns
    -------
    namedtuple
        If ``pointwise`` is False (default), a namedtuple named ``CRPS`` or ``SCRPS`` with fields
        ``mean`` and ``se``. If ``pointwise`` is True, the namedtuple also includes a ``pointwise``
        field with per-observation values.

    Examples
    --------
    Compute scores and return the mean and standard error:

    .. ipython::
        :okwarning:

        In [1]: from arviz_stats import loo_score
           ...: from arviz_base import load_arviz_data
           ...: dt = load_arviz_data("centered_eight")
           ...: loo_score(dt, kind="crps")

    .. ipython::
        :okwarning:

        In [2]: loo_score(dt, kind="scrps")

    We can also pass previously computed PSIS-LOO weights and return the pointwise values:

    .. ipython::
        :okwarning:

        In [3]: from arviz_stats import loo
           ...: loo_data = loo(dt, pointwise=True)
           ...: loo_score(dt, kind="crps",
           ...:           log_weights=loo_data.log_weights,
           ...:           pareto_k=loo_data.pareto_k,
           ...:           pointwise=True)

    Notes
    -----
    For a single observation with posterior-predictive draws :math:`x_1, \ldots, x_S`
    and PSIS-LOO-CV weights :math:`w_i \propto \exp(\ell_i)` normalized so that
    :math:`\sum_{i=1}^S w_i = 1`, define the PSIS-LOO-CV expectation and the left-continuous
    weighted CDF as

    .. math::

        E_{\text{loo}}[g(X)] := \sum_{i=1}^S w_i\, g(x_i), \quad
        F_{\text{loo}}(x') := \sum_{i: x_i < x} w_i.

    The first probability-weighted moment is
    :math:`b_1 := E_{\text{loo}}\left[X\,F_{\text{loo}}(X')\right]`.
    With this, the nonnegative CRPS under PSIS-LOO-CV is

    .. math::

        \operatorname{CRPS}_{\text{loo}}(F, y)
        = E_{\text{loo}}\left[\,|X-y|\,\right]
        + E_{\text{loo}}[X] - 2\,b_1.

    For the scale term for the SCRPS, we use the PSIS-LOO-CV weighted Gini mean difference given by
    :math:`\Delta_{\text{loo}} := E_{\text{loo}}\left[\,|X - X'|\,\right]`.
    This admits the PWM representation given by

    .. math::

        \Delta_{\text{loo}} =
        2\,E_{\text{loo}}\left[\,X\,\left(2F_{\text{loo}}(X') - 1\right)\,\right].

    A finite-sample weighted order-statistic version of this is used in the function and is given by

    .. math::

        \Delta_{\text{loo}} =
        2 \sum_{i=1}^S w_{(i)}\, x_{(i)} \left\{\,2 F^-_{(i)} + w_{(i)} - 1\,\right\},

    where :math:`x_{(i)}` are the values sorted increasingly, :math:`w_{(i)}` are the
    corresponding normalized weights, and :math:`F^-_{(i)} = \sum_{j<i} w_{(j)}`.

    The locally scale-invariant score returned for ``kind="scrps"`` is

    .. math::

        S_{\text{SCRPS}}(F, y)
        = -\frac{E_{\text{loo}}\left[\,|X-y|\,\right]}{\Delta_{\text{loo}}}
        - \frac{1}{2}\log \Delta_{\text{loo}}.

    When PSIS weights are highly variable (large Pareto :math:`k`), Monte-Carlo noise can
    increase. This function surfaces PSIS-LOO-CV diagnostics via ``pareto_k`` and warns when
    tail behavior suggests unreliability.

    References
    ----------

    .. [1] Bolin, D., & Wallin, J. (2023). *Local scale invariance and robustness of
       proper scoring rules*. Statistical Science, 38(1), 140–159. https://doi.org/10.1214/22-STS864
       arXiv preprint https://arxiv.org/abs/1912.05642
    .. [2] Gneiting, T., & Raftery, A. E. (2007). *Strictly Proper Scoring Rules,
       Prediction, and Estimation*. Journal of the American Statistical Association,
       102(477), 359–378. https://doi.org/10.1198/016214506000001437
    .. [3] Taillardat M, Mestre O, Zamo M, Naveau P (2016). *Calibrated ensemble forecasts using
       quantile regression forests and ensemble model output statistics*. Mon Weather Rev
       144(6):2375–2393. https://doi.org/10.1175/MWR-D-15-0260.1
    .. [4] Vehtari, A., Gelman, A., & Gabry, J. (2017). *Practical Bayesian model
       evaluation using leave-one-out cross-validation and WAIC*. Statistics and Computing,
       27(5), 1413–1432. https://doi.org/10.1007/s11222-016-9696-4
       arXiv preprint https://arxiv.org/abs/1507.04544
    .. [5] Vehtari, A., et al. (2024). *Pareto Smoothed Importance Sampling*. Journal of
       Machine Learning Research, 25(72). https://jmlr.org/papers/v25/19-556.html
       arXiv preprint https://arxiv.org/abs/1507.02646
    """
    if kind not in {"crps", "scrps"}:
        raise ValueError(f"kind must be either 'crps' or 'scrps'. Got {kind}")

    data = convert_to_datatree(data)

    loo_inputs = _prepare_loo_inputs(data, var_name)
    var_name = loo_inputs.var_name
    log_likelihood = loo_inputs.log_likelihood

    y_pred = extract(data, group="posterior_predictive", var_names=var_name, combined=False)
    y_obs = extract(data, group="observed_data", var_names=var_name, combined=False)

    n_samples = loo_inputs.n_samples
    sample_dims = loo_inputs.sample_dims
    obs_dims = loo_inputs.obs_dims
    r_eff = _get_r_eff(data, n_samples)

    _validate_crps_input(y_pred, y_obs, log_likelihood, sample_dims=sample_dims, obs_dims=obs_dims)

    if (log_weights is None) != (pareto_k is None):
        raise ValueError(
            "Both log_weights and pareto_k must be provided together or both must be None. "
            "Only one was provided."
        )

    if log_weights is None and pareto_k is None:
        log_weights_da, pareto_k = log_likelihood.azstats.psislw(r_eff=r_eff, dim=sample_dims)
    else:
        log_weights_da = log_weights

    abs_error = np.abs(y_pred - y_obs)

    loo_weighted_abs_error = _loo_weighted_mean(abs_error, log_weights_da, sample_dims)
    loo_weighted_mean_prediction = _loo_weighted_mean(y_pred, log_weights_da, sample_dims)
    pwm_first_moment_b1 = _apply_pointwise_weighted_statistic(
        y_pred, log_weights_da, sample_dims, _compute_pwm_first_moment_b1
    )

    crps_pointwise = (
        loo_weighted_abs_error + loo_weighted_mean_prediction - 2.0 * pwm_first_moment_b1
    )

    if kind == "crps":
        pointwise_scores = -crps_pointwise
        khat_da = pareto_k
    else:
        gini_mean_difference = _apply_pointwise_weighted_statistic(
            y_pred, log_weights_da, sample_dims, _compute_weighted_gini_mean_difference
        )
        pointwise_scores = -(loo_weighted_abs_error / gini_mean_difference) - 0.5 * np.log(
            gini_mean_difference
        )
        khat_da = pareto_k

    _warn_pareto_k(khat_da, n_samples)

    n_pts = int(np.prod([pointwise_scores.sizes[d] for d in pointwise_scores.dims]))
    mean = pointwise_scores.mean().values.item()
    se = (pointwise_scores.std(ddof=0).values / (n_pts**0.5)).item()
    name = "SCRPS" if kind == "scrps" else "CRPS"

    if pointwise:
        return namedtuple(name, ["mean", "se", "pointwise"])(
            round_num(mean, round_to),
            round_num(se, round_to),
            pointwise_scores,
        )
    return namedtuple(name, ["mean", "se"])(
        round_num(mean, round_to),
        round_num(se, round_to),
    )


def _compute_pwm_first_moment_b1(values_sorted, weights):
    """Compute first PWM using a left-continuous weighted CDF."""
    values_sorted, weights_sorted = _sort_values_and_normalize_weights(values_sorted, weights)
    cumulative_weights = np.cumsum(weights_sorted)
    f_minus = cumulative_weights - weights_sorted
    return np.sum(weights_sorted * values_sorted * f_minus).item()


def _compute_weighted_gini_mean_difference(values, weights):
    """Compute PSIS-LOO-CV weighted Gini mean difference."""
    values_sorted, weights_sorted = _sort_values_and_normalize_weights(values, weights)
    cumulative_weights = np.cumsum(weights_sorted)
    cumulative_before = cumulative_weights - weights_sorted
    bracket = 2.0 * cumulative_before + weights_sorted - 1.0
    return (2.0 * np.sum(weights_sorted * values_sorted * bracket)).item()


def _loo_weighted_mean(values, log_weights, dim):
    """Compute PSIS-LOO-CV weighted mean."""
    log_num = logsumexp(log_weights, dims=dim, b=values)
    log_den = logsumexp(log_weights, dims=dim)
    return np.exp(log_num - log_den)


def _apply_pointwise_weighted_statistic(x, log_weights, sample_dims, stat_func):
    """Apply a weighted statistic over sample dims."""
    max_logw = log_weights.max(dim=sample_dims)
    weights = np.exp(log_weights - max_logw)
    stacked = "__sample__"
    xs = x.stack({stacked: sample_dims})
    ws = weights.stack({stacked: sample_dims})
    return xr.apply_ufunc(
        stat_func,
        xs,
        ws,
        input_core_dims=[[stacked], [stacked]],
        output_core_dims=[[]],
        vectorize=True,
        output_dtypes=[float],
    )


def _sort_values_and_normalize_weights(values, weights):
    """Sort values by ascending order and normalize weights."""
    idx = np.argsort(values, kind="mergesort")
    values_sorted = values[idx]
    weights_sorted = weights[idx]
    weights_sorted = weights_sorted / np.sum(weights_sorted)
    return values_sorted, weights_sorted
