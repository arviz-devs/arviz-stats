"""Continuously ranked probability scores with PSIS-LOO-CV weights."""

from collections import namedtuple

import numpy as np
from arviz_base import convert_to_datatree, extract

from arviz_stats.base.stats_utils import round_num
from arviz_stats.loo.helper_loo import (
    _get_r_eff,
    _prepare_loo_inputs,
    _validate_crps_input,
    _warn_pareto_k,
)


def loo_score(
    data,
    var_name=None,
    kind="crps",
    pointwise=False,
    round_to=None,
    log_weights=None,
    pareto_k=None,
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
        - 2\cdot E_{\text{loo}} \left[X\,F_{\text{mid}}(X) \right],

    where :math:`F_{\text{mid}}` is the midpoint CDF estimator defined as
    :math:`F_{\text{mid}}(x_{(i)}) := F^-_{(i)} + w_{(i)}/2`. This midpoint formulation
    provides improved accuracy for weighted samples compared to the left-continuous CDF.

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
    kind : str, default "crps"
        The kind of score to compute. Available options are:

        - 'crps': continuous ranked probability score. Default.
        - 'scrps': scale-invariant continuous ranked probability score.
    pointwise : bool, default False
        If True, include per-observation score values in the return object.
    round_to : int or str, optional
        If integer, number of decimal places to round the result. If string of the
        form '2g' number of significant digits to round the result. Defaults to '2g'.
        Use None to return raw numbers.
    log_weights : DataArray, optional
        Pre-computed smoothed log weights from PSIS. Must be provided together with pareto_k.
        If not provided, PSIS will be computed internally.
    pareto_k : DataArray, optional
        Pre-computed Pareto k-hat diagnostic values. Must be provided together with log_weights.

    Returns
    -------
    namedtuple
        If ``pointwise`` is False (default), a namedtuple named ``CRPS`` or ``SCRPS`` with fields
        ``mean`` and ``se``. If ``pointwise`` is True, the namedtuple also includes ``pointwise``
        and ``pareto_k`` fields.

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

    Notes
    -----
    For a single observation with posterior-predictive draws :math:`x_1, \ldots, x_S`
    and PSIS-LOO-CV weights :math:`w_i \propto \exp(\ell_i)` normalized so that
    :math:`\sum_{i=1}^S w_i = 1`, define the PSIS-LOO-CV expectation as

    .. math::

        E_{\text{loo}}[g(X)] := \sum_{i=1}^S w_i\, g(x_i).

    For weighted samples, we use the midpoint CDF estimator rather than the left-continuous CDF.
    Given sorted values :math:`x_{(1)} \leq \cdots \leq x_{(S)}` with corresponding weights
    :math:`w_{(i)}`, define the left-cumulative weight :math:`F^-_{(i)} = \sum_{j<i} w_{(j)}`
    and the midpoint CDF as

    .. math::

        F_{\text{mid}}(x_{(i)}) := F^-_{(i)} + \frac{w_{(i)}}{2}.

    The first probability-weighted moment using the midpoint CDF is
    :math:`b_1 := \sum_{i=1}^S w_{(i)}\, x_{(i)}\, F_{\text{mid}}(x_{(i)})`.
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

    if log_weights is not None and pareto_k is not None:
        pointwise_scores, pareto_k_out = y_pred.azstats.loo_score(
            y_obs=y_obs,
            log_weights=log_weights,
            pareto_k=pareto_k,
            kind=kind,
            r_eff=r_eff,
            sample_dims=sample_dims,
        )
    else:
        log_ratios = -log_likelihood
        pointwise_scores, pareto_k_out = y_pred.azstats.loo_score(
            y_obs=y_obs, log_ratios=log_ratios, kind=kind, r_eff=r_eff, sample_dims=sample_dims
        )

    _warn_pareto_k(pareto_k_out, n_samples)

    n_pts = int(np.prod([pointwise_scores.sizes[d] for d in pointwise_scores.dims]))
    mean = pointwise_scores.mean().values.item()
    se = (pointwise_scores.std(ddof=0).values / (n_pts**0.5)).item()
    name = "SCRPS" if kind == "scrps" else "CRPS"

    if pointwise:
        return namedtuple(name, ["mean", "se", "pointwise", "pareto_k"])(
            round_num(mean, round_to),
            round_num(se, round_to),
            pointwise_scores,
            pareto_k_out,
        )
    return namedtuple(name, ["mean", "se"])(
        round_num(mean, round_to),
        round_num(se, round_to),
    )
