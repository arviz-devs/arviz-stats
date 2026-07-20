"""Continuously ranked probability scores with PSIS-LOO-CV weights."""

from collections import namedtuple

import numpy as np
from arviz_base import convert_to_datatree, extract

from arviz_stats.base.stats_utils import round_num
from arviz_stats.loo.loo_helper import (
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
    r"""Compute CRPS or SCRPS with PSIS-LOO-CV weights.

    Computes the continuous ranked probability score (CRPS) or its scale-invariant variant
    (SCRPS) using Pareto-smoothed importance sampling leave-one-out (PSIS-LOO-CV) weights, and
    returns its negative as a maximization score (larger is better). This assumes that the
    PSIS-LOO-CV approximation is working well.

    Writing :math:`E_{\text{loo}}` for the PSIS-LOO-CV weighted expectation over
    posterior-predictive draws :math:`X` (with :math:`X'` an independent copy), the CRPS is

    .. math::

        \operatorname{CRPS}_{\text{loo}}(F, y)
        = E_{\text{loo}}\left[|X - y|\right]
        - \frac{1}{2}\, E_{\text{loo}}\left[|X - X'|\right],

    and the SCRPS is

    .. math::

        S_{\text{SCRPS}}(F, y)
        = -\frac{E_{\text{loo}}\left[|X - y|\right]}{E_{\text{loo}}\left[|X - X'|\right]}
        - \frac{1}{2}\log E_{\text{loo}}\left[|X - X'|\right].

    Both are evaluated with the probability-weighted-moment (PWM) estimator of [3]_, which
    computes them exactly in :math:`\mathcal{O}(S \log S)` from a single set of draws.
    Traditional CRPS and SCRPS are described in [1]_ and [2]_, and the PSIS-LOO-CV method in
    [4]_ and [5]_.

    Parameters
    ----------
    data : DataTree or InferenceData
        Input data. It should contain the ``posterior_predictive``, ``observed_data`` and
        ``log_likelihood`` groups. A ``posterior`` group is also required when
        ``log_weights`` and ``pareto_k`` are not provided, as it is used to compute the
        relative efficiency of the importance sampling estimate.
    var_name : str, optional
        The name of the variable in the log_likelihood group to use. If None, the only
        variable in the ``log_likelihood`` group is used (an error is raised if there are
        several) and assumed to match the ``observed_data`` and ``posterior_predictive``
        names.
    kind : str, default "crps"
        The kind of score to compute. Available options are:

        - 'crps': continuous ranked probability score. Default.
        - 'scrps': scale-invariant continuous ranked probability score.
    pointwise : bool, default False
        If True, include per-observation score values in the return object.
    round_to : int or str, optional
        If integer, number of decimal places to round the result. If string of the
        form '2g' number of significant digits to round the result. Defaults to None,
        which returns raw numbers.
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
    if (log_weights is None) != (pareto_k is None):
        raise ValueError("log_weights and pareto_k must be provided together.")

    data = convert_to_datatree(data)

    loo_inputs = _prepare_loo_inputs(data, var_name)
    var_name = loo_inputs.var_name
    log_likelihood = loo_inputs.log_likelihood
    n_samples = loo_inputs.n_samples
    sample_dims = loo_inputs.sample_dims
    obs_dims = loo_inputs.obs_dims

    y_pred = extract(
        data,
        group="posterior_predictive",
        var_names=var_name,
        combined=False,
        sample_dims=sample_dims,
    )
    y_obs = extract(data, group="observed_data", var_names=var_name, combined=False, sample_dims=[])

    _validate_crps_input(y_pred, y_obs, log_likelihood, sample_dims=sample_dims, obs_dims=obs_dims)

    if log_weights is not None:
        pointwise_scores, pareto_k_out = y_pred.azstats.loo_score(
            y_obs=y_obs,
            log_weights=log_weights,
            pareto_k=pareto_k,
            kind=kind,
            sample_dims=sample_dims,
        )
    else:
        r_eff = _get_r_eff(data, n_samples)
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
