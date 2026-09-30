"""Continuous ranked probability scores with leave-future-out cross-validation."""

from collections import namedtuple

import numpy as np
import xarray as xr
from arviz_base import convert_to_datatree, rcParams

from arviz_stats.base.stats_utils import round_num
from arviz_stats.loo.lfo_cv_helper import (
    _forecast_draws,
    _forecast_origins,
    _get_observed,
    _label_origins,
    _prepare_lfo_inputs,
    _validate_lfo_method,
    _warn_lfo_refits,
)

__all__ = ["lfo_score"]


def lfo_score(
    data,
    wrapper,
    min_observations,
    forecast_horizon,
    kind="crps",
    time_dim="time",
    pointwise=None,
    var_name=None,
    method="approx",
    k_threshold=0.7,
    round_to=None,
):
    r"""Compute CRPS or SCRPS with leave-future-out cross-validation (LFO-CV).

    Scores M-step-ahead forecasts of time series models with the continuous ranked
    probability score (CRPS) or its scale-invariant variant (SCRPS), where M is set by
    ``forecast_horizon``. The model is trained on the observations up to each forecast
    origin, predictive draws for the next M observations are requested from the wrapper,
    and each observation in the block is scored against its draws. The per-origin score is
    the sum over the block. Both scores are returned as maximization scores where larger is
    better, following :func:`loo_score`.

    As in :func:`lfo_cv`, the posterior is by default carried between forecast origins with
    Pareto-smoothed importance sampling (PSIS) and the model is only refit when the
    approximation becomes unreliable (see ``method``). The importance weights are computed
    from the log likelihood of the observations added since the last refit and applied to
    the predictive draws.

    The PSIS-LFO-CV method is described in [1]_. The CRPS is described in [2]_ and the
    SCRPS in [3]_.

    Parameters
    ----------
    data : DataTree or InferenceData
        Input data containing the posterior, log_likelihood and observed_data groups from
        the full model fit. Must have a time dimension. Will be converted to DataTree.
    wrapper : SamplingWrapper
        An instance of :class:`~arviz_stats.SamplingWrapper` (or subclass) handling
        model refitting. Must implement ``sel_observations``, ``sample``,
        ``get_inference_data``, ``log_likelihood__i`` and ``posterior_predictive__i``.
        Each refit excludes all observations from the forecast origin through the end, and
        ``log_likelihood__i`` is used to form the importance ratios between refits. At every
        forecast origin ``sel_observations`` is called again with the indices of the
        forecast block only, and the excluded observations it returns are passed to
        ``posterior_predictive__i`` together with the current fit. With ``method="approx"``
        the current fit is the most recent refit, which may have been trained on fewer
        observations than precede the block. ``log_likelihood__i`` must condition each
        observation on the observed values before it, because it forms the importance ratios.
    min_observations : int
        Minimum number of observations required before making predictions.
        The first prediction is made at time min_observations.
    forecast_horizon : int
        Number of steps ahead to predict.
    kind : str, default "crps"
        The kind of score to compute. Available options are:

        - 'crps': continuous ranked probability score. Default.
        - 'scrps': scale-invariant continuous ranked probability score.
    time_dim : str, default="time"
        Name of the time dimension in the data.
    pointwise : bool, optional
        If True, include per-origin score values in the return object. Defaults to
        ``rcParams["stats.ic_pointwise"]``.
    var_name : str, optional
        The name of the variable in log_likelihood group storing the pointwise log
        likelihood data to use for computation. The same name is used to retrieve the
        observed values from the observed_data group.
    method : str, default="approx"
        Whether to refit the model at every forecast origin ("exact") or to carry the
        posterior forward with Pareto-smoothed importance sampling (PSIS), refitting
        only when the importance weights become unreliable ("approx").
    k_threshold : float, default=0.7
        Pareto k threshold for triggering refit. If k > k_threshold, refit the model.
    round_to : int or str, optional
        If integer, number of decimal places to round the result. If string of the
        form '2g' number of significant digits to round the result. Defaults to None,
        which returns raw numbers.

    Returns
    -------
    namedtuple
        A namedtuple named ``CRPS`` or ``SCRPS`` with fields ``mean`` and ``se`` computed
        over forecast origins, ``refits`` with the time indices where refits occurred and
        ``n_refits``. For ``method="approx"`` the initial fit at the first forecast origin is
        not counted as a refit. If ``pointwise`` is True, the namedtuple also includes
        ``pointwise`` with the per-origin scores and ``pareto_k`` with the per-origin
        Pareto k diagnostics, which is None for ``method="exact"``. At origins where the
        model was refit, ``pareto_k`` holds the value that triggered the refit.

    Notes
    -----
    Which forecast distribution is scored is decided by the wrapper. When the likelihood
    depends on lagged values of the response, ``posterior_predictive__i`` should simulate
    the earlier observations of the block rather than condition on their observed values,
    so that the draws are forecast trajectories from the origin and predictive uncertainty
    accumulates across the horizon. This differs from the chain rule factorization used by
    the log score in :func:`lfo_cv`, so results from the two functions estimate different
    quantities.

    See Also
    --------
    :func:`lfo_cv` : Leave-future-out cross-validation with the log score.
    :func:`loo_score` : CRPS and SCRPS with PSIS-LOO-CV weights.

    References
    ----------

    .. [1] Bürkner et al. *Approximate leave-future-out cross-validation for Bayesian
       time series models*. Journal of Statistical Computation and Simulation. 90(14) (2020)
       2499-2523. https://doi.org/10.1080/00949655.2020.1783262
       arXiv preprint https://arxiv.org/abs/1902.06281

    .. [2] Gneiting, T., & Raftery, A. E. (2007). *Strictly Proper Scoring Rules,
       Prediction, and Estimation*. Journal of the American Statistical Association,
       102(477), 359–378. https://doi.org/10.1198/016214506000001437

    .. [3] Bolin, D., & Wallin, J. (2023). *Local scale invariance and robustness of
       proper scoring rules*. Statistical Science, 38(1), 140–159. https://doi.org/10.1214/22-STS864
       arXiv preprint https://arxiv.org/abs/1912.05642
    """
    if kind not in {"crps", "scrps"}:
        raise ValueError(f"kind must be either 'crps' or 'scrps'. Got {kind}")

    pointwise = rcParams["stats.ic_pointwise"] if pointwise is None else pointwise

    method = _validate_lfo_method(method)

    data = convert_to_datatree(data)
    lfo_inputs = _prepare_lfo_inputs(
        data, var_name, wrapper, min_observations, forecast_horizon, time_dim
    )
    if wrapper.check_implemented_methods(["posterior_predictive__i"]):
        raise ValueError(
            "The following methods must be implemented in the SamplingWrapper: "
            "['posterior_predictive__i']"
        )
    y_obs = _get_observed(data, lfo_inputs.log_likelihood, time_dim)

    origins = lfo_inputs.origins
    scores = np.empty(len(origins))
    pareto_ks = np.full(len(origins), np.nan)
    refits = []

    for origin in _forecast_origins(lfo_inputs, wrapper, method, k_threshold):
        cutoff = origin.cutoff
        block_idx = np.arange(cutoff, cutoff + forecast_horizon)
        _, block_obs = wrapper.sel_observations(block_idx)
        y_pred = _forecast_draws(wrapper, block_obs, origin, time_dim, forecast_horizon)

        if origin.log_weights is None:
            log_weights = xr.zeros_like(y_pred)
        else:
            log_weights = origin.log_weights.broadcast_like(y_pred)
        block_y = y_obs.isel({time_dim: slice(cutoff, cutoff + forecast_horizon)})

        block_scores, _ = y_pred.azstats.loo_score(
            y_obs=block_y, log_weights=log_weights, kind=kind, sample_dims=origin.sample_dims
        )
        scores[origin.pos] = block_scores.sum().values
        pareto_ks[origin.pos] = origin.pareto_k
        if origin.refit:
            refits.append(cutoff)

    refits = np.array(refits, dtype=int)
    n_refits = len(refits)
    _warn_lfo_refits(method, n_refits, len(origins))

    mean = round_num(scores.mean(), round_to)
    se = round_num(scores.std() / np.sqrt(len(scores)), round_to)
    name = "SCRPS" if kind == "scrps" else "CRPS"

    if not pointwise:
        return namedtuple(name, ["mean", "se", "refits", "n_refits"])(mean, se, refits, n_refits)

    pointwise_scores = _label_origins(lfo_inputs, scores)
    pareto_k = _label_origins(lfo_inputs, pareto_ks) if method == "approx" else None
    return namedtuple(name, ["mean", "se", "pointwise", "pareto_k", "refits", "n_refits"])(
        mean, se, pointwise_scores, pareto_k, refits, n_refits
    )
