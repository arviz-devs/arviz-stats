"""Continuously ranked probability scores with leave-future-out cross-validation."""

from collections import namedtuple

import numpy as np
import xarray as xr
from arviz_base import convert_to_datatree, rcParams

from arviz_stats.base.stats_utils import round_num
from arviz_stats.loo.lfo_cv_helper import _forecast_origins, _prepare_lfo_inputs, _warn_lfo_refits

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
        ``posterior_predictive__i`` together with the current fit.
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
        ``n_refits``. If ``pointwise`` is True, the namedtuple also includes ``pointwise``
        with the per-origin scores and ``pareto_k`` with the per-origin Pareto k
        diagnostics, which is None for ``method="exact"``. At origins where the model was
        refit, ``pareto_k`` holds the value that triggered the refit.

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

    method = method.lower()
    if method not in ("exact", "approx"):
        raise ValueError(
            f"method must be 'exact' or 'approx', got '{method}'. "
            "Use 'exact' for always refitting or 'approx' for PSIS approximation."
        )

    data = convert_to_datatree(data)
    lfo_inputs = _prepare_lfo_inputs(
        data, var_name, wrapper, min_observations, forecast_horizon, time_dim
    )
    if wrapper.check_implemented_methods(["posterior_predictive__i"]):
        raise ValueError(
            "The following methods must be implemented in the SamplingWrapper: "
            "['posterior_predictive__i']"
        )
    y_obs = _get_observed(data, lfo_inputs.log_likelihood.name, time_dim)

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
            log_weights = xr.DataArray(origin.log_weights.values, dims=origin.log_weights.dims)
            log_weights = log_weights.broadcast_like(y_pred).transpose(*y_pred.dims)
        block_y = xr.DataArray(y_obs.values[cutoff : cutoff + forecast_horizon], dims=[time_dim])

        block_scores, _ = y_pred.azstats.loo_score(
            y_obs=block_y,
            log_weights=log_weights,
            pareto_k=origin.pareto_k,
            kind=kind,
            sample_dims=origin.sample_dims,
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

    origin_coord = lfo_inputs.log_likelihood.coords[time_dim].isel({time_dim: origins}).values
    pointwise_scores = xr.DataArray(scores, dims=[time_dim], coords={time_dim: origin_coord})
    pareto_k = None
    if method == "approx":
        pareto_k = xr.DataArray(pareto_ks, dims=[time_dim], coords={time_dim: origin_coord})
    return namedtuple(name, ["mean", "se", "pointwise", "pareto_k", "refits", "n_refits"])(
        mean, se, pointwise_scores, pareto_k, refits, n_refits
    )


def _get_observed(data, var_name, time_dim):
    """Return the observed values of ``var_name`` ordered along the time dimension."""
    if "observed_data" not in data.children:
        raise ValueError("data must contain an observed_data group to compute lfo_score")
    observed = data.observed_data.to_dataset()
    if var_name not in observed:
        raise ValueError(
            f"Variable '{var_name}' not found in observed_data. "
            f"Available variables: {list(observed.data_vars)}"
        )
    y_obs = observed[var_name]
    if y_obs.dims != (time_dim,):
        raise ValueError(
            f"observed_data['{var_name}'] must have dimensions ('{time_dim}',), got {y_obs.dims}"
        )
    return y_obs


def _forecast_draws(wrapper, block_obs, origin, time_dim, horizon):
    """Request predictive draws for the forecast block and check their layout."""
    y_pred = wrapper.posterior_predictive__i(block_obs, origin.idata)
    if y_pred.sizes.get(time_dim) != horizon:
        raise ValueError(
            "posterior_predictive__i must return one value per excluded observation. "
            f"Expected size {horizon} along '{time_dim}', got {y_pred.sizes.get(time_dim)} "
            f"for forecast origin {origin.cutoff}."
        )
    pred_dims = [dim for dim in y_pred.dims if dim != time_dim]
    if sorted(pred_dims) != sorted(origin.sample_dims) or any(
        y_pred.sizes[dim] != origin.log_lik.sizes[dim] for dim in pred_dims
    ):
        raise ValueError(
            "posterior_predictive__i must return draws with the same sample dimensions as "
            f"log_likelihood__i. Expected {dict(origin.log_lik.sizes)} without '{time_dim}', "
            f"got {dict(y_pred.sizes)}."
        )
    y_pred = y_pred.transpose(*origin.sample_dims, time_dim)
    return xr.DataArray(y_pred.values, dims=y_pred.dims)
