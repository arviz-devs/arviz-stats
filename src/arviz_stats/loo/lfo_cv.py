"""Leave-Future-Out Cross-Validation (LFO-CV) for time series models."""

import warnings

import numpy as np
from arviz_base import rcParams

from arviz_stats.loo.lfo_cv_helper import (
    _compute_lfo_approx,
    _compute_lfo_exact,
    _prepare_lfo_inputs,
)
from arviz_stats.utils import ELPDData

__all__ = ["lfo_cv"]


def lfo_cv(
    data,
    wrapper,
    min_observations,
    forecast_horizon,
    time_dim="time",
    pointwise=None,
    var_name=None,
    method="approx",
    k_threshold=0.7,
):
    """Compute leave-future-out cross-validation (LFO-CV) for time series models.

    Estimates the expected log predictive density (elpd) of M-step-ahead predictions for
    time series models, where M is set by ``forecast_horizon``. The model is trained on the
    observations up to each forecast origin and scored on the joint log predictive density
    of the next M observations, so predictions are never conditioned on data from the future.
    By default, the posterior is carried between forecast origins with Pareto-smoothed
    importance sampling (PSIS) and the model is only refit when the approximation becomes
    unreliable (see ``method``).

    The PSIS-LFO-CV method is described in [1]_ and builds on the PSIS-LOO-CV method described
    in [2]_ and [3]_.

    Parameters
    ----------
    data : DataTree or InferenceData
        Input data containing the posterior and log_likelihood groups from the
        full model fit. Must have a time dimension. Will be converted to DataTree.
    wrapper : SamplingWrapper
        An instance of :class:`~arviz_stats.SamplingWrapper` (or subclass) handling
        model refitting. Must implement ``sel_observations``, ``sample``,
        ``get_inference_data``, and ``log_likelihood__i``. Following the same pattern as
        :func:`loo_kfold`, ``sel_observations`` receives an array of integer indices to
        exclude from training and returns ``(modified_observed_data, excluded_observed_data)``.
        The forecast window is selected from ``excluded_observed_data`` internally. Each refit
        excludes all observations from the forecast origin through the end. The approximate
        method also uses later excluded values to compute subsequent importance ratios.
    min_observations : int
        Minimum number of observations required before making predictions.
        The first prediction is made at time min_observations.
    forecast_horizon : int
        Forecast horizon: number of steps ahead to predict.
    time_dim : str, default="time"
        Name of the time dimension in the data.
    pointwise : bool, optional
        If True, return pointwise estimates. Defaults to
        ``rcParams["stats.ic_pointwise"]``.
    var_name : str, optional
        The name of the variable in log_likelihood group storing the pointwise log
        likelihood data to use for computation.
    method : str, default="approx"
        Whether to refit the model at every forecast origin ("exact") or to carry the
        posterior forward with Pareto-smoothed importance sampling (PSIS) [3]_, refitting
        only when the importance weights become unreliable ("approx").
    k_threshold : float, default=0.7
        Pareto k threshold for triggering refit. If k > k_threshold, refit the model.
        Recommended values: 0.6 (conservative) or 0.7 (default).

    Returns
    -------
    ELPDData
        Object with the following attributes:

        - **kind**: "lfo_cv"
        - **elpd**: expected log pointwise predictive density
        - **se**: standard error of the elpd
        - **p**: effective number of parameters
        - **n_samples**: number of posterior samples
        - **n_data_points**: number of forecast origins evaluated
        - **scale**: "log"
        - **warning**: bool - True if more than half of the forecast origins required a
          refit (only if ``method="approx"``, otherwise False)
        - **good_k**: float - The k-threshold value used for refit decisions
          (only if ``method="approx"``, otherwise None)
        - **elpd_i**: :class:`~xarray.DataArray` with pointwise predictive accuracy,
          only if ``pointwise=True``
        - **pareto_k**: :class:`~xarray.DataArray` with Pareto k diagnostics,
          only if ``method="approx"`` and ``pointwise=True``
        - **p_lfo_i**: :class:`~xarray.DataArray` with pointwise effective number of
          parameters, only if ``pointwise=True``
        - **forecast_horizon**: forecast horizon
        - **min_observations**: minimum observations
        - **refits**: array of time indices where refits occurred. For ``method="approx"``
          the initial fit at the first forecast origin is not counted as a refit
        - **n_refits**: number of refits performed

    Notes
    -----
    When ``forecast_horizon`` is greater than 1, each forecast origin is scored with the
    joint log predictive density of the next ``forecast_horizon`` observations. If the
    likelihood depends on lagged values of the response, as in autoregressive models,
    there is more than one way to evaluate this joint density.

    ``lfo_cv`` uses the chain rule factorization, where each observation in the block is
    scored conditional on the observed values of the observations before it. This measures
    how well the model predicts the block that actually occurred given the training data.

    The alternative is to score each observation in the block against simulated values of
    its predecessors drawn from the model, letting predictive uncertainty accumulate across
    the horizon. This measures how well the model forecasts the block when the intermediate
    observations are never revealed, and is common in software oriented toward forecast
    evaluation. The two approaches coincide when ``forecast_horizon=1`` and whenever the
    likelihood does not depend on lagged values of the response. Otherwise they estimate
    different quantities, so results computed one way should not be compared with results
    computed the other way.

    See Also
    --------
    :func:`loo` : Pareto-smoothed importance sampling LOO-CV.
    :func:`compare` : Compare models based on their ELPD.

    References
    ----------

    .. [1] Bürkner et al. *Approximate leave-future-out cross-validation for Bayesian
       time series models*. Journal of Statistical Computation and Simulation. 90(14) (2020)
       2499-2523. https://doi.org/10.1080/00949655.2020.1783262
       arXiv preprint https://arxiv.org/abs/1902.06281

    .. [2] Vehtari et al. *Practical Bayesian model evaluation using leave-one-out cross-validation
       and WAIC*. Statistics and Computing. 27(5) (2017) https://doi.org/10.1007/s11222-016-9696-4
       arXiv preprint https://arxiv.org/abs/1507.04544.

    .. [3] Vehtari et al. *Pareto Smoothed Importance Sampling*.
       Journal of Machine Learning Research, 25(72) (2024) https://jmlr.org/papers/v25/19-556.html
       arXiv preprint https://arxiv.org/abs/1507.02646
    """
    pointwise = rcParams["stats.ic_pointwise"] if pointwise is None else pointwise

    method = method.lower()
    if method not in ("exact", "approx"):
        raise ValueError(
            f"method must be 'exact' or 'approx', got '{method}'. "
            "Use 'exact' for always refitting or 'approx' for PSIS approximation."
        )

    lfo_inputs = _prepare_lfo_inputs(
        data, var_name, wrapper, min_observations, forecast_horizon, time_dim
    )

    if method == "exact":
        lfo_results = _compute_lfo_exact(lfo_inputs, wrapper)
    else:
        lfo_results = _compute_lfo_approx(lfo_inputs, wrapper, k_threshold)

    warning = False
    good_k = k_threshold if method == "approx" else None
    n_refits = len(lfo_results.refits)
    if method == "approx":
        if n_refits > lfo_results.n_data_points / 2:
            warnings.warn(
                f"LFO-CV triggered {n_refits} refits out of {lfo_results.n_data_points} forecast "
                "origins. The importance sampling approximation may be unreliable. "
                "Consider method='exact'.",
                UserWarning,
            )
            warning = True

    elpd_data = ELPDData(
        kind="lfo_cv",
        elpd=lfo_results.elpd,
        se=lfo_results.se,
        p=lfo_results.p,
        n_samples=lfo_inputs.n_samples,
        n_data_points=lfo_results.n_data_points,
        scale="log",
        warning=warning,
        good_k=good_k,
        elpd_i=lfo_results.elpd_i if pointwise else None,
        pareto_k=lfo_results.pareto_k if (pointwise and method == "approx") else None,
        forecast_horizon=forecast_horizon,
        min_observations=min_observations,
        refits=np.asarray(lfo_results.refits),
        n_refits=n_refits,
        p_lfo_i=lfo_results.p_lfo_i if pointwise else None,
    )

    return elpd_data
