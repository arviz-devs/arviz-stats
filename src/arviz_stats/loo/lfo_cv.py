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
    """Compute leave-future-out cross-validation (LFO-CV).

    Estimates the expected log pointwise predictive density (elpd) for time series models by
    forecasting future observations from progressively expanding training windows, scoring the
    joint predictive density of the next few observations at each step as described in [1]_.
    The exact method refits the model at every step, whereas the approximate method carries the
    posterior forward with Pareto-smoothed importance sampling (PSIS) [3]_ and refits only
    when the importance weights become unreliable.

    Parameters
    ----------
    data : DataTree or InferenceData
        Input data containing the posterior and log_likelihood groups from the
        full model fit. Must have a time dimension. Will be converted to DataTree.
    wrapper : SamplingWrapper
        An instance of :class:`~arviz_stats.loo.SamplingWrapper` (or subclass) handling
        model refitting. Must implement ``sel_observations``, ``sample``,
        ``get_inference_data``, and ``log_likelihood__i``. Following the same pattern as
        :func:`loo_kfold`, ``sel_observations`` receives an array of integer indices to
        exclude from training and returns ``(modified_observed_data, excluded_observed_data)``.
        The forecast window is selected from ``excluded_observed_data`` internally.
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
        Method to use: "exact" (always refit) or "approx" (use PSIS when possible).
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
        - **warning**: bool - True if many refits were needed or high k values remain
          (only if ``method="approx"``, otherwise False)
        - **good_k**: float - The k-threshold value used for refit decisions
          (only if ``method="approx"``, otherwise None)
        - **elpd_i**: :class:`~xarray.DataArray` with pointwise predictive accuracy,
          only if ``pointwise=True``
        - **pareto_k**: :class:`~xarray.DataArray` with Pareto k diagnostics,
          only if ``method="approx"`` and ``pointwise=True``
        - **forecast_horizon**: forecast horizon
        - **min_observations**: minimum observations
        - **refits**: array of time indices where refits occurred
        - **n_refits**: number of refits performed

    Examples
    --------
    LFO-CV refits the model as the training window expands, so we describe how to refit
    with a :class:`~arviz_stats.loo.SamplingWrapper`. First we simulate a short series and
    assemble the inference data from the full-data fit:

    .. ipython::

        In [1]: import numpy as np
           ...: import xarray as xr
           ...: from scipy import stats
           ...: from arviz_base import from_dict
           ...: from arviz_stats import lfo_cv
           ...: from arviz_stats.loo import SamplingWrapper
           ...:
           ...: rng = np.random.default_rng(0)
           ...: n_time = 25
           ...: y = rng.normal(size=n_time)
           ...: idata = from_dict(
           ...:     {
           ...:         "posterior": {"mu": rng.normal(size=(2, 400))},
           ...:         "log_likelihood": {"obs": rng.normal(-1.4, 0.1, size=(2, 400, n_time))},
           ...:         "observed_data": {"obs": y},
           ...:     },
           ...:     dims={"obs": ["time"]},
           ...:     coords={"time": np.arange(n_time)},
           ...: )

    The wrapper implements the four methods ``lfo_cv`` calls to refit on a training window
    and score the forecast observations. Here we use a simple normal-mean model:

    .. ipython::

        In [2]: class TimeSeriesWrapper(SamplingWrapper):
           ...:     def __init__(self, idata):
           ...:         super().__init__(model=None, idata_orig=idata)
           ...:         self.y = idata.observed_data["obs"].values
           ...:
           ...:     def sel_observations(self, idx):
           ...:         train_idx = np.setdiff1d(np.arange(len(self.y)), idx)
           ...:         train = xr.DataArray(
           ...:             self.y[train_idx], dims=["time"], coords={"time": train_idx}
           ...:         )
           ...:         test = xr.DataArray(self.y[idx], dims=["time"], coords={"time": idx})
           ...:         return train, test
           ...:
           ...:     def sample(self, modified_observed_data):
           ...:         n = len(modified_observed_data)
           ...:         mean = modified_observed_data.values.mean()
           ...:         local = np.random.default_rng(0)
           ...:         return {"mu": local.normal(mean, 1.0 / np.sqrt(n), 400)}
           ...:
           ...:     def get_inference_data(self, fitted_model):
           ...:         post = {"mu": fitted_model["mu"].reshape(1, -1)}
           ...:         return from_dict({"posterior": post})
           ...:
           ...:     def log_likelihood__i(self, excluded_obs, idata__i):
           ...:         mu = idata__i.posterior["mu"].values.flatten()
           ...:         obs = np.atleast_1d(excluded_obs.values)
           ...:         log_lik = stats.norm.logpdf(obs, loc=mu[:, None], scale=1.0)
           ...:         return xr.DataArray(
           ...:             log_lik.T[np.newaxis, :, :],
           ...:             dims=["chain", "time", "draw"],
           ...:             coords={"time": np.atleast_1d(excluded_obs.coords["time"].values)},
           ...:         )

    We then run one-step-ahead LFO-CV. The approximate method refits only when the PSIS
    diagnostic degrades, so it uses far fewer fits than one per step:

    .. ipython::

        In [3]: wrapper = TimeSeriesWrapper(idata)
           ...: lfo_cv(idata, wrapper, min_observations=10, forecast_horizon=1)

    For multi-step-ahead prediction (M-SAP), increase ``forecast_horizon``. Each origin then
    scores the joint predictive density of the next ``forecast_horizon`` observations:

    .. ipython::

        In [4]: lfo_cv(idata, wrapper, min_observations=10, forecast_horizon=4)

    See Also
    --------
    :func:`loo` : Pareto-smoothed importance sampling LOO-CV.
    :func:`loo_kfold` : Exact :math:`K`-fold cross-validation.
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
    good_k = None
    n_refits = len(lfo_results.refits)
    if method == "approx":
        good_k = k_threshold
        if n_refits > lfo_results.n_data_points / 2:
            warnings.warn(
                f"LFO-CV triggered {n_refits} refits out of {lfo_results.n_data_points} forecast "
                "origins. The importance sampling approximation may be unreliable; "
                "consider method='exact'.",
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
    )

    elpd_data.forecast_horizon = forecast_horizon
    elpd_data.min_observations = min_observations
    elpd_data.refits = np.asarray(lfo_results.refits)
    elpd_data.n_refits = n_refits

    if pointwise:
        elpd_data.p_lfo_i = lfo_results.p_lfo_i

    return elpd_data
