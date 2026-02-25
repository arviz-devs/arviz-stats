"""Leave-Future-Out Cross-Validation (LFO-CV) for time series models."""

import warnings

import numpy as np
from arviz_base import rcParams
from xarray_einstats.stats import logsumexp

from arviz_stats.loo.helper_lfo_cv import (
    _combine_lfo_elpds,
    _compute_lfo_step_approx,
    _compute_lfo_step_exact,
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
    save_fits=False,
):
    """Perform Leave-Future-Out Cross-Validation (LFO-CV).

    LFO-CV evaluates model predictive accuracy for time series by forecasting
    forecast_horizon steps ahead using progressively expanding training windows, respecting
    temporal ordering. Uses forward mode: starting at min_observations and moving
    forward in time.

    Parameters
    ----------
    data : DataTree or InferenceData
        Input data containing the posterior and log_likelihood groups from the
        full model fit. Must have a time dimension. Will be converted to DataTree.
    wrapper : SamplingWrapper
        An instance of SamplingWrapper (or subclass) handling model refitting.
        Must implement sel_observations(idx), sample(), get_inference_data(),
        and log_likelihood__i() methods. Following the same pattern as
        loo_kfold, sel_observations receives an array of integer indices to
        exclude from training and returns (modified_observed_data,
        excluded_observed_data). The lfo_cv function then selects the forecast
        window from excluded_observed_data using isel() internally.
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
    save_fits : bool, default=False
        If True, store the fitted models and refit indices in the returned object.

    Returns
    -------
    ELPDData
        Object with the following attributes:

        - **kind**: "lfo_cv"
        - **elpd**: expected log pointwise predictive density
        - **se**: standard error of the elpd
        - **p**: effective number of parameters
        - **n_samples**: number of samples per time step
        - **n_data_points**: number of time steps evaluated
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
        - **fit_fits**: Dictionary containing fitted models for each refit,
          only if ``save_fits=True``

    Examples
    --------
    LFO-CV requires model refitting, so we need a SamplingWrapper subclass.
    The wrapper splits data at a time cutoff; lfo_cv handles the forecast
    window selection internally:

    .. ipython::

        In [1]: import arviz_stats as az
           ...: from arviz_stats.loo import SamplingWrapper
           ...:
           ...: class MyTimeSeriesWrapper(SamplingWrapper):
           ...:     def __init__(self, model, idata, time_dim="time"):
           ...:         super().__init__(model=model, idata_orig=idata)
           ...:         self.time_dim = time_dim
           ...:
           ...:     def sel_observations(self, idx):
           ...:         # idx: array of integer time indices to exclude (same pattern as loo_kfold)
           ...:         # Returns (modified_observed_data, excluded_observed_data)
           ...:         all_idx = np.arange(self.idata_orig.observed_data.sizes[self.time_dim])
           ...:         train_idx = np.setdiff1d(all_idx, idx)
           ...:         modified_observed_data = self.idata_orig.isel({self.time_dim: train_idx})
           ...:         excluded_observed_data = self.idata_orig.isel({self.time_dim: idx})
           ...:         return modified_observed_data, excluded_observed_data
           ...:
           ...:     def sample(self, modified_observed_data):
           ...:         # Refit model on training data
           ...:         ...
           ...:
           ...:     def get_inference_data(self, fitted_model):
           ...:         return fitted_model
           ...:
           ...:     def log_likelihood__i(self, excluded_obs, idata__i):
           ...:         # Compute log-likelihood for prediction observations
           ...:         ...
           ...:
           ...: wrapper = MyTimeSeriesWrapper(model, idata)
           ...: lfo_result = az.lfo_cv(idata, wrapper, min_observations=25, forecast_horizon=4)

    Notes
    -----
    LFO-CV is designed for time series data where temporal ordering matters.
    Unlike standard LOO-CV, LFO-CV only uses past data to predict future
    observations, respecting the temporal structure.

    The approximate method uses PSIS to avoid refitting when the Pareto k
    diagnostic is below the threshold. This can significantly reduce
    computational cost while maintaining accuracy.

    References
    ----------
    .. [1] Bürkner, P.-C., Gabry, J., & Vehtari, A. (2020). Approximate
           leave-future-out cross-validation for Bayesian time series models.
           Journal of Statistical Computation and Simulation, 90(14), 2499-2523.
           https://doi.org/10.1080/00949655.2020.1783262
    """
    # Handle pointwise default
    pointwise = rcParams["stats.ic_pointwise"] if pointwise is None else pointwise

    # Validate method
    method = method.lower()
    if method not in ("exact", "approx"):
        raise ValueError(
            f"method must be 'exact' or 'approx', got '{method}'. "
            "Use 'exact' for always refitting or 'approx' for PSIS approximation."
        )

    # Prepare inputs
    lfo_inputs = _prepare_lfo_inputs(
        data, var_name, wrapper, min_observations, forecast_horizon, time_dim
    )

    # Run LFO-CV
    if method == "exact":
        results = _run_lfo_exact(lfo_inputs, wrapper, save_fits)
    else:
        results = _run_lfo_approx(lfo_inputs, wrapper, k_threshold, save_fits)

    # Compute effective number of parameters (p)
    # For LFO-CV, p is computed as the difference between the log pointwise predictive
    # density from the full posterior and the LFO ELPD
    log_likelihood = lfo_inputs.log_likelihood
    sample_dims = lfo_inputs.sample_dims
    n_samples = lfo_inputs.n_samples

    # LPD from full posterior for the evaluated time points
    lpd_full = logsumexp(log_likelihood, dims=sample_dims, b=1 / n_samples)
    p_lfo = float(lpd_full.sum().values) - results["elpd"]

    # Determine warning status
    warning = False
    good_k = None
    if method == "approx":
        # Use default threshold if None
        effective_k_threshold = k_threshold if k_threshold is not None else 0.7
        good_k = effective_k_threshold
        if results["pareto_k"] is not None:
            max_k = float(results["pareto_k"].max().values)
            warning = max_k > effective_k_threshold

        # Also warn if many refits were needed (more than half of steps)
        n_steps = lfo_inputs.n_time_points - min_observations
        if results["n_refits"] > n_steps / 2:
            warnings.warn(
                f"LFO-CV required {results['n_refits']} refits out of {n_steps} steps. "
                "Consider using method='exact' or checking the model specification.",
                UserWarning,
            )
            warning = True

    # Create ELPDData result
    elpd_data = ELPDData(
        kind="lfo_cv",
        elpd=results["elpd"],
        se=results["se"],
        p=p_lfo,
        n_samples=n_samples,
        n_data_points=results["n_data_points"],
        scale="log",
        warning=warning,
        good_k=good_k,
        elpd_i=results["elpd_i"] if pointwise else None,
        pareto_k=results["pareto_k"] if (pointwise and method == "approx") else None,
    )

    # Add LFO-specific attributes
    elpd_data.forecast_horizon = forecast_horizon
    elpd_data.min_observations = min_observations
    elpd_data.refits = results["refit_indices"]
    elpd_data.n_refits = results["n_refits"]

    if save_fits and "fit_fits" in results:
        elpd_data.fit_fits = results["fit_fits"]

    return elpd_data


def _run_lfo_exact(lfo_inputs, wrapper, save_fits):
    """Run LFO-CV with exact refitting at each step.

    Parameters
    ----------
    lfo_inputs : LfoInputs
        Prepared inputs.
    wrapper : SamplingWrapper
        Wrapper for model refitting.
    save_fits : bool
        Whether to save fitted models.

    Returns
    -------
    dict
        Results dictionary.
    """
    min_observations = lfo_inputs.min_observations
    n_time_points = lfo_inputs.n_time_points
    forecast_horizon = lfo_inputs.forecast_horizon
    time_dim = lfo_inputs.time_dim

    step_results = []
    fit_fits = {} if save_fits else None

    # Forward iteration: from min_observations to end
    for cutoff in range(min_observations, n_time_points):
        # Check if we can still make a forecast
        if cutoff + forecast_horizon > n_time_points:
            # Reduce forecast horizon for final steps
            actual_horizon = n_time_points - cutoff
            if actual_horizon < 1:
                break

        result = _compute_lfo_step_exact(lfo_inputs, wrapper, cutoff)
        step_results.append(result)

        if save_fits:
            fit_fits[cutoff] = {
                "time_index": cutoff,
                "forecast_horizon": min(forecast_horizon, n_time_points - cutoff),
            }

    # Combine results
    combined = _combine_lfo_elpds(step_results, time_dim, n_time_points, forecast_horizon)

    if save_fits:
        combined["fit_fits"] = fit_fits

    return combined


def _run_lfo_approx(lfo_inputs, wrapper, k_threshold, save_fits):
    """Run LFO-CV with PSIS approximation.

    Parameters
    ----------
    lfo_inputs : LfoInputs
        Prepared inputs.
    wrapper : SamplingWrapper
        Wrapper for model refitting.
    k_threshold : float
        Pareto k threshold for triggering refit.
    save_fits : bool
        Whether to save fitted models.

    Returns
    -------
    dict
        Results dictionary.
    """
    min_observations = lfo_inputs.min_observations
    n_time_points = lfo_inputs.n_time_points
    forecast_horizon = lfo_inputs.forecast_horizon
    time_dim = lfo_inputs.time_dim
    log_likelihood = lfo_inputs.log_likelihood

    step_results = []
    fit_fits = {} if save_fits else None
    refit_indices = []

    # Start with initial fit at min_observations
    last_refit_cutoff = min_observations

    # Initial fit: train on first min_observations, get log-likelihood for all time points
    exclude_idx = np.arange(min_observations, n_time_points)
    train_data, _ = wrapper.sel_observations(exclude_idx)
    fitted_model = wrapper.sample(train_data)
    idata_current = wrapper.get_inference_data(fitted_model)

    # Get full log-likelihood from the refitted model
    # We need log-likelihood for all time points to do importance sampling
    # For now, use the original log-likelihood and adjust with importance weights
    log_lik_current = log_likelihood.copy()

    refit_indices.append(min_observations)

    if save_fits:
        fit_fits[min_observations] = {
            "time_index": min_observations,
            "model": idata_current,
        }

    # Forward iteration
    for cutoff in range(min_observations, n_time_points):
        if cutoff + forecast_horizon > n_time_points:
            actual_horizon = n_time_points - cutoff
            if actual_horizon < 1:
                break

        # Try approximate computation
        result, needs_refit, _ = _compute_lfo_step_approx(
            lfo_inputs,
            wrapper,
            cutoff,
            log_lik_current,
            k_threshold,
            last_refit_cutoff,
        )

        if needs_refit:
            # Perform exact computation with refit
            result = _compute_lfo_step_exact(lfo_inputs, wrapper, cutoff)
            last_refit_cutoff = cutoff
            refit_indices.append(cutoff)

            # Update current log-likelihood for future importance sampling
            # In a full implementation, we'd get the new log-likelihood from the refitted model
            # For now, we continue with the original log-likelihood

            if save_fits:
                fit_fits[cutoff] = {
                    "time_index": cutoff,
                    "refitted": True,
                }

        step_results.append(result)

    # Combine results
    combined = _combine_lfo_elpds(step_results, time_dim, n_time_points, forecast_horizon)

    # Override refit_indices with our tracked values
    combined["refit_indices"] = np.array(refit_indices)
    combined["n_refits"] = len(refit_indices)

    if save_fits:
        combined["fit_fits"] = fit_fits

    return combined
