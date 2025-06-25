"""Compute exact Leave-One-Out cross validation refitting for problematic observations."""

from copy import deepcopy

import numpy as np
import xarray as xr
from arviz_base import rcParams
from xarray_einstats.stats import logsumexp

from arviz_stats.loo.helper_loo import _prepare_loo_inputs
from arviz_stats.loo.loo import loo
from arviz_stats.loo.wrapper import SamplingWrapper
from arviz_stats.utils import ELPDData

__all__ = ["reloo"]


def reloo(
    wrapper,
    loo_orig=None,
    k_threshold=-np.inf,
    pointwise=None,
):
    r"""Recalculate exact Leave-One-Out cross validation refitting where the approximation fails.

    :func:`arviz_stats.loo` estimates the values of Leave-One-Out (LOO) cross validation using
    Pareto Smoothed Importance Sampling (PSIS) to approximate its value. PSIS works well when
    the posterior and the posterior_i (excluding observation i from the data used to fit)
    are similar. In some cases, there are highly influential observations for which PSIS
    cannot approximate the LOO-CV, and a warning of a large Pareto shape is sent by ArviZ.
    These cases typically have a handful of bad or very bad Pareto shapes, and a majority of
    good or ok shapes.

    Therefore, this may not indicate that the model is not robust enough nor that these
    observations are inherently bad, only that PSIS cannot approximate LOO-CV correctly.
    Thus, we can use PSIS for all observations where the Pareto shape is below a threshold
    and refit the model to perform exact cross validation for the handful of observations
    where PSIS cannot be used. This approach allows us to properly approximate LOO-CV with
    only a handful of refits, which in most cases is still much less computationally expensive
    than exact LOO-CV, which needs one refit per observation.

    Parameters
    ----------
    wrapper : SamplingWrapper
        An instance of a SamplingWrapper subclass that implements the necessary
        methods for model refitting. This wrapper allows ``reloo`` to work with
        any modeling framework.
    loo_orig : ELPDData, optional
        Existing LOO results with pointwise data. If None, will compute
        PSIS-LOO-CV first using the data from ``wrapper``.
    k_threshold : float, optional
        Pareto shape threshold. Observations with k values above this
        threshold will trigger a refit. Defaults to :math:`\min(1 - 1/\log_{10}(S), 0.7)`,
        where S is the number of samples.
    pointwise : bool, optional
        If True, return pointwise LOO data. Defaults to
        ``rcParams["stats.ic_pointwise"]``.

    Returns
    -------
    ELPDData
        Updated LOO results where high Pareto k observations have been
        replaced with exact LOO-CV values from refitting.

    Notes
    -----
    It is strongly recommended to first compute :func:`arviz_stats.loo` on the inference results to
    confirm that the number of values above the threshold is small enough. Otherwise,
    prohibitive computation time may be needed to perform all required refits.

    As an extreme case, artificially assigning all ``pareto_k`` values to something
    larger than the threshold would make ``reloo`` perform the whole exact LOO-CV.
    This is not generally recommended nor intended, however, if needed, this function can
    be used to achieve the result.

    The term `exact cross-validation` is used here to refer to the process of refitting the model
    for each observation, which is also commonly called `brute-force cross-validation`. While
    this eliminates the PSIS approximation, other approximations remain in the process, such as
    those inherent in the fitting procedure itself and the use of cross-validation as a proxy
    for expected log predictive density (ELPD).

    Warnings
    --------
    Refitting can be computationally expensive. Check the number of high Pareto k
    values before using ``reloo`` to ensure the computation time is acceptable.

    See Also
    --------
    loo : Pareto smoothed importance sampling LOO-CV
    loo_moment_match : Moment matching for problematic observations

    References
    ----------

    .. [1] Vehtari et al. *Practical Bayesian model evaluation using leave-one-out cross-validation
        and WAIC*. Statistics and Computing. 27(5) (2017) https://doi.org/10.1007/s11222-016-9696-4
        arXiv preprint https://arxiv.org/abs/1507.04544.

    .. [2] Vehtari et al. *Pareto Smoothed Importance Sampling*.
        Journal of Machine Learning Research, 25(72) (2024) https://jmlr.org/papers/v25/19-556.html
        arXiv preprint https://arxiv.org/abs/1507.02646
    """
    if not isinstance(wrapper, SamplingWrapper):
        raise TypeError(
            "wrapper must be an instance of SamplingWrapper or a subclass. "
            "See the SamplingWrapper documentation for implementation details."
        )

    required_methods = ["sel_observations", "sample", "get_inference_data", "log_likelihood__i"]
    not_implemented = wrapper.check_implemented_methods(required_methods)

    if not_implemented:
        raise TypeError(
            "Passed wrapper instance does not implement all methods required for reloo "
            f"to work. Check the documentation of SamplingWrapper. {not_implemented} must be "
            "implemented and were not found."
        )

    pointwise = rcParams["stats.ic_pointwise"] if pointwise is None else pointwise

    if loo_orig is None:
        loo_orig = loo(wrapper.data, var_name=wrapper.log_lik_var_name, pointwise=True)

    if not isinstance(loo_orig, ELPDData):
        raise TypeError("loo_orig must be an ELPDData object.")

    if loo_orig.pareto_k is None or loo_orig.elpd_i is None:
        raise ValueError(
            "reloo requires pointwise LOO results with Pareto k values. "
            "Please compute the initial LOO with pointwise=True."
        )

    sample_dims = ["chain", "draw"]
    loo_refitted = deepcopy(loo_orig)

    loo_inputs = _prepare_loo_inputs(wrapper.data, wrapper.log_lik_var_name)
    obs_dims = loo_inputs.obs_dims
    n_data_points = loo_inputs.n_data_points
    n_samples = loo_inputs.n_samples
    log_likelihood = loo_inputs.log_likelihood

    if k_threshold is None:
        k_threshold = min(1 - 1 / np.log10(n_samples), 0.7)

    pareto_k_stacked = loo_orig.pareto_k.stack(__obs__=obs_dims).transpose("__obs__")
    bad_k_mask = pareto_k_stacked > k_threshold

    bad_obs_flat_indices = np.where(bad_k_mask.values)[0]

    if len(bad_obs_flat_indices) == 0:
        if not pointwise:
            loo_refitted.elpd_i = None
            loo_refitted.pareto_k = None
        return loo_refitted

    if not hasattr(loo_refitted, "p_loo_i") or loo_refitted.p_loo_i is None:
        loo_refitted.p_loo_i = xr.full_like(loo_refitted.elpd_i, np.nan)
        lpd_i = logsumexp(log_likelihood, b=1 / n_samples, dims=sample_dims)
        loo_refitted.p_loo_i = lpd_i - loo_refitted.elpd_i

    if len(obs_dims) == 1:
        bad_obs_indices = bad_obs_flat_indices
    else:
        obs_shape = [loo_orig.pareto_k.sizes[dim] for dim in obs_dims]
        bad_obs_indices = np.array(np.unravel_index(bad_obs_flat_indices, obs_shape)).T

    for i, _ in enumerate(bad_obs_flat_indices):
        if len(obs_dims) == 1:
            obs_idx = bad_obs_indices[i]
            coord_value = loo_orig.pareto_k.coords[obs_dims[0]].values[obs_idx]
            obs_idx_dict = {obs_dims[0]: coord_value}
            sel_idx = obs_idx
        else:
            obs_idx = bad_obs_indices[i]
            obs_idx_dict = {}
            for j, dim in enumerate(obs_dims):
                coord_value = loo_orig.pareto_k.coords[dim].values[obs_idx[j]]
                obs_idx_dict[dim] = coord_value
            sel_idx = obs_idx_dict

        log_lik_i = log_likelihood.sel(obs_idx_dict)
        hat_lpd_i = logsumexp(log_lik_i, dims=sample_dims, b=1 / n_samples).item()

        new_obs, excluded_obs = wrapper.sel_observations(sel_idx)
        fit = wrapper.sample(new_obs)
        idata_idx = wrapper.get_inference_data(fit)
        log_lik_idx = wrapper.log_likelihood__i(excluded_obs, idata_idx)
        elpd_loo_i = logsumexp(log_lik_idx, dims=sample_dims, b=1 / log_lik_idx.size).item()

        loo_refitted.elpd_i.loc[obs_idx_dict] = elpd_loo_i
        loo_refitted.pareto_k.loc[obs_idx_dict] = 0.0
        loo_refitted.p_loo_i.loc[obs_idx_dict] = hat_lpd_i - elpd_loo_i

    loo_refitted.elpd = np.sum(loo_refitted.elpd_i.values)
    loo_refitted.se = np.sqrt(n_data_points * np.var(loo_refitted.elpd_i.values))
    loo_refitted.p = np.sum(loo_refitted.p_loo_i.values)

    loo_refitted.warning = np.any(loo_refitted.pareto_k.values > loo_refitted.good_k)

    if not pointwise:
        loo_refitted.elpd_i = None
        loo_refitted.pareto_k = None
        loo_refitted.p_loo_i = None

    return loo_refitted
