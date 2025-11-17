"""Compute moment matching for problematic observations in PSIS-LOO-CV."""

import warnings
from collections import namedtuple
from copy import deepcopy

import arviz_base as azb
import numpy as np
import xarray as xr
from arviz_base import dataset_to_dataarray, rcParams
from xarray_einstats.stats import logsumexp

from arviz_stats.loo.helper_loo import (
    _get_log_likelihood_i,
    _get_r_eff,
    _get_r_eff_i,
    _get_weights_and_k_i,
    _prepare_loo_inputs,
    _shift,
    _shift_and_cov,
    _shift_and_scale,
    _warn_pareto_k,
)
from arviz_stats.sampling_diagnostics import ess
from arviz_stats.utils import ELPDData

SplitMomentMatch = namedtuple("SplitMomentMatch", ["lwi", "lwfi", "log_liki", "reff"])
UpdateQuantities = namedtuple("UpdateQuantities", ["lwi", "lwfi", "ki", "kfi", "log_liki"])
LooMomentMatchResult = namedtuple(
    "LooMomentMatchResult",
    ["final_log_liki", "final_lwi", "final_ki", "kfs_i", "reff_i", "original_ki", "i"],
)


def loo_moment_match(
    data,
    loo_orig,
    log_prob_upars_fn,
    log_lik_i_upars_fn,
    upars=None,
    var_name=None,
    reff=None,
    max_iters=30,
    k_threshold=None,
    split=True,
    cov=False,
    pointwise=None,
):
    r"""Compute moment matching for problematic observations in PSIS-LOO-CV.

    Adjusts the results of a previously computed Pareto smoothed importance sampling leave-one-out
    cross-validation (PSIS-LOO-CV) object by applying a moment matching algorithm to
    observations with high Pareto k diagnostic values. The moment matching algorithm iteratively
    adjusts the posterior draws in the unconstrained parameter space to better approximate the
    leave-one-out posterior.

    The moment matching algorithm is described in [1]_ and the PSIS-LOO-CV method is described in
    [2]_ and [3]_.

    Parameters
    ----------
    data : DataTree or InferenceData
        Input data. It should contain the posterior and the log_likelihood groups.
    loo_orig : ELPDData
        An existing ELPDData object from a previous `loo` result. Must contain
        pointwise Pareto k values (`pointwise=True` must have been used).
    log_prob_upars_fn : callable
        Function that computes the log probability density of the full posterior
        distribution evaluated at unconstrained parameter draws.
        The function signature is ``log_prob_upars_fn(upars)`` where ``upars``
        is a :class:`~xarray.DataArray` of unconstrained parameter draws with dimensions
        ``chain``, ``draw``, and a parameter dimension. It should return a
        :class:`~xarray.DataArray` with dimensions ``chain``, ``draw``.
    log_lik_i_upars_fn : callable
        Function that computes the log-likelihood of a single left-out observation
        evaluated at unconstrained parameter draws.
        The function signature is ``log_lik_i_upars_fn(upars, i)`` where ``upars``
        is a :class:`~xarray.DataArray` of unconstrained parameter draws and ``i``
        is the integer index of the left-out observation. It should return a
        :class:`~xarray.DataArray` with dimensions ``chain``, ``draw``.
    upars : DataArray, optional
        Posterior draws transformed to the unconstrained parameter space. Must have
        ``chain`` and ``draw`` dimensions, plus one additional dimension containing all
        parameters. Parameter names can be provided as coordinate values on this
        dimension. If not provided, will attempt to use the ``unconstrained_posterior``
        group from the input data if available.
    var_name : str, optional
        The name of the variable in log_likelihood group storing the pointwise log
        likelihood data to use for loo computation.
    reff: float, optional
        Relative MCMC efficiency, ``ess / n`` i.e. number of effective samples divided by the number
        of actual samples. Computed from trace by default.
    max_iters : int, default 30
        Maximum number of moment matching iterations for each problematic observation.
    k_threshold : float, optional
        Threshold value for Pareto k values above which moment matching is applied.
        Defaults to :math:`\min(1 - 1/\log_{10}(S), 0.7)`, where S is the number of samples.
    split : bool, default True
        If True, only transform half of the draws and use multiple importance sampling to combine
        them with untransformed draws.
    cov : bool, default False
        If True, match the covariance structure during the transformation, in addition
        to the mean and marginal variances. Ignored if ``split=False``.
    pointwise: bool, optional
        If True, the pointwise predictive accuracy will be returned. Defaults to
        ``rcParams["stats.ic_pointwise"]``. Moment matching always requires
        pointwise data from ``loo_orig``. This argument controls whether the returned
        object includes pointwise data.

    Returns
    -------
    ELPDData
        Object with the following attributes:

        - **kind**: "loo"
        - **elpd**: expected log pointwise predictive density
        - **se**: standard error of the elpd
        - **p**: effective number of parameters
        - **n_samples**: number of samples
        - **n_data_points**: number of data points
        - **scale**: "log"
        - **warning**: True if the estimated shape parameter of Pareto distribution is greater
          than ``good_k``.
        - **good_k**: For a sample size S, the threshold is computed as
          ``min(1 - 1/log10(S), 0.7)``
        - **elpd_i**: :class:`~xarray.DataArray` with the pointwise predictive accuracy, only if
          ``pointwise=True``.
        - **pareto_k**: :class:`~xarray.DataArray` with moment-matched Pareto shape values, only if
          ``pointwise=True``.
        - **approx_posterior**: False (not used for standard LOO)
        - **log_weights**: Smoothed log weights.
        - **influence_pareto_k**: :class:`~xarray.DataArray` with original (pre-moment-matching)
          Pareto shape values, only if ``pointwise=True``.

    Examples
    --------
    Moment matching can improve PSIS-LOO-CV estimates for observations with high Pareto k values
    without having to refit the model for each problematic observation. We will use the non-centered
    eight schools data which has 1 problematic observation. In practice, moment matching is useful
    when you have a potentially large number of problematic observations:

    .. ipython::
        :okwarning:

        In [1]: import arviz_base as azb
           ...: import numpy as np
           ...: import xarray as xr
           ...: from scipy import stats
           ...: from arviz_stats import loo
           ...:
           ...: idata = azb.load_arviz_data("non_centered_eight")
           ...: posterior = idata.posterior
           ...: schools = posterior.theta_t.coords["school"].values
           ...: y_obs = idata.observed_data.obs
           ...: obs_dim = y_obs.dims[0]
           ...:
           ...: loo_orig = loo(idata, pointwise=True, var_name="obs")
           ...: loo_orig

    The moment matching algorithm applies affine transformations to posterior draws in
    unconstrained parameter space. To enable this, we need to collect the posterior
    parameters from their original space, transform them to unconstrained space if
    needed, and stack them into a single :class:`xarray.DataArray` that matches the
    expected ``(chain, draw, param)`` structure. Some parameters may already be in
    unconstrained space, so we don't need to transform them. This will depend on the
    model and the choice of parameterization:

    .. ipython::
        :okwarning:

        In [2]: upars_ds = xr.Dataset(
           ...:     {
           ...:         **{
           ...:             f"theta_t_{school}": posterior.theta_t.sel(school=school, drop=True)
           ...:             for school in schools
           ...:         },
           ...:         "mu": posterior.mu,
           ...:         "log_tau": xr.apply_ufunc(np.log, posterior.tau),
           ...:     }
           ...: )
           ...: upars = azb.dataset_to_dataarray(
           ...:     upars_ds, sample_dims=["chain", "draw"], new_dim="upars_dim"
           ...: )

    Moment matching requires two functions: one for the joint log probability (likelihood + priors)
    and another for the pointwise log-likelihood of a single observation. We first define functions
    that accept the data they need as keyword-only arguments:

    .. ipython::
        :okwarning:

        In [3]: sigmas = xr.DataArray(
           ...:     [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0],
           ...:     dims=[obs_dim],
           ...: )
           ...:
           ...: def log_prob_upars(upars, *, sigmas, y, schools, obs_dim):
           ...:     theta_t = xr.concat(
           ...:         [upars.sel(upars_dim=f"theta_t_{school}") for school in schools],
           ...:         dim=obs_dim,
           ...:     )
           ...:     mu = upars.sel(upars_dim="mu")
           ...:     log_tau = upars.sel(upars_dim="log_tau")
           ...:     tau = xr.apply_ufunc(np.exp, log_tau)
           ...:     theta = mu + tau * theta_t
           ...:
           ...:     log_prior = xr.apply_ufunc(stats.norm(0, 5).logpdf, mu)
           ...:     log_prior = log_prior + xr.apply_ufunc(
           ...:         stats.halfcauchy(0, 5).logpdf,
           ...:         tau,
           ...:     )
           ...:     log_prior = log_prior + log_tau
           ...:     log_prior = log_prior + xr.apply_ufunc(
           ...:         stats.norm(0, 1).logpdf,
           ...:         theta_t,
           ...:     ).sum(obs_dim)
           ...:
           ...:     const = -0.5 * np.log(2 * np.pi)
           ...:     log_like = const - np.log(sigmas) - 0.5 * ((y - theta) / sigmas) ** 2
           ...:     log_like = log_like.sum(obs_dim)
           ...:     return log_prior + log_like
           ...:
           ...: def log_lik_i_upars(upars, i, *, sigmas, y, schools, obs_dim):
           ...:     mu = upars.sel(upars_dim="mu")
           ...:     log_tau = upars.sel(upars_dim="log_tau")
           ...:     tau = xr.apply_ufunc(np.exp, log_tau)
           ...:
           ...:     theta_t_i = upars.sel(upars_dim=f"theta_t_{schools[i]}")
           ...:     theta_i = mu + tau * theta_t_i
           ...:
           ...:     sigma_i = sigmas.isel({obs_dim: i})
           ...:     y_i = y.isel({obs_dim: i})
           ...:     const = -0.5 * np.log(2 * np.pi)
           ...:     return const - np.log(sigma_i) - 0.5 * ((y_i - theta_i) / sigma_i) ** 2

    Now, we can specialise these functions with :func:`functools.partial` so the resulting functions
    match the signature expected by :func:`loo_moment_match()`:

    .. ipython::
        ::

        In [4]: from functools import partial
           ...: log_prob_fn = partial(
           ...:     log_prob_upars,
           ...:     sigmas=sigmas,
           ...:     y=y_obs,
           ...:     schools=schools,
           ...:     obs_dim=obs_dim,
           ...: )
           ...: log_lik_i_fn = partial(
           ...:     log_lik_i_upars,
           ...:     sigmas=sigmas,
           ...:     y=y_obs,
           ...:     schools=schools,
           ...:     obs_dim=obs_dim,
           ...: )

    Finally, we can run moment matching using the prepared inputs. Now, we
    have no problematic observations anymore:

    .. ipython::
        :okwarning:

        In [5]: from arviz_stats import loo_moment_match
           ...: loo_mm = loo_moment_match(
           ...:     idata,
           ...:     loo_orig,
           ...:     upars=upars,
           ...:     log_prob_upars_fn=log_prob_fn,
           ...:     log_lik_i_upars_fn=log_lik_i_fn,
           ...:     var_name="obs",
           ...:     split=True,
           ...: )
           ...: loo_mm

    Notes
    -----
    The moment matching algorithm considers three affine transformations of the posterior draws:
    For a specific draw :math:`\theta^{(s)}`, a generic affine transformation includes a square
    matrix :math:`\mathbf{A}` representing a linear map and a vector :math:`\mathbf{b}`
    representing a translation such that

    .. math::
        T : \theta^{(s)} \mapsto \mathbf{A}\theta^{(s)} + \mathbf{b}
        =: \theta^{*{(s)}}.

    The first transformation, :math:`T_1`, is a translation that matches the mean of the sample
    to its importance weighted mean given by

    .. math::
        \mathbf{\theta^{*{(s)}}} = T_1(\mathbf{\theta^{(s)}}) =
        \mathbf{\theta^{(s)}} - \bar{\theta} + \bar{\theta}_w,

    where :math:`\bar{\theta}` is the mean of the sample and :math:`\bar{\theta}_w` is the
    importance weighted mean of the sample. The second transformation, :math:`T_2`, is a scaling
    that matches the marginal variances in addition to the means given by

    .. math::
        \mathbf{\theta^{*{(s)}}} = T_2(\mathbf{\theta^{(s)}}) =
        \mathbf{v}^{1/2}_w \circ \mathbf{v}^{-1/2} \circ (\mathbf{\theta^{(s)}} - \bar{\theta}) +
        \bar{\theta}_w,

    where :math:`\mathbf{v}` and :math:`\mathbf{v}_w` are the sample and weighted variances, and
    :math:`\circ` denotes the pointwise product of the elements of two vectors. The third
    transformation, :math:`T_3`, is a covariance transformation that matches the covariance matrix
    of the sample to its importance weighted covariance matrix given by

    .. math::
        \mathbf{\theta^{*{(s)}}} = T_3(\mathbf{\theta^{(s)}}) =
        \mathbf{L}_w \mathbf{L}^{-1} (\mathbf{\theta^{(s)}} - \bar{\theta}) + \bar{\theta}_w,

    where :math:`\mathbf{L}` and :math:`\mathbf{L}_w` are the Cholesky decompositions of the
    covariance matrix and the weighted covariance matrix, respectively, e.g.,

    .. math::
        \mathbf{LL}^T = \mathbf{\Sigma} = \frac{1}{S} \sum_{s=1}^S (\mathbf{\theta^{(s)}} -
        \bar{\theta}) (\mathbf{\theta^{(s)}} - \bar{\theta})^T

    and

    .. math::
        \mathbf{L}_w \mathbf{L}_w^T = \mathbf{\Sigma}_w = \frac{\frac{1}{S} \sum_{s=1}^S
        w^{(s)} (\mathbf{\theta^{(s)}} - \bar{\theta}_w) (\mathbf{\theta^{(s)}} -
        \bar{\theta}_w)^T}{\sum_{s=1}^S w^{(s)}}.

    We iterate on :math:`T_1` repeatedly and move onto :math:`T_2` and :math:`T_3` only
    if :math:`T_1` fails to yield a Pareto-k statistic below the threshold.

    See Also
    --------
    loo : Standard PSIS-LOO-CV.
    reloo : Exact re-fitting for problematic observations.

    References
    ----------

    .. [1] Paananen, T., Piironen, J., Buerkner, P.-C., Vehtari, A. (2021). Implicitly Adaptive
        Importance Sampling. Statistics and Computing. 31(2) (2021)
        https://doi.org/10.1007/s11222-020-09982-2
        arXiv preprint https://arxiv.org/abs/1906.08850.
    .. [2] Vehtari et al. *Practical Bayesian model evaluation using leave-one-out cross-validation
        and WAIC*. Statistics and Computing. 27(5) (2017) https://doi.org/10.1007/s11222-016-9696-4
        arXiv preprint https://arxiv.org/abs/1507.04544.
    .. [3] Vehtari et al. *Pareto Smoothed Importance Sampling*.
        Journal of Machine Learning Research, 25(72) (2024) https://jmlr.org/papers/v25/19-556.html
        arXiv preprint https://arxiv.org/abs/1507.02646
    """
    if not isinstance(loo_orig, ELPDData):
        raise TypeError("loo_orig must be an ELPDData object.")
    if loo_orig.pareto_k is None or loo_orig.elpd_i is None:
        raise ValueError(
            "Moment matching requires pointwise LOO results with Pareto k values. "
            "Please compute the initial LOO with pointwise=True."
        )

    sample_dims = ["chain", "draw"]

    if upars is None:
        if hasattr(data, "unconstrained_posterior"):
            upars_ds = azb.get_unconstrained_samples(data, return_dataset=True)
            upars = dataset_to_dataarray(
                upars_ds, sample_dims=sample_dims, new_dim="unconstrained_parameter"
            )
        else:
            raise ValueError(
                "upars must be provided or data must contain an 'unconstrained_posterior' group."
            )

    if not isinstance(upars, xr.DataArray):
        raise TypeError("upars must be a DataArray.")
    if not all(dim_name in upars.dims for dim_name in sample_dims):
        raise ValueError(f"upars must have dimensions {sample_dims}.")

    param_dim_list = [dim for dim in upars.dims if dim not in sample_dims]

    if len(param_dim_list) == 0:
        param_dim_name = "upars_dim"
        upars = upars.expand_dims(dim={param_dim_name: 1})
    elif len(param_dim_list) == 1:
        param_dim_name = param_dim_list[0]
    else:
        raise ValueError("upars must have at most one dimension besides 'chain' and 'draw'.")

    loo_data = deepcopy(loo_orig)
    loo_data.method = "loo_moment_match"
    pointwise = rcParams["stats.ic_pointwise"] if pointwise is None else pointwise

    loo_inputs = _prepare_loo_inputs(data, var_name)
    log_likelihood = loo_inputs.log_likelihood
    obs_dims = loo_inputs.obs_dims
    n_samples = loo_inputs.n_samples
    var_name = loo_inputs.var_name

    n_params = upars.sizes[param_dim_name]
    n_data_points = loo_orig.n_data_points

    if reff is None:
        reff = _get_r_eff(data, n_samples)

    try:
        orig_log_prob = log_prob_upars_fn(upars)
        if not isinstance(orig_log_prob, xr.DataArray):
            raise TypeError("log_prob_upars_fn must return a DataArray.")
        if not all(dim in orig_log_prob.dims for dim in sample_dims):
            raise ValueError(f"Original log probability must have dimensions {sample_dims}.")
        if len(orig_log_prob.dims) != len(sample_dims):
            raise ValueError(
                f"Original log probability should only have dimensions {sample_dims}, "
                f"found {orig_log_prob.dims}"
            )
    except Exception as e:
        raise ValueError(f"Error executing log_prob_upars_fn: {e}") from e

    if k_threshold is None:
        k_threshold = min(1 - 1 / np.log10(n_samples), 0.7) if n_samples > 1 else 0.7

    loo_data.influence_pareto_k = loo_data.pareto_k.copy()

    ks = loo_data.pareto_k.stack(__obs__=obs_dims).transpose("__obs__").values
    bad_obs_indices = np.where(ks > k_threshold)[0]

    if len(bad_obs_indices) == 0:
        warnings.warn("No Pareto k values exceed the threshold. Returning original LOO data.")
        if not pointwise:
            loo_data.elpd_i = None
            loo_data.pareto_k = None
            loo_data.influence_pareto_k = None
            if hasattr(loo_data, "p_loo_i"):
                loo_data.p_loo_i = None
        return loo_data

    lpd = logsumexp(log_likelihood, dims=sample_dims, b=1 / n_samples)
    loo_data.p_loo_i = lpd - loo_data.elpd_i
    kfs = np.zeros(n_data_points)
    log_weights = getattr(loo_data, "log_weights", None)
    r_eff_data = getattr(loo_data, "r_eff", reff)

    # Moment matching algorithm
    for i in bad_obs_indices:
        mm_result = _loo_moment_match_i(
            i=i,
            upars=upars,
            log_likelihood=log_likelihood,
            log_prob_upars_fn=log_prob_upars_fn,
            log_lik_i_upars_fn=log_lik_i_upars_fn,
            max_iters=max_iters,
            k_threshold=k_threshold,
            split=split,
            cov=cov,
            orig_log_prob=orig_log_prob,
            ks=ks,
            log_weights=log_weights,
            pareto_k=loo_data.pareto_k,
            r_eff=r_eff_data,
            sample_dims=sample_dims,
            obs_dims=obs_dims,
            n_samples=n_samples,
            n_params=n_params,
            param_dim_name=param_dim_name,
            var_name=var_name,
        )

        kfs[i] = mm_result.kfs_i

        if mm_result.final_ki < mm_result.original_ki:
            new_elpd_i = logsumexp(
                mm_result.final_log_liki + mm_result.final_lwi, dims=sample_dims
            ).item()
            original_log_liki = _get_log_likelihood_i(log_likelihood, i, obs_dims)

            _update_loo_data_i(
                loo_data,
                i,
                new_elpd_i,
                mm_result.final_ki,
                mm_result.final_log_liki,
                sample_dims,
                obs_dims,
                n_samples,
                original_log_liki,
                suppress_warnings=True,
            )
        else:
            warnings.warn(
                f"Observation {i}: Moment matching did not improve k "
                f"({mm_result.original_ki:.2f} -> {mm_result.final_ki:.2f}). Reverting.",
                UserWarning,
                stacklevel=2,
            )
            if hasattr(loo_orig, "p_loo_i") and loo_orig.p_loo_i is not None:
                if len(obs_dims) == 1:
                    idx_dict = {obs_dims[0]: i}
                else:
                    coords = np.unravel_index(i, tuple(loo_data.elpd_i.sizes[d] for d in obs_dims))
                    idx_dict = dict(zip(obs_dims, coords))
                loo_data.p_loo_i[idx_dict] = loo_orig.p_loo_i[idx_dict]

    final_ks = loo_data.pareto_k.stack(__obs__=obs_dims).transpose("__obs__").values

    if np.any(final_ks[bad_obs_indices] > k_threshold):
        warnings.warn(
            f"After Moment Matching, {np.sum(final_ks > k_threshold)} observations still have "
            f"Pareto k > {k_threshold:.2f}.",
            UserWarning,
            stacklevel=2,
        )

    if not split and np.any(kfs > k_threshold):
        warnings.warn(
            "The accuracy of self-normalized importance sampling may be bad. "
            "Setting the argument 'split' to 'True' will likely improve accuracy.",
            UserWarning,
            stacklevel=2,
        )

    elpd_raw = logsumexp(log_likelihood, dims=sample_dims, b=1 / n_samples).sum().values
    loo_data.p = elpd_raw - loo_data.elpd

    if not pointwise:
        loo_data.elpd_i = None
        loo_data.pareto_k = None
        loo_data.influence_pareto_k = None
        if hasattr(loo_data, "p_loo_i"):
            loo_data.p_loo_i = None
    return loo_data


def _split_moment_match(
    upars,
    cov,
    total_shift,
    total_scaling,
    total_mapping,
    i,
    reff,
    log_prob_upars_fn,
    log_lik_i_upars_fn,
):
    r"""Split moment matching importance sampling for PSIS-LOO-CV.

    Applies affine transformations based on the total moment matching transformation
    to half of the posterior draws, leaving the other half unchanged. These approximations
    to the leave-one-out posterior are then combined using multiple importance sampling.

    Based on the implicit adaptive importance sampling algorithm of [1]_ and the
    PSIS-LOO-CV method of [2]_ and [3]_.

    Parameters
    ----------
    upars : DataArray
        A DataArray representing the posterior draws of the model parameters in the
        unconstrained space. Must contain the dimensions `chain` and `draw` and a final
        dimension representing the different unconstrained parameters.
    cov : bool
        Whether to match the full covariance matrix of the samples (True) or just the
        marginal variances (False). Using the full covariance is more computationally
        expensive.
    total_shift : ndarray
        Vector containing the total shift (translation) applied to the parameters. Shape should
        match the parameter dimension of ``upars``.
    total_scaling : ndarray
        Vector containing the total scaling factors for the marginal variances. Shape should
        match the parameter dimension of ``upars``.
    total_mapping : ndarray
        Square matrix representing the linear transformation applied to the covariance matrix.
        Shape should be (d, d) where d is the parameter dimension.
    i : int
        Index of the specific observation to be left out for computing leave-one-out
        likelihood.
    reff : float
        Relative MCMC efficiency, ``ess / n`` i.e. number of effective samples divided by the number
        of actual samples.
    log_prob_upars_fn : callable
        Function that computes the log probability density of the *full posterior*
        distribution evaluated at unconstrained parameter draws.
        The function signature is ``log_prob_upars_fn(upars)`` where ``upars``
        is a :class:`~xarray.DataArray` of unconstrained parameter draws.
        It should return a :class:`~xarray.DataArray` with dimensions ``chain``, ``draw``.
    log_lik_i_upars_fn : callable
        Function that computes the log-likelihood of the *left-out observation* ``i``
        evaluated at unconstrained parameter draws.
        The function signature is ``log_lik_i_upars_fn(upars, i)`` where ``upars``
        is a :class:`~xarray.DataArray` of unconstrained parameter draws and ``i``
        is the integer index of the observation.
        It should return a :class:`~xarray.DataArray` with dimensions ``chain``, ``draw``.

    Returns
    -------
    SplitMomentMatch
        A namedtuple containing:

        - lwi: Updated log importance weights for each sample
        - lwfi: Updated log importance weights for full distribution
        - log_liki: Updated log likelihood values for the specific observation
        - reff: Relative MCMC efficiency (updated based on the split samples)

    References
    ----------

    .. [1] Paananen, T., Piironen, J., Buerkner, P.-C., Vehtari, A. (2021). *Implicitly Adaptive
        Importance Sampling*. Statistics and Computing. 31(2) (2021)
        https://doi.org/10.1007/s11222-020-09982-2
        arXiv preprint https://arxiv.org/abs/1906.08850.
    .. [2] Vehtari et al. *Practical Bayesian model evaluation using leave-one-out cross-validation
        and WAIC*. Statistics and Computing. 27(5) (2017) https://doi.org/10.1007/s11222-016-9696-4
        arXiv preprint https://arxiv.org/abs/1507.04544.
    .. [3] Vehtari et al. *Pareto Smoothed Importance Sampling*.
        Journal of Machine Learning Research, 25(72) (2024) https://jmlr.org/papers/v25/19-556.html
        arXiv preprint https://arxiv.org/abs/1507.02646
    """
    sample_dims = ["chain", "draw"]
    param_dim = next(dim for dim in upars.dims if dim not in sample_dims)

    dim = upars.sizes[param_dim]
    n_chains = upars.sizes["chain"]
    n_draws = upars.sizes["draw"]
    n_samples = n_chains * n_draws
    n_samples_half = n_samples // 2

    stack_dims = ["draw", "chain"]
    upars_stacked = upars.stack(__sample__=stack_dims).transpose("__sample__", param_dim)
    mean_original = upars_stacked.mean(dim="__sample__")

    # Forward transformation
    upars_trans = upars_stacked - mean_original
    upars_trans = upars_trans * xr.DataArray(total_scaling, dims=param_dim)

    if cov and dim > 0:
        upars_trans = xr.DataArray(
            upars_trans.data @ total_mapping.T,
            coords=upars_trans.coords,
            dims=upars_trans.dims,
        )

    # Inverse Transformation
    upars_trans = upars_trans + (xr.DataArray(total_shift, dims=param_dim) + mean_original)
    upars_trans_inv = upars_stacked - (xr.DataArray(total_shift, dims=param_dim) + mean_original)

    if cov and dim > 0:
        try:
            inv_mapping_t = np.linalg.inv(total_mapping.T)
            upars_trans_inv = xr.DataArray(
                upars_trans_inv.data @ inv_mapping_t,
                coords=upars_trans_inv.coords,
                dims=upars_trans_inv.dims,
            )
        except np.linalg.LinAlgError:
            warnings.warn("Could not invert mapping matrix. Using identity.", UserWarning)

    upars_trans_inv = upars_trans_inv / xr.DataArray(total_scaling, dims=param_dim)
    upars_trans_inv = upars_trans_inv + (mean_original - xr.DataArray(total_shift, dims=param_dim))

    upars_trans_half_stacked = upars_stacked.copy(deep=True)
    upars_trans_half_stacked.data[:n_samples_half, :] = upars_trans.data[:n_samples_half, :]
    upars_trans_half = upars_trans_half_stacked.unstack("__sample__").transpose(
        *reversed(stack_dims), param_dim
    )

    upars_trans_half_inv_stacked = upars_stacked.copy(deep=True)
    upars_trans_half_inv_stacked.data[n_samples_half:, :] = upars_trans_inv.data[n_samples_half:, :]
    upars_trans_half_inv = upars_trans_half_inv_stacked.unstack("__sample__").transpose(
        *reversed(stack_dims), param_dim
    )

    try:
        log_prob_half_trans = log_prob_upars_fn(upars_trans_half)
        if not isinstance(log_prob_half_trans, xr.DataArray):
            raise TypeError("log_prob_upars_fn must return a DataArray.")
        if not all(dim in log_prob_half_trans.dims for dim in sample_dims) or len(
            log_prob_half_trans.dims
        ) != len(sample_dims):
            raise ValueError(
                f"log_prob_upars_fn must return a DataArray with dimensions {sample_dims}, "
                f"but got {log_prob_half_trans.dims}"
            )

        log_prob_half_trans_inv = log_prob_upars_fn(upars_trans_half_inv)
        if not isinstance(log_prob_half_trans_inv, xr.DataArray):
            raise TypeError("log_prob_upars_fn must return a DataArray.")
        if not all(dim in log_prob_half_trans_inv.dims for dim in sample_dims) or len(
            log_prob_half_trans_inv.dims
        ) != len(sample_dims):
            raise ValueError(
                f"log_prob_upars_fn must return a DataArray with dimensions {sample_dims}, "
                f"but got {log_prob_half_trans_inv.dims}"
            )
    except Exception as e:
        raise ValueError(
            f"Could not compute log probabilities for transformed parameters: {e}"
        ) from e

    try:
        log_liki_half = log_lik_i_upars_fn(upars_trans_half, i)
        if not all(dim in log_liki_half.dims for dim in sample_dims) or len(
            log_liki_half.dims
        ) != len(sample_dims):
            raise ValueError(
                f"log_lik_i_upars_fn must return a DataArray with dimensions {sample_dims}"
            )
        if (
            log_liki_half.sizes["chain"] != upars.sizes["chain"]
            or log_liki_half.sizes["draw"] != upars.sizes["draw"]
        ):
            raise ValueError(
                "log_lik_i_upars_fn output shape does not match input sample dimensions"
            )
    except Exception as e:
        raise ValueError(f"Could not compute log likelihood for observation {i}: {e}") from e

    log_jacobian_det = 0.0
    if dim > 0:
        log_jacobian_det = -np.sum(np.log(total_scaling))
        try:
            log_jacobian_det -= np.log(np.linalg.det(total_mapping))
        except np.linalg.LinAlgError:
            log_jacobian_det -= np.inf

    log_prob_half_trans_inv_adj = log_prob_half_trans_inv + log_jacobian_det

    # Multiple importance sampling
    use_forward_log_prob = log_prob_half_trans > log_prob_half_trans_inv_adj
    raw_log_weights_half = -log_liki_half + log_prob_half_trans

    log_sum_terms = xr.where(
        use_forward_log_prob,
        log_prob_half_trans
        + xr.ufuncs.log1p(np.exp(log_prob_half_trans_inv_adj - log_prob_half_trans)),
        log_prob_half_trans_inv_adj
        + xr.ufuncs.log1p(np.exp(log_prob_half_trans - log_prob_half_trans_inv_adj)),
    )

    raw_log_weights_half -= log_sum_terms
    raw_log_weights_half = xr.where(np.isnan(raw_log_weights_half), -np.inf, raw_log_weights_half)
    raw_log_weights_half = xr.where(
        np.isposinf(raw_log_weights_half), -np.inf, raw_log_weights_half
    )

    # PSIS smoothing for half posterior
    lwi_psis_da, _ = _wrap__psislw(raw_log_weights_half, sample_dims, reff)

    lr_full = lwi_psis_da + log_liki_half
    lr_full = xr.where(np.isnan(lr_full) | (np.isinf(lr_full) & (lr_full > 0)), -np.inf, lr_full)

    # PSIS smoothing for full posterior
    lwfi_psis_da, _ = _wrap__psislw(lr_full, sample_dims, reff)
    n_chains = upars.sizes["chain"]

    if n_chains == 1:
        reff_updated = reff
    else:
        log_liki_half_1 = log_liki_half.isel(
            chain=slice(None), draw=slice(0, n_samples_half // n_chains)
        )
        log_liki_half_2 = log_liki_half.isel(
            chain=slice(None), draw=slice(n_samples_half // n_chains, None)
        )

        liki_half_1 = np.exp(log_liki_half_1)
        liki_half_2 = np.exp(log_liki_half_2)

        ess_1 = liki_half_1.azstats.ess(method="mean")
        ess_2 = liki_half_2.azstats.ess(method="mean")

        ess_1_value = ess_1.values if hasattr(ess_1, "values") else ess_1
        ess_2_value = ess_2.values if hasattr(ess_2, "values") else ess_2

        n_samples_1 = log_liki_half_1.size
        n_samples_2 = log_liki_half_2.size

        r_eff_1 = ess_1_value / n_samples_1
        r_eff_2 = ess_2_value / n_samples_2

        reff_updated = min(r_eff_1, r_eff_2)

    return SplitMomentMatch(
        lwi=lwi_psis_da,
        lwfi=lwfi_psis_da,
        log_liki=log_liki_half,
        reff=reff_updated,
    )


def _loo_moment_match_i(
    i,
    upars,
    log_likelihood,
    log_prob_upars_fn,
    log_lik_i_upars_fn,
    max_iters,
    k_threshold,
    split,
    cov,
    orig_log_prob,
    ks,
    log_weights,
    pareto_k,
    r_eff,
    sample_dims,
    obs_dims,
    n_samples,
    n_params,
    param_dim_name,
    var_name,
):
    """Compute moment matching for a single observation."""
    n_chains = upars.sizes["chain"]
    n_draws = upars.sizes["draw"]

    log_liki = _get_log_likelihood_i(log_likelihood, i, obs_dims).squeeze(drop=True)

    if isinstance(r_eff, xr.DataArray):
        reff_i = _get_r_eff_i(r_eff, i, obs_dims)
    elif r_eff is not None:
        reff_i = r_eff
    else:
        liki = np.exp(log_liki)
        liki_reshaped = liki.values.reshape(n_chains, n_draws).T
        ess_val = ess(liki_reshaped, method="mean").item()
        reff_i = ess_val / n_samples if n_samples > 0 else 1.0

    original_ki = ks[i]

    if log_weights is not None:
        log_weights_i, ki = _get_weights_and_k_i(
            log_weights=log_weights,
            pareto_k=pareto_k,
            i=i,
            obs_dims=obs_dims,
            sample_dims=sample_dims,
            data=log_likelihood,
            n_samples=n_samples,
            reff=reff_i,
            log_lik_i=log_liki,
            var_name=var_name,
        )
        lwi = log_weights_i.squeeze(drop=True).transpose(*sample_dims).astype(np.float64)
    else:
        log_ratio_i_init = -log_liki
        lwi, ki = _wrap__psislw(log_ratio_i_init, sample_dims, reff_i)

    upars_i = upars.copy(deep=True)
    total_shift = np.zeros(upars_i.sizes[param_dim_name])
    total_scaling = np.ones(upars_i.sizes[param_dim_name])
    total_mapping = np.eye(upars_i.sizes[param_dim_name])

    iterind = 1
    transformations_applied = False
    kfs_i = 0

    while iterind <= max_iters and ki > k_threshold:
        if iterind == max_iters:
            warnings.warn(
                f"Maximum number of moment matching iterations ({max_iters}) reached "
                f"for observation {i}. Final Pareto k is {ki:.2f}.",
                UserWarning,
                stacklevel=2,
            )
            break

        # Try Mean Shift
        try:
            shift_res = _shift(upars_i, lwi)
            quantities_i = _update_quantities_i(
                shift_res.upars,
                i,
                orig_log_prob,
                log_prob_upars_fn,
                log_lik_i_upars_fn,
                reff_i,
                sample_dims,
            )
            if quantities_i.ki < ki:
                ki = quantities_i.ki
                lwi = quantities_i.lwi
                log_liki = quantities_i.log_liki
                kfs_i = quantities_i.kfi
                upars_i = shift_res.upars
                total_shift = total_shift + shift_res.shift
                transformations_applied = True
                iterind += 1
                continue  # Restart, try mean shift again

        except RuntimeError as e:
            warnings.warn(
                f"Error during mean shift calculation for observation {i}: {e}. "
                "Stopping moment matching for this observation.",
                UserWarning,
                stacklevel=2,
            )
            break

        # Try Scale Shift
        try:
            scale_res = _shift_and_scale(upars_i, lwi)
            quantities_i = _update_quantities_i(
                scale_res.upars,
                i,
                orig_log_prob,
                log_prob_upars_fn,
                log_lik_i_upars_fn,
                reff_i,
                sample_dims,
            )
            if quantities_i.ki < ki:
                ki = quantities_i.ki
                lwi = quantities_i.lwi
                log_liki = quantities_i.log_liki
                kfs_i = quantities_i.kfi
                upars_i = scale_res.upars
                total_shift = total_shift + scale_res.shift
                total_scaling = total_scaling * scale_res.scaling
                transformations_applied = True
                iterind += 1
                continue  # Restart, try mean shift again

        except RuntimeError as e:
            warnings.warn(
                f"Error during scale shift calculation for observation {i}: {e}. "
                "Stopping moment matching for this observation.",
                UserWarning,
                stacklevel=2,
            )
            break

        # Try Covariance Shift
        if cov and n_samples >= 10 * n_params:
            try:
                cov_res = _shift_and_cov(upars_i, lwi)
                quantities_i = _update_quantities_i(
                    cov_res.upars,
                    i,
                    orig_log_prob,
                    log_prob_upars_fn,
                    log_lik_i_upars_fn,
                    reff_i,
                    sample_dims,
                )
                if quantities_i.ki < ki:
                    ki = quantities_i.ki
                    lwi = quantities_i.lwi
                    log_liki = quantities_i.log_liki
                    kfs_i = quantities_i.kfi
                    upars_i = cov_res.upars
                    total_shift = total_shift + cov_res.shift
                    total_mapping = cov_res.mapping @ total_mapping
                    transformations_applied = True
                    iterind += 1
                    continue  # Restart, try mean shift again

            except RuntimeError as e:
                warnings.warn(
                    f"Error during covariance shift calculation for observation {i}: {e}. "
                    "Stopping moment matching for this observation.",
                    UserWarning,
                    stacklevel=2,
                )
                break

        break

    if split and transformations_applied:
        try:
            split_res = _split_moment_match(
                upars=upars,
                cov=cov,
                total_shift=total_shift,
                total_scaling=total_scaling,
                total_mapping=total_mapping,
                i=i,
                reff=reff_i,
                log_prob_upars_fn=log_prob_upars_fn,
                log_lik_i_upars_fn=log_lik_i_upars_fn,
            )

            final_log_liki = split_res.log_liki
            final_lwi = split_res.lwi
            final_ki = ki
            reff_i = split_res.reff

        except RuntimeError as e:
            warnings.warn(
                f"Error during split moment matching for observation {i}: {e}. "
                "Using non-split transformation result.",
                UserWarning,
                stacklevel=2,
            )
            final_log_liki = log_liki
            final_lwi = lwi
            final_ki = ki
    else:
        final_log_liki = log_liki
        final_lwi = lwi
        final_ki = ki

        if transformations_applied:
            liki_final = np.exp(final_log_liki)
            liki_final_reshaped = liki_final.values.reshape(n_chains, n_draws).T
            ess_val_final = ess(liki_final_reshaped, method="mean").item()
            reff_i = ess_val_final / n_samples if n_samples > 0 else 1.0

    return LooMomentMatchResult(
        final_log_liki=final_log_liki,
        final_lwi=final_lwi,
        final_ki=final_ki,
        kfs_i=kfs_i,
        reff_i=reff_i,
        original_ki=original_ki,
        i=i,
    )


def _update_loo_data_i(
    loo_data,
    i,
    new_elpd_i,
    new_pareto_k,
    log_liki,
    sample_dims,
    obs_dims,
    n_samples,
    original_log_liki=None,
    suppress_warnings=False,
):
    """Update the ELPDData object for a single observation."""
    if loo_data.elpd_i is None or loo_data.pareto_k is None:
        raise ValueError("loo_data must contain pointwise elpd_i and pareto_k values.")

    lpd_i_log_lik = original_log_liki if original_log_liki is not None else log_liki
    lpd_i = logsumexp(lpd_i_log_lik, dims=sample_dims, b=1 / n_samples).item()
    p_loo_i = lpd_i - new_elpd_i

    if len(obs_dims) == 1:
        idx_dict = {obs_dims[0]: i}
    else:
        coords = np.unravel_index(i, tuple(loo_data.elpd_i.sizes[d] for d in obs_dims))
        idx_dict = dict(zip(obs_dims, coords))

    loo_data.elpd_i[idx_dict] = new_elpd_i
    loo_data.pareto_k[idx_dict] = new_pareto_k

    if not hasattr(loo_data, "p_loo_i") or loo_data.p_loo_i is None:
        loo_data.p_loo_i = xr.full_like(loo_data.elpd_i, np.nan)

    loo_data.p_loo_i[idx_dict] = p_loo_i
    loo_data.elpd = np.nansum(loo_data.elpd_i.values)
    loo_data.se = np.sqrt(loo_data.n_data_points * np.nanvar(loo_data.elpd_i.values, ddof=1))

    loo_data.warning, loo_data.good_k = _warn_pareto_k(
        loo_data.pareto_k.values[~np.isnan(loo_data.pareto_k.values)],
        loo_data.n_samples,
        suppress=suppress_warnings,
    )


def _update_quantities_i(
    upars,
    i,
    orig_log_prob,
    log_prob_upars_fn,
    log_lik_i_upars_fn,
    reff_i,
    sample_dims,
):
    """Update the moment matching quantities for a single observation."""
    log_prob_new = log_prob_upars_fn(upars)
    log_liki_new = log_lik_i_upars_fn(upars, i)

    log_ratio_i = -log_liki_new + log_prob_new - orig_log_prob
    lwi_new, ki_new = _wrap__psislw(log_ratio_i, sample_dims, reff_i)

    log_ratio_full = log_prob_new - orig_log_prob
    lwfi_new, kfi_new = _wrap__psislw(log_ratio_full, sample_dims, reff_i)

    return UpdateQuantities(
        lwi=lwi_new,
        lwfi=lwfi_new,
        ki=ki_new,
        kfi=kfi_new,
        log_liki=log_liki_new,
    )


def _wrap__psislw(log_weights, sample_dims, r_eff):
    """Apply PSIS smoothing over sample dimensions."""
    if not isinstance(log_weights, xr.DataArray):
        raise TypeError("log_weights must be an xarray.DataArray")

    missing_dims = [dim for dim in sample_dims if dim not in log_weights.dims]

    if missing_dims:
        raise ValueError(
            f"All sample dimensions must be present in the input; missing {missing_dims}."
        )

    other_dims = [dim for dim in log_weights.dims if dim not in sample_dims]
    if other_dims:
        raise ValueError(
            "_wrap__psislw expects `log_weights` to include only sample dimensions; "
            f"found extra dims {other_dims}."
        )

    stacked = log_weights.stack(__sample__=sample_dims)
    stacked_for_psis = -stacked

    try:
        lw_stacked, k = stacked_for_psis.azstats.psislw(dim="__sample__", r_eff=r_eff)
    except ValueError as err:
        err_message = str(err)
        fallback_errors = ("All tail values are the same", "n_draws_tail must be at least 5")
        if not any(msg in err_message for msg in fallback_errors):
            raise

        log_norm = logsumexp(stacked, dims="__sample__")
        lw_stacked = stacked - log_norm

        k = np.inf

    lw = lw_stacked.unstack("__sample__").transpose(*log_weights.dims)

    if isinstance(k, xr.DataArray):
        if k.dims:
            raise ValueError("Unexpected dimensions on Pareto k output; expected scalar result.")
        k_val = k.item()
    elif isinstance(k, np.ndarray):
        if k.ndim != 0:
            raise ValueError("Unexpected array shape for Pareto k; expected scalar result.")
        k_val = k.item()
    else:
        try:
            k_val = k
        except (TypeError, ValueError) as exc:
            raise TypeError("Unable to convert PSIS tail index to float") from exc
    return lw, k_val
