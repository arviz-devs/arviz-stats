"""Pareto-smoothed importance sampling LOO (PSIS-LOO-CV)."""

import warnings

from arviz_base import rcParams
from xarray_einstats.stats import logsumexp

from arviz_stats.loo.helper_loo import (
    _check_log_jacobian,
    _compute_loo_results,
    _get_log_likelihood_i,
    _get_r_eff,
    _get_weights_and_k_i,
    _log_lik_i,
    _prepare_loo_inputs,
    _warn_pareto_k,
    _warn_pointwise_loo,
)
from arviz_stats.utils import ELPDData


def loo(
    data,
    pointwise=None,
    var_name=None,
    reff=None,
    log_weights=None,
    pareto_k=None,
    log_jacobian=None,
    mixture=False,
):
    r"""Compute Pareto-smoothed importance sampling leave-one-out cross-validation (PSIS-LOO-CV).

    Estimates the expected log pointwise predictive density (elpd) using Pareto-smoothed
    importance sampling leave-one-out cross-validation (PSIS-LOO-CV). Also calculates LOO's
    standard error and the effective number of parameters. The method is described in [2]_
    and [3]_.

    Parameters
    ----------
    data : DataTree or InferenceData
        Input data. It should contain the posterior and the log_likelihood groups.
    pointwise : bool, optional
        If True the pointwise predictive accuracy will be returned. Defaults to
        ``rcParams["stats.ic_pointwise"]``.
    var_name : str, optional
        The name of the variable in log_likelihood groups storing the pointwise log
        likelihood data to use for loo computation.
    reff : float, optional
        Relative MCMC efficiency, ``ess / n`` i.e. number of effective samples divided by the number
        of actual samples. Computed from trace by default.
    log_weights : DataArray, optional
        Smoothed log weights. It must have the same shape as the log likelihood data.
        Defaults to None. If not provided, it will be computed using the PSIS-LOO method.
        Must be provided together with pareto_k or both must be None.
    pareto_k : DataArray, optional
        Pareto shape values. It must have the same shape as the log likelihood data.
        Defaults to None. If not provided, it will be computed using the PSIS-LOO method.
        Must be provided together with log_weights or both must be None.
    log_jacobian : DataArray, optional
        Log-Jacobian adjustment for variable transformations. Required when the model was fitted
        on transformed response data :math:`z = T(y)` but you want to compute ELPD on the
        original response scale :math:`y`. The value should be :math:`\log|\frac{dz}{dy}|`
        (the log absolute value of the derivative of the transformation). Must be a DataArray
        with dimensions matching the observation dimensions.
    mixture : bool, optional
        If True, use mixture importance sampling LOO (Mix-IS-LOO) instead of PSIS-LOO.
        This is appropriate when the log-likelihood was generated from a mixture distribution.
        The method is described in [1]_. Defaults to False.

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
          ``pointwise=True``
        - **pareto_k**: :class:`~xarray.DataArray` with Pareto shape values,
          only if ``pointwise=True``
        - **approx_posterior**: False (not used for standard LOO)
        - **log_weights**: Smoothed log weights.
        - **log_jacobian**: Log-Jacobian adjustment for variable transformations.

    Examples
    --------
    Calculate LOO of a model:

    .. ipython::

        In [1]: from arviz_stats import loo
           ...: from arviz_base import load_arviz_data
           ...: data = load_arviz_data("centered_eight")
           ...: loo_data = loo(data)
           ...: loo_data

    Return the pointwise values:

    .. ipython::

        In [2]: loo_data.elpd_i

    If a model was fit on a transformed response :math:`z = T(y)` (e.g., :math:`z=\sqrt{y}`), you
    can report LOO results on the original :math:`y` scale by adding the log-Jacobian of the forward
    transform to each pointwise log-likelihood value via the ``log_jacobian`` argument.

    For example, with :math:`z=\sqrt{y}` (and :math:`y>0`), the derivative of :math:`z` with respect
    to :math:`y` is :math:`\tfrac{dz}{dy} = \tfrac{1}{2\sqrt{y}}`. So, the log-Jacobian is
    :math:`\log\!\left|\tfrac{dz}{dy}\right| = -\log 2 - \tfrac{1}{2}\log y`:

    .. ipython::

        In [3]: import numpy as np
           ...: import xarray as xr
           ...: from arviz_stats import loo
           ...: from arviz_base import load_arviz_data
           ...: data = load_arviz_data("centered_eight")
           ...: y = data.observed_data["obs"]
           ...: y_positive = y - y.min() + 1  # Positive values for sqrt
           ...: log_jacobian_values = -np.log(2) - 0.5 * np.log(y_positive)
           ...: log_jacobian = xr.DataArray(
           ...:     log_jacobian_values,
           ...:     dims=y.dims,
           ...:     coords=y.coords
           ...: )
           ...: loo_data_adjusted = loo(data, log_jacobian=log_jacobian)
           ...: loo_data_adjusted

    See Also
    --------
    :func:`compare` : Compare models based on their ELPD.
    :func:`arviz_plots.plot_compare` : Summary plot for model comparison.

    References
    ----------

    .. [1] Silva and Zanella. *Robust Leave-One-Out Cross-Validation for High-Dimensional
       Bayesian Models*. Journal of the American Statistical Association. 119(547) (2023)
       2369-2381. https://doi.org/10.1080/01621459.2023.2257893
       arXiv preprint https://arxiv.org/abs/2209.09190

    .. [2] Vehtari et al. *Practical Bayesian model evaluation using leave-one-out cross-validation
       and WAIC*. Statistics and Computing. 27(5) (2017) https://doi.org/10.1007/s11222-016-9696-4
       arXiv preprint https://arxiv.org/abs/1507.04544.

    .. [3] Vehtari et al. *Pareto Smoothed Importance Sampling*.
       Journal of Machine Learning Research, 25(72) (2024) https://jmlr.org/papers/v25/19-556.html
       arXiv preprint https://arxiv.org/abs/1507.02646
    """
    loo_inputs = _prepare_loo_inputs(data, var_name)
    pointwise = rcParams["stats.ic_pointwise"] if pointwise is None else pointwise

    if reff is None:
        reff = _get_r_eff(data, loo_inputs.n_samples)

    if (log_weights is None) != (pareto_k is None):
        raise ValueError(
            "Both log_weights and pareto_k must be provided together or both must be None. "
            "Only one was provided."
        )

    jacobian_da = _check_log_jacobian(log_jacobian, loo_inputs.obs_dims)

    if log_weights is None and pareto_k is None:
        log_weights, pareto_k = loo_inputs.log_likelihood.azstats.psislw(
            r_eff=reff, dim=loo_inputs.sample_dims
        )

    if mixture:
        warnings.warn(
            "Using mixture=True assumes the log-likelihood values were computed from samples "
            "drawn from a mixture posterior distribution. "
            "See Silva and Zanella (2023) https://doi.org/10.1080/01621459.2023.2257893 "
            "for details on how to sample from the mixture posterior.",
            UserWarning,
            stacklevel=2,
        )
        log_lik = loo_inputs.log_likelihood
        sample_dims = loo_inputs.sample_dims
        n_samples = loo_inputs.n_samples
        n_data_points = loo_inputs.n_data_points

        elpd_i, p_loo_i, mix_log_weights = log_lik.azstats.loo_mixture(
            sample_dims=sample_dims, log_jacobian=jacobian_da
        )

        elpd, elpd_se, p_loo, _ = elpd_i.azstats.loo_summary(p_loo_i)

        _, mix_pareto_k = mix_log_weights.azstats.psislw(r_eff=reff, dim=sample_dims)
        warn_mg, good_k = _warn_pareto_k(mix_pareto_k, n_samples)

        if pointwise:
            _warn_pointwise_loo(elpd, elpd_i.values)

        return ELPDData(
            kind="loo",
            elpd=elpd,
            se=elpd_se,
            p=p_loo,
            n_samples=n_samples,
            n_data_points=n_data_points,
            scale="log",
            warning=warn_mg,
            good_k=good_k,
            elpd_i=elpd_i if pointwise else None,
            pareto_k=mix_pareto_k if pointwise else None,
            approx_posterior=False,
            log_weights=mix_log_weights,
        )

    return _compute_loo_results(
        log_likelihood=loo_inputs.log_likelihood,
        var_name=loo_inputs.var_name,
        pointwise=pointwise,
        sample_dims=loo_inputs.sample_dims,
        n_samples=loo_inputs.n_samples,
        n_data_points=loo_inputs.n_data_points,
        log_weights=log_weights,
        pareto_k=pareto_k,
        approx_posterior=False,
        log_jacobian=jacobian_da,
    )


def loo_i(
    i,
    data,
    var_name=None,
    reff=None,
    log_lik_fn=None,
    log_weights=None,
    pareto_k=None,
    log_jacobian=None,
):
    r"""Compute PSIS-LOO-CV for a single observation.

    Estimates the expected log pointwise predictive density (elpd) using Pareto-smoothed
    importance sampling leave-one-out cross-validation (PSIS-LOO-CV) for a single observation.
    The method is described in [1]_ and [2]_.

    Parameters
    ----------
    i : int | dict | scalar
        Observation selector. Must be one of:

        - **int**: Positional index in flattened observation order across all observation
          dimensions.
        - **dict**: Label-based mapping ``{obs_dim: coord_value}`` for all observation
          dimensions. Uses ``.sel`` semantics.
        - **scalar label**: Only when there is exactly one observation dimension.
    data : DataTree or InferenceData
        Input data. It should contain the posterior and the ``log_likelihood`` group.
    var_name : str, optional
        The name of the variable in log_likelihood groups storing the pointwise log
        likelihood data to use for loo computation.
    reff : float, optional
        Relative MCMC efficiency, ``ess / n`` i.e. number of effective samples divided by the number
        of actual samples. Computed from trace by default.
    log_lik_fn : callable, optional
        Custom log-likelihood function for a single observation. The signature must be
        ``log_lik_fn(observed, data)`` where ``observed`` is the observed data and ``data``
        is the full :class:`~arviz_base.DataTree` or :class:`~arviz_base.InferenceData`.
        The function must return either a :class:`~xarray.DataArray` or a :class:`~numpy.ndarray`
        with dimensions ``("chain", "draw")`` containing the per-draw log-likelihood values for
        that single observation. Array outputs are automatically wrapped in a DataArray before
        further validation. When provided, ``loo_i`` uses this function instead of the
        ``log_likelihood`` group.
    log_weights : DataArray, optional
        Smoothed log weights. It must have the same shape as the log likelihood data.
        Defaults to None. If not provided, it will be computed using the PSIS-LOO method.
        Must be provided together with pareto_k or both must be None.
    pareto_k : DataArray, optional
        Pareto shape values. It must have the same shape as the log likelihood data.
        Defaults to None. If not provided, it will be computed using the PSIS-LOO method.
        Must be provided together with log_weights or both must be None.
    log_jacobian : DataArray, optional
        Log-Jacobian adjustment for variable transformations. Required when the model was fitted
        on transformed response data :math:`z = T(y)` but you want to compute ELPD on the
        original response scale :math:`y`. The value should be :math:`\log|\frac{dz}{dy}|`
        (the log absolute value of the derivative of the transformation). Must be a DataArray
        with dimensions matching the observation dimensions.

    Returns
    -------
    ELPDData
        Object with the following attributes:

        - **kind**: "loo"
        - **elpd**: expected log pointwise predictive density for observation :math:`i`
        - **se**: standard error (set to 0.0 as SE is undefined for a single observation)
        - **p**: effective number of parameters for observation :math:`i`
        - **n_samples**: number of samples
        - **n_data_points**: 1 (single observation)
        - **scale**: "log"
        - **warning**: True if the estimated shape parameter of Pareto distribution is greater
          than ``good_k``
        - **good_k**: For a sample size S, the threshold is computed as
          ``min(1 - 1/log10(S), 0.7)``
        - **elpd_i**: scalar value of the expected log pointwise predictive density for observation
          :math:`i`
        - **pareto_k**: scalar Pareto shape value for observation :math:`i`
        - **approx_posterior**: False (not used for standard LOO)
        - **log_weights**: Smoothed log weights for observation :math:`i`
        - **log_jacobian**: Log-Jacobian adjustment for variable transformations.

    Notes
    -----
    This function is useful for testing log-likelihood functions and getting detailed diagnostics
    for individual observations. It's particularly helpful when debugging PSIS-LOO-CV computations
    for large datasets using :func:`loo_subsample` with the PLPD approximation method, or when
    verifying log-likelihood implementations with :func:`loo_moment_match`.

    Since this computes PSIS-LOO-CV for a single observation, the standard error is set to 0.0 as
    variance cannot be computed from a single value.

    Examples
    --------
    Calculate LOO for a single observation using the school name:

    .. ipython::
       :okwarning:

       In [1]: from arviz_stats import loo_i
          ...: from arviz_base import load_arviz_data
          ...: import xarray as xr
          ...: data = load_arviz_data("centered_eight")
          ...: loo_i({"school": "Choate"}, data)

    If you prefer simple integer indexing across flattened observations, you can use the index:

    .. ipython::
       :okwarning:

       In [2]: loo_i(0, data)

    For multi-dimensional data, specify all observation dimensions. For example,
    with data that has two observation dimensions (y_dim_0 and y_dim_1), you can select by index:

    .. ipython::
       :okwarning:

       In [3]: import arviz_base as azb
          ...: import numpy as np
          ...: np.random.seed(0)
          ...: idata = azb.from_dict({
          ...:     "posterior": {"theta": np.random.randn(2, 100, 3, 4)},
          ...:     "log_likelihood": {"y": np.random.randn(2, 100, 3, 4)},
          ...:     "observed_data": {"y": np.random.randn(3, 4)},
          ...: })
          ...: loo_i({"y_dim_0": 1, "y_dim_1": 2}, idata)

    With a single observation dimension, you can pass a single label directly:

    .. ipython::
       :okwarning:

       In [4]: loo_i("Choate", data)

    We can also use a custom log-likelihood function:

    .. ipython::
       :okwarning:

       In [5]: from scipy import stats
          ...: sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])
          ...: sigma_da = xr.DataArray(sigma,
          ...:                         dims=["school"],
          ...:                         coords={"school": data.observed_data.school.values})
          ...: data['constant_data'] = (
          ...:     data['constant_data'].to_dataset().assign(sigma=sigma_da)
          ...: )
          ...:
          ...: def log_lik_fn(obs_da, data):
          ...:     theta = data.posterior["theta"]
          ...:     sigma = data.constant_data["sigma"]
          ...:     ll = stats.norm.logpdf(obs_da, loc=theta, scale=sigma)
          ...:     return xr.DataArray(ll, dims=theta.dims, coords=theta.coords)
          ...:
          ...: loo_i({"school": "Choate"}, data, log_lik_fn=log_lik_fn)

    See Also
    --------
    :func:`loo` : Compute LOO for all observations
    :func:`compare` : Compare models based on their ELPD.

    References
    ----------

    .. [1] Vehtari et al. *Practical Bayesian model evaluation using leave-one-out cross-validation
       and WAIC*. Statistics and Computing. 27(5) (2017) https://doi.org/10.1007/s11222-016-9696-4
       arXiv preprint https://arxiv.org/abs/1507.04544.

    .. [2] Vehtari et al. *Pareto Smoothed Importance Sampling*.
       Journal of Machine Learning Research, 25(72) (2024) https://jmlr.org/papers/v25/19-556.html
       arXiv preprint https://arxiv.org/abs/1507.02646
    """
    log_lik_i, sample_dims, obs_dims, n_samples = _log_lik_i(i, data, var_name, log_lik_fn)

    log_weights_i, pareto_k_i = _get_weights_and_k_i(
        log_weights, pareto_k, i, obs_dims, sample_dims, data, n_samples, reff, log_lik_i, var_name
    )

    log_weights_sum = log_weights_i + log_lik_i
    elpd_i = logsumexp(log_weights_sum, dims=sample_dims).item()
    lppd_i = logsumexp(log_lik_i, b=1 / n_samples, dims=sample_dims).item()

    jacobian_da = _check_log_jacobian(log_jacobian, obs_dims)
    if jacobian_da is not None:
        jacobian_i = _get_log_likelihood_i(jacobian_da, i, obs_dims)
        jac_value = jacobian_i.squeeze().item()
    else:
        jac_value = 0.0

    if jac_value:
        elpd_i += jac_value
        lppd_i += jac_value

    p_loo_i = lppd_i - elpd_i
    elpd_se = 0.0

    warn_mg, good_k = _warn_pareto_k(pareto_k_i, n_samples)

    return ELPDData(
        kind="loo",
        elpd=elpd_i,
        se=elpd_se,
        p=p_loo_i,
        n_samples=n_samples,
        n_data_points=1,
        scale="log",
        warning=warn_mg,
        good_k=good_k,
        elpd_i=elpd_i,
        pareto_k=pareto_k_i,
        approx_posterior=False,
        log_weights=log_weights_i,
    )
