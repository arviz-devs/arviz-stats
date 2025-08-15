"""Pareto-smoothed importance sampling LOO (PSIS-LOO-CV)."""

from arviz_base import rcParams

from arviz_stats.loo.helper_loo import (
    _compute_loo_results,
    _get_r_eff,
    _prepare_loo_inputs,
)


def loo(
    data,
    pointwise=None,
    var_name=None,
    reff=None,
    log_weights=None,
    pareto_k=None,
    log_jacobian=None,
):
    r"""Compute Pareto-smoothed importance sampling leave-one-out cross-validation (PSIS-LOO-CV).

    Estimates the expected log pointwise predictive density (elpd) using Pareto-smoothed
    importance sampling leave-one-out cross-validation (PSIS-LOO-CV). Also calculates LOO's
    standard error and the effective number of parameters. The method is described in [1]_
    and [2]_.

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

    Returns
    -------
    ELPDData
        Object with the following attributes:

        - **elpd**: expected log pointwise predictive density
        - **se**: standard error of the elpd
        - **p**: effective number of parameters
        - **n_samples**: number of samples
        - **n_data_points**: number of data points
        - **warning**: True if the estimated shape parameter of Pareto distribution is greater
          than ``good_k``.
        - **elpd_i**: :class:`~xarray.DataArray` with the pointwise predictive accuracy, only if
          ``pointwise=True``
        - **pareto_k**: array of Pareto shape values, only if ``pointwise=True``
        - **good_k**: For a sample size S, the threshold is computed as
          ``min(1 - 1/log10(S), 0.7)``
        - **approx_posterior**: True if approximate posterior was used.
        - **log_weights**: Smoothed log weights.

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

    .. [1] Vehtari et al. *Practical Bayesian model evaluation using leave-one-out cross-validation
       and WAIC*. Statistics and Computing. 27(5) (2017) https://doi.org/10.1007/s11222-016-9696-4
       arXiv preprint https://arxiv.org/abs/1507.04544.

    .. [2] Vehtari et al. *Pareto Smoothed Importance Sampling*.
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

    if log_weights is None and pareto_k is None:
        log_weights, pareto_k = loo_inputs.log_likelihood.azstats.psislw(
            r_eff=reff, dim=loo_inputs.sample_dims
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
        log_jacobian=log_jacobian,
    )
