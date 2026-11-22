"""Pointwise conditional log-likelihoods for PSIS-LOO-CV with non-factorized models."""

from arviz_base import convert_to_datatree

from arviz_stats.loo.loo_helper import _var_name_custom_ll


def log_lik_nonfactorized(data, mean, cov=None, prec=None, df=None, var_name=None, dims=None):
    r"""Compute pointwise conditional log-likelihoods for non-factorized models.

    PSIS-LOO-CV, described in [3]_ and [4]_, needs the log-likelihood of each observation, but a
    model whose observations follow a multivariate normal or a multivariate Student-t only
    provides the joint log-likelihood of all of them. This function computes the log-likelihood
    of each observation given all the others, :math:`\log p(y_i \mid y_{-i}, \theta)`, for every
    posterior draw. The result can be used in :func:`loo` and related functions in place of the
    pointwise log-likelihood of a factorized model. The method is described in [1]_ and [2]_.

    Parameters
    ----------
    data : DataTree or InferenceData
        Input data. It should contain the ``posterior`` and ``observed_data`` groups.
    mean : str or float
        Name of the variable in the ``posterior`` group with the mean of the observation model.
        A number can be passed instead when the mean is a known constant.
    cov : str, optional
        Name of the variable in the ``posterior`` group with the covariance matrix of the
        multivariate normal or the scale matrix of the multivariate Student-t. Must be provided
        if ``prec`` is None.
    prec : str, optional
        Name of the variable in the ``posterior`` group with the precision matrix, the inverse
        of the covariance or scale matrix. Must be provided if ``cov`` is None.
    df : str or float, optional
        Name of the variable in the ``posterior`` group with the degrees of freedom of the
        multivariate Student-t. A number can be passed instead when they are a known constant.
        Defaults to None, in which case the observation model is multivariate normal.
    var_name : str, optional
        The name of the variable in the ``observed_data`` group to use. If None, the only
        variable in the ``observed_data`` group is used (an error is raised if there are
        several).
    dims : sequence of hashable, optional
        The two dimensions of ``cov`` or ``prec`` that define the matrix. Defaults to its
        dimensions that are not sample dimensions.

    Returns
    -------
    DataArray
        Pointwise conditional log-likelihood with the sample dimensions followed by the
        dimensions of the observed variable.

    Notes
    -----
    For a multivariate normal with mean :math:`\mu` and covariance matrix :math:`\Sigma`

    .. math::

        \log p(y_i \mid y_{-i}, \theta) =
        -\frac{1}{2}\log(2\pi) + \frac{1}{2}\log\bar{c}_{ii} - \frac{g_i^2}{2\bar{c}_{ii}},

    where :math:`g = \Sigma^{-1}(y - \mu)` and :math:`\bar{c}_{ii} = [\Sigma^{-1}]_{ii}`.
    For a multivariate Student-t with :math:`\nu` degrees of freedom and :math:`N` observations,
    the conditional distribution of each observation is a Student-t with :math:`\nu + N - 1`
    degrees of freedom. Its location and scale are given in [1]_.

    Models that can be rewritten as a multivariate normal or Student-t are also covered.
    For example, the lagged simultaneous autoregressive model
    :math:`(I - \rho W) y \sim N(\eta, \sigma^2 I)` has mean :math:`(I - \rho W)^{-1} \eta`
    and precision matrix :math:`(I - \rho W)^T (I - \rho W) / \sigma^2`.

    Examples
    --------
    Simulate correlated observations with a common mean, a known correlation matrix and an
    unknown variance. With the prior :math:`p(\mu, \sigma^2) \propto 1 / \sigma^2` the posterior
    is known, so we can draw from it directly.

    .. ipython::

        In [1]: import numpy as np
           ...: from arviz_base import from_dict
           ...: from arviz_stats import loo, log_lik_nonfactorized
           ...:
           ...: rng = np.random.default_rng(0)
           ...: n_obs = 20
           ...: lag = np.abs(np.subtract.outer(np.arange(n_obs), np.arange(n_obs)))
           ...: corr = 0.6**lag
           ...: y = rng.multivariate_normal(np.ones(n_obs), corr)
           ...:
           ...: prec = np.linalg.inv(corr)
           ...: mu_hat = (prec @ y).sum() / prec.sum()
           ...: scale = (y - mu_hat) @ prec @ (y - mu_hat)
           ...: sigma2 = scale / rng.chisquare(n_obs - 1, size=(4, 500))
           ...: mu = rng.normal(mu_hat, np.sqrt(sigma2 / prec.sum()))

    Store the posterior draws of the mean and of the covariance matrix together with the
    observed data, and compute the pointwise conditional log-likelihood.

    .. ipython::

        In [2]: data = from_dict(
           ...:     {
           ...:         "posterior": {"mu": mu, "cov": sigma2[..., None, None] * corr},
           ...:         "observed_data": {"y": y},
           ...:     }
           ...: )
           ...: log_lik = log_lik_nonfactorized(data, mean="mu", cov="cov")
           ...: log_lik.dims

    Store it in the ``log_likelihood`` group and calculate LOO as usual.

    .. ipython::

        In [3]: data["log_likelihood"] = log_lik.to_dataset()
           ...: loo(data)

    See Also
    --------
    :func:`loo` : Compute the ELPD using PSIS-LOO-CV.
    :func:`loo_i` : Compute the ELPD for a single observation.

    References
    ----------

    .. [1] Bürkner et al. *Efficient leave-one-out cross-validation for Bayesian non-factorized
       normal and Student-t models*. Computational Statistics. 36(2) (2021)
       https://doi.org/10.1007/s00180-020-01045-4
       arXiv preprint https://arxiv.org/abs/1810.10559.

    .. [2] Vehtari et al. *Leave-one-out cross-validation for non-factorized models*.
       https://mc-stan.org/loo/articles/loo2-non-factorized.html

    .. [3] Vehtari et al. *Practical Bayesian model evaluation using leave-one-out cross-validation
       and WAIC*. Statistics and Computing. 27(5) (2017) https://doi.org/10.1007/s11222-016-9696-4
       arXiv preprint https://arxiv.org/abs/1507.04544.

    .. [4] Vehtari et al. *Pareto Smoothed Importance Sampling*.
       Journal of Machine Learning Research, 25(72) (2024) https://jmlr.org/papers/v25/19-556.html
       arXiv preprint https://arxiv.org/abs/1507.02646
    """
    data = convert_to_datatree(data)
    var_name = _var_name_custom_ll(data, var_name)
    mean, cov, prec, df = (
        data.posterior[param] if isinstance(param, str) else param
        for param in (mean, cov, prec, df)
    )

    return data.observed_data[var_name].azstats.log_lik_nonfactorized(
        mean, cov=cov, prec=prec, df=df, dims=dims
    )
