"""Pareto-smoothed importance sampling LOO (PSIS-LOO-CV) related functions."""

import warnings

import numpy as np
from arviz_base import convert_to_datatree, rcParams
from xarray_einstats.stats import logsumexp

from arviz_stats.utils import ELPDData, get_log_likelihood_dataset


def loo(data, pointwise=None, var_name=None, reff=None):
    """Compute Pareto-smoothed importance sampling leave-one-out cross-validation (PSIS-LOO-CV).

    Estimates the expected log pointwise predictive density (elpd) using Pareto-smoothed
    importance sampling leave-one-out cross-validation (PSIS-LOO-CV). Also calculates LOO's
    standard error and the effective number of parameters. The method is described in [1]_
    and [2]_.

    Parameters
    ----------
    data: obj
        Any object that can be converted to an :class:`arviz.InferenceData` object.
        Refer to documentation of
        :func:`arviz.convert_to_dataset` for details.
    pointwise: bool, optional
        If True the pointwise predictive accuracy will be returned. Defaults to
        ``rcParams["stats.ic_pointwise"]``.
    var_name : str, optional
        The name of the variable in log_likelihood groups storing the pointwise log
        likelihood data to use for loo computation.
    reff: float, optional
        Relative MCMC efficiency, ``ess / n`` i.e. number of effective samples divided by the number
        of actual samples. Computed from trace by default.

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
        - **elp_i**: :class:`~xarray.DataArray` with the pointwise predictive accuracy, only if
          ``pointwise=True``
        - **pareto_k**: array of Pareto shape values, only if ``pointwise=True``
        - **good_k**: For a sample size S, the threshold is computed as
          ``min(1 - 1/log10(S), 0.7)``

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

    References
    ----------

    .. [1] Vehtari et al. *Practical Bayesian model evaluation using leave-one-out cross-validation
        and WAIC*. Statistics and Computing. 27(5) (2017) https://doi.org/10.1007/s11222-016-9696-4
        arXiv preprint https://arxiv.org/abs/1507.04544.

    .. [2] Vehtari et al. *Pareto Smoothed Importance Sampling*.
        Journal of Machine Learning Research, 25(72) (2024) https://jmlr.org/papers/v25/19-556.html
        arXiv preprint https://arxiv.org/abs/1507.02646
    """
    data = convert_to_datatree(data)

    log_likelihood = get_log_likelihood_dataset(data, var_names=var_name)
    var_name = list(log_likelihood.data_vars.keys())[0]
    pointwise = rcParams["stats.ic_pointwise"] if pointwise is None else pointwise
    sample_dims = ["chain", "draw"]

    n_samples = log_likelihood.chain.size * log_likelihood.draw.size
    n_data_points = np.prod(
        [log_likelihood[dim].size for dim in log_likelihood.dims if dim not in sample_dims]
    )

    if reff is None:
        if not hasattr(data, "posterior"):
            raise TypeError("Must be able to extract a posterior group from data.")
        posterior = data.posterior
        n_chains = len(posterior.chain)
        if n_chains == 1:
            reff = 1.0
        else:
            ess_p = posterior.azstats.ess(method="mean")
            # this mean is over all data variables
            reff = (
                np.hstack([ess_p[v].values.flatten() for v in ess_p.data_vars]).mean() / n_samples
            )

    log_weights, pareto_k = log_likelihood.azstats.psislw(r_eff=reff, dims=sample_dims)
    pareto_k_da = pareto_k[var_name]
    log_weights += log_likelihood

    warn_mg = False
    good_k = min(1 - 1 / np.log10(n_samples), 0.7)

    if np.any(pareto_k_da > good_k):
        warnings.warn(
            f"Estimated shape parameter of Pareto distribution is greater than {good_k:.2f} "
            "for one or more samples. You should consider using a more robust model, this is "
            "because importance sampling is less likely to work well if the marginal posterior "
            "and LOO posterior are very different. This is more likely to happen with a "
            "non-robust model and highly influential observations."
        )
        warn_mg = True

    loo_lppd_i = logsumexp(log_weights, dims=sample_dims)[var_name].values
    lppd = logsumexp(log_likelihood, b=1 / n_samples, dims=sample_dims).sum()[var_name].values
    loo_lppd = loo_lppd_i.sum()
    loo_lppd_se = (n_data_points * np.var(loo_lppd_i)) ** 0.5
    p_loo = lppd - loo_lppd

    if not pointwise:
        # return good_k as well
        return ELPDData(
            "loo", loo_lppd, loo_lppd_se, p_loo, n_samples, n_data_points, "log", warn_mg, good_k
        )

    if np.equal(loo_lppd, loo_lppd_i).all():  # pylint: disable=no-member
        warnings.warn(
            "The point-wise LOO is the same with the sum LOO, please double check "
            "the Observed RV in your model to make sure it returns element-wise logp."
        )

    return ELPDData(
        "loo",
        loo_lppd,
        loo_lppd_se,
        p_loo,
        n_samples,
        n_data_points,
        "log",
        warn_mg,
        good_k,
        loo_lppd_i,
        pareto_k_da,
    )
