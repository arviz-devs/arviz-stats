"""Compute leave one out (PSIS-LOO) probability integral transform (PIT) values."""

import xarray as xr
from arviz_base import convert_to_datatree, extract

from arviz_stats.loo.helper_loo import _get_r_eff
from arviz_stats.utils import get_log_likelihood_dataset


def loo_pit(
    data,
    var_names=None,
    log_weights=None,
    pareto_k=None,
):
    r"""Compute leave one out (PSIS-LOO) probability integral transform (PIT) values.

    The LOO-PIT values are :math:`p(\tilde{y}_i \le y_i \mid y_{-i})`, where :math:`y_i`
    represents the observed data for index :math:`i` and :math:`\tilde y_i` represents the
    posterior predictive sample at index :math:`i`. Note that :math:`y_{-i}` indicates we have
    left out the :math:`i`-th observation. LOO-PIT values are computed using the PSIS-LOO-CV
    method described in [1]_ and [2]_.

    Parameters
    ----------
    data : DataTree or InferenceData
        It should contain posterior, posterior_predictive and log_likelihood groups.
    var_names : str or list of str, optional
        Names of the variables to be used to compute the LOO-PIT values. If None, all
        variables are used. The function assumes that the observed and log_likelihood
        variables share the same names.
    log_weights : Dataset, optional
        Pre-computed smoothed log weights from PSIS. Must be a Dataset with variables
        matching var_names. Must be provided together with pareto_k.
    pareto_k : Dataset, optional
        Pre-computed Pareto k-hat diagnostic values. Must be a Dataset with variables
        matching var_names. Must be provided together with log_weights.

    Returns
    -------
    loo_pit: array or DataArray
        Value of the LOO-PIT at each observed data point.

    Examples
    --------
    Calculate LOO-PIT values using as test quantity the observed values themselves.

    .. ipython::

        In [1]: from arviz_stats import loo_pit
           ...: from arviz_base import load_arviz_data
           ...: dt = load_arviz_data("centered_eight")
           ...: loo_pit(dt)

    Calculate LOO-PIT values using as test quantity the square of the difference between
    each observation and `mu`. For this we create a new DataTree, copying the posterior and
    log_likelihood groups and creating new observed and posterior_predictive groups.

    .. ipython::

        In [1]: from arviz_base import from_dict
           ...: new_dt = from_dict({"posterior": dt.posterior,
           ...:                 "log_likelihood": dt.log_likelihood,
           ...:                 "observed_data": {
           ...:                     "obs": (dt.observed_data.obs
           ...:                            - dt.posterior.mu.median(dim=("chain", "draw")))**2},
           ...:                 "posterior_predictive": {
           ...:                     "obs": (dt.posterior_predictive.obs - dt.posterior.mu)**2}})
           ...: loo_pit(new_dt)

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

    if var_names is None:
        var_names = list(data.observed_data.data_vars.keys())
    elif isinstance(var_names, str):
        var_names = [var_names]

    log_likelihood = get_log_likelihood_dataset(data, var_names=var_names)
    n_samples = log_likelihood.chain.size * log_likelihood.draw.size
    r_eff = _get_r_eff(data, n_samples)

    posterior_predictive = extract(
        data,
        group="posterior_predictive",
        combined=False,
        var_names=var_names,
        keep_dataset=True,
    )
    observed_data = extract(
        data,
        group="observed_data",
        combined=False,
        var_names=var_names,
        keep_dataset=True,
    )

    sample_dims = ["chain", "draw"]
    loo_pit_values = xr.Dataset(coords=observed_data.coords)

    for var in var_names:
        pred = posterior_predictive[var]
        obs = observed_data[var]

        if log_weights is not None and pareto_k is not None:
            pit_values, _ = pred.azstats.loo_pit(
                y_obs=obs,
                log_weights=log_weights[var],
                pareto_k=pareto_k[var],
                r_eff=r_eff,
                sample_dims=sample_dims,
            )
        else:
            log_ratios = -log_likelihood[var]
            pit_values, _ = pred.azstats.loo_pit(
                y_obs=obs,
                log_ratios=log_ratios,
                r_eff=r_eff,
                sample_dims=sample_dims,
            )
        loo_pit_values[var] = pit_values

    return loo_pit_values
