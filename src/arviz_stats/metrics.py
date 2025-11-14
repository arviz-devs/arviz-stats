"""Collection of metrics for evaluating the performance of probabilistic models."""

from collections import namedtuple

import numpy as np
from arviz_base import convert_to_datatree, dataset_to_dataarray, extract, rcParams
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance, wasserstein_distance_nd

from arviz_stats.base import array_stats
from arviz_stats.utils import round_num


def bayesian_r2(
    data,
    pred_mean=None,
    scale=None,
    scale_kind="sd",
    summary=True,
    point_estimate=None,
    ci_kind=None,
    ci_prob=None,
    circular=False,
    round_to="2g",
):
    r"""Bayesian :math:`R^2` for regression models.

    The :math:`R^2`, or coefficient of determination, is defined as the proportion of variance
    in the data that is explained by the model.

    The Bayesian :math:`R^2` (or modeled :math:`R^2`) differs from other definitions of
    :math:`R^2` in that it is computed only using posterior quantities from the fitted model.
    For details of the Bayesian :math:`R^2` see [1]_.

    Briefly, it is defined as:

    .. math::

        R^2 = \frac{\mathrm{Var}_{\mu}}{\mathrm{Var}_{\mu} + \mathrm{Var}_{\mathrm{res}}}

    where :math:`\mathrm{Var}_{\mu}` is the variance of the predicted means,
    and :math:`\mathrm{Var}_{\mathrm{res}}` is the modelled residual variance.

    For a Gaussian family, this is :math:`\\sigma^2`.
    For a Bernoulli family, this is :math:`p(1-p)`, where :math:`p` is the predicted
    probability of success (see [2]_ for details). This is computed internally if
    `scale` is not provided.

    For other models, you may need to compute the appropriate scale variable
    representing the modeled variance (or pseudo-variance) and pass it using
    the ``scale`` argument.

    Parameters
    ----------
    data : DataTree or InferenceData
        Input data. It should contain the posterior and posterior_predictive groups.
    pred_mean : str
        Name of the variable representing the predicted mean.
    scale : str, optional
        Name of the variable representing the modeled variance (or pseudo-variance).
        It can be omitted for binary classification problems, in which case the pseudo-variance
        is computed internally.
    scale_kind : str
        Whether the variable referenced by `scale` is a standard deviation ("sd")
        or variance ("var"). Defaults to "sd".
        If "sd", it is squared internally to obtain the variance. Omitted if `scale` is None.
    summary: bool
        Whether to return a summary (default) or an array of :math:`R^2` samples.
        The summary is a named tuple with a point estimate and a credible interval
    point_estimate: str
        The point estimate to compute. If None, the default value is used.
        Defaults values are defined in rcParams["stats.point_estimate"]. Ignored if
        summary is False.
    ci_kind: str
        The kind of credible interval to compute. If None, the default value is used.
        Defaults values are defined in rcParams["stats.ci_kind"]. Ignored if
        summary is False.
    ci_prob: float
        The probability for the credible interval. If None, the default value is used.
        Defaults values are defined in rcParams["stats.ci_prob"]. Ignored if
        summary is False.
    circular: bool
        Whether to compute the Bayesian :math:`R^2` for circular data. Defaults to False.
        It's assumed that the circular data is in radians and ranges from -π to π.
        We use the same definition of :math:`R^2` for circular data as in the linear case,
        but using the linearized circular variance. The linearized circular variance is defined as
        :math:`-2 \log(1 - V_c)`, where :math:`V_c` is the circular variance. Thus, this
        function assumes that the `scale` variable is already in linearized form.
    round_to: int or str, optional
        If integer, number of decimal places to round the result. If string of the
        form '2g' number of significant digits to round the result. Defaults to '2g'.
        Use None to return raw numbers.

    Returns
    -------
    Namedtuple or array

    See Also
    --------
    arviz_stats.residual_r2 : Residual :math:`R^2`.
    arviz_stats.loo_r2 : LOO-adjusted :math:`R^2`.

    Examples
    --------
    Calculate Bayesian :math:`R^2` for logistic regression:

    .. ipython::

        In [1]: from arviz_stats import bayesian_r2
           ...: from arviz_base import load_arviz_data
           ...: data = load_arviz_data('anes')
           ...: bayesian_r2(data, pred_mean="p")

    Calculate Bayesian :math:`R^2` for circular regression. The posterior has
    the concentration parameter ``kappa`` (from the VonMises distribution).
    Instead of the typical circular variance, which ranges from 0 to 1, we use the
    linearized variance, which ranges from 0 to ∞.

    .. ipython::

        In [1]: from scipy.special import i0, i1
           ...: data = load_arviz_data('periwinkles')
           ...: kappa = data.posterior['kappa']
           ...: data.posterior["variance"] = -2 * np.log(1 - (1 - i1(kappa) / i0(kappa)))
           ...: bayesian_r2(data, pred_mean='mu', scale='variance',
           ...:             scale_kind="var", circular=True)

    References
    ----------

    .. [1] Gelman et al. *R-squared for Bayesian regression models*.
        The American Statistician. 73(3) (2019). https://doi.org/10.1080/00031305.2018.1549100
        preprint http://www.stat.columbia.edu/~gelman/research/published/bayes_R2_v3.pdf.
    .. [2] Tjur, T. *Coefficient of determination in logistic regression models-A new proposal:
        The coefficient of discrimination* The American Statistician, 63(4) (2009).
        https://doi.org/10.1198/tast.2009.08210

    """
    if point_estimate is None:
        point_estimate = rcParams["stats.point_estimate"]
    if ci_kind is None:
        ci_kind = rcParams["stats.ci_kind"]
    if ci_prob is None:
        ci_prob = rcParams["stats.ci_prob"]

    mu_pred = extract(data, group="posterior", var_names=pred_mean).values.T
    if scale is not None:
        scale = extract(data, group="posterior", var_names=scale).values.T

    r_squared = array_stats.bayesian_r2(mu_pred, scale, scale_kind, circular)

    if summary:
        return _summary_r2(r_squared, point_estimate, ci_kind, ci_prob, round_to)

    return r_squared


def residual_r2(
    data,
    pred_mean=None,
    obs_name=None,
    summary=True,
    point_estimate=None,
    ci_kind=None,
    ci_prob=None,
    circular=False,
    round_to="2g",
):
    r"""Residual :math:`R^2` for Bayesian regression models.

    The :math:`R^2`, or coefficient of determination, is defined as the proportion of variance
    in the data that is explained by the model. For details of the residual :math:`R^2` see [1]_.

    Briefly, it is defined as:

    .. math::

        R^2 = \frac{\mathrm{Var}_{\mu}}{\mathrm{Var}_{\mu} + \mathrm{Var}_{\mathrm{res}}}

    where :math:`\mathrm{Var}_{\mu}` is the variance of the predicted means,
    and :math:`\mathrm{Var}_{\mathrm{res}}` is the residual variance.

    .. math::

        \mathrm{Var}_{\mathrm{res}}^s = V_{n=1}^N \hat{e}_n^s,

    where :math:`\hat{e}_n^s=y_n-\hat{y}_n^s` are the residuals for observation :math:`n` in
    posterior sample :math:`s`.

    The residual :math:`R^2` differs from the Bayesian :math:`R^2` in that it computes
    residual variance from the observed data, while for the Bayesian :math:`R^2` all
    variance terms come from the model, and not directly from the data.


    Parameters
    ----------
    data : DataTree or InferenceData
        Input data. It should contain the posterior_predictive and observed_data groups.
    pred_name : str
        Name of the variable representing the predicted mean.
    obs_name : str, optional
        Name of the variable representing the observed data.
    summary: bool
        Whether to return a summary (default) or an array of :math:`R^2` samples.
        The summary is a named tuple with a point estimate and a credible interval
    point_estimate: str
        The point estimate to compute. If None, the default value is used.
        Defaults values are defined in rcParams["stats.point_estimate"]. Ignored if
        summary is False.
    ci_kind: str
        The kind of credible interval to compute. If None, the default value is used.
        Defaults values are defined in rcParams["stats.ci_kind"]. Ignored if
        summary is False.
    ci_prob: float
        The probability for the credible interval. If None, the default value is used.
        Defaults values are defined in rcParams["stats.ci_prob"]. Ignored if
        summary is False.
    circular: bool
        Whether to compute the residual :math:`R^2` for circular data. Defaults to False.
        It's assumed that the circular data is in radians and ranges from -π to π.
        We use the same definition of :math:`R^2` for circular data as in the linear case,
        but using the linearized circular variance. The linearized circular variance is defined as
        :math:`-2 \log(1 - V_c)`, where :math:`V_c` is the circular variance.
    round_to: int or str, optional
        If integer, number of decimal places to round the result. If string of the
        form '2g' number of significant digits to round the result. Defaults to '2g'.
        Use None to return raw numbers.

    Returns
    -------
    Namedtuple or array


    See Also
    --------
    arviz_stats.bayesian_r2 : Bayesian :math:`R^2`.
    arviz_stats.loo_r2 : LOO-adjusted :math:`R^2`.

    Examples
    --------
    Calculate residual :math:`R^2` for Bayesian logistic regression:

    .. ipython::

        In [1]: from arviz_stats import residual_r2
           ...: from arviz_base import load_arviz_data
           ...: data = load_arviz_data('anes')
           ...: residual_r2(data, pred_mean='p', obs_name='vote')

    Calculate residual :math:`R^2` for Bayesian circular regression:

    .. ipython::

        In [1]: data = load_arviz_data('periwinkles')
           ...: residual_r2(data, pred_mean='mu', obs_name='direction', circular=True)


    References
    ----------

    .. [1] Gelman et al. *R-squared for Bayesian regression models*.
        The American Statistician. 73(3) (2019). https://doi.org/10.1080/00031305.2018.1549100
        preprint http://www.stat.columbia.edu/~gelman/research/published/bayes_R2_v3.pdf.
    """
    if point_estimate is None:
        point_estimate = rcParams["stats.point_estimate"]
    if ci_kind is None:
        ci_kind = rcParams["stats.ci_kind"]
    if ci_prob is None:
        ci_prob = rcParams["stats.ci_prob"]

    y_true = extract(data, group="observed_data", var_names=obs_name, combined=False).values
    mu_pred = extract(data, group="posterior", var_names=pred_mean).values.T

    r_squared = array_stats.residual_r2(y_true, mu_pred, circular)

    if summary:
        return _summary_r2(r_squared, point_estimate, ci_kind, ci_prob, round_to)

    return r_squared


def metrics(data, kind="rmse", var_name=None, sample_dims=None, round_to="2g"):
    """
    Compute performace metrics.

    Currently supported metrics are mean absolute error, mean squared error and
    root mean squared error.
    For classification problems, accuracy and balanced accuracy are also supported.

    Parameters
    ----------
    data: DataTree or InferenceData
        It should contain groups `observed_data` and `posterior_predictive`.
    kind: str
        The kind of metric to compute. Available options are:

        - 'mae': mean absolute error.
        - 'mse': mean squared error.
        - 'rmse': root mean squared error. Default.
        - 'acc': classification accuracy.
        - 'acc_balanced': balanced classification accuracy.

    var_name: str, optional
        The name of the observed and predicted variable.
    sample_dims: iterable of hashable, optional
        Dimensions to be considered sample dimensions and are to be reduced.
        Default ``rcParams["data.sample_dims"]``.
    round_to: int or str, optional
        If integer, number of decimal places to round the result. If string of the
        form '2g' number of significant digits to round the result. Defaults to '2g'.
        Use None to return raw numbers.

    Returns
    -------
    estimate: namedtuple
        A namedtuple with the mean of the computed metric and its standard error.

    Examples
    --------
    Calculate root mean squared error

    .. ipython::

        In [1]: from arviz_stats import metrics
           ...: from arviz_base import load_arviz_data
           ...: dt = load_arviz_data("radon")
           ...: metrics(dt, kind="rmse")

    Calculate accuracy of a logistic regression model

    .. ipython::

        In [1]: dt = load_arviz_data("anes")
           ...: metrics(dt, kind="acc")

    Notes
    -----
    The computation of the metrics is done by first reducing the posterior predictive
    samples, this is done to mirror the computation of the metrics by the
    :func:`arviz_stats.loo_metrics` function, and hence make comparison easier to perform.
    """
    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]

    if var_name is None:
        var_name = list(data.observed_data.data_vars.keys())[0]

    observed = data.observed_data[var_name]
    predicted = data.posterior_predictive[var_name].mean(dim=sample_dims)

    return _metrics(observed, predicted, kind, round_to)


def kl_divergence(
    data1,
    data2,
    group="posterior",
    var_names=None,
    sample_dims=None,
    num_samples=500,
    round_to="2g",
    random_seed=212480,
):
    """Compute the Kullback-Leibler (KL) divergence.

    The KL-divergence is a measure of how different two probability distributions are.
    It represents how much extra uncertainty are we introducing when we use one
    distribution to approximate another. The KL-divergence is not symmetric, thus
    changing the order of the `data1` and `data2` arguments will change the result.

    For details of the approximation used to the compute the KL-divergence see [1]_.

    Parameters
    ----------
    data1, data2 : DataArray, Dataset, DataTree, or InferenceData
    group : hashable, default "posterior"
        Group on which to compute the kl-divergence.
    var_names : str or list of str, optional
        Names of the variables for which the KL-divergence should be computed.
    sample_dims : iterable of hashable, optional
        Dimensions to be considered sample dimensions and are to be reduced.
        Default ``rcParams["data.sample_dims"]``.
    num_samples : int
        Number of samples to use for the distance calculation. Default is 500.
    round_to: int or str, optional
        If integer, number of decimal places to round the result. If string of the
        form '2g' number of significant digits to round the result. Defaults to '2g'.
        Use None to return raw numbers.
    random_seed : int
        Random seed for reproducibility. Use None for no seed.

    Returns
    -------
    KL-divergence : float


    Examples
    --------
    Calculate the KL-divergence between the posterior distributions
    for the variable mu in the centered and non-centered eight schools models

    .. ipython::

        In [1]: from arviz_stats import kl_divergence
           ...: from arviz_base import load_arviz_data
           ...: data1 = load_arviz_data('centered_eight')
           ...: data2 = load_arviz_data('non_centered_eight')
           ...: kl_divergence(data1, data2, var_names="mu")

    References
    ----------
    .. [1] F. Perez-Cruz, *Kullback-Leibler divergence estimation of continuous distributions*
        IEEE International Symposium on Information Theory. (2008)
        https://doi.org/10.1109/ISIT.2008.4595271.
        preprint https://www.tsc.uc3m.es/~fernando/bare_conf3.pdf
    """
    dist1, dist2 = _prepare_distribution_pair(
        data1,
        data2,
        group=group,
        var_names=var_names,
        sample_dims=sample_dims,
        num_samples=num_samples,
        random_seed=random_seed,
    )

    kl_d = _kld(dist1, dist2)

    if round_to is not None and round_to not in ("None", "none"):
        kl_d = round_num(kl_d, round_to)

    return kl_d


def wasserstein(
    data1,
    data2,
    group="posterior",
    var_names=None,
    sample_dims=None,
    joint=True,
    num_samples=500,
    round_to="2g",
    random_seed=212480,
):
    """Compute the Wasserstein-1 distance.

    The Wasserstein distance, also called the Earth mover’s distance or the optimal transport
    distance, is a similarity metric between two probability distributions [1]_.

    Parameters
    ----------
    data1, data2 : DataArray, Dataset, DataTree, or InferenceData
    group : hashable, default "posterior"
        Group on which to compute the Wasserstein distance.
    var_names : str or list of str, optional
        Names of the variables for which the Wasserstein distance should be computed.
    sample_dims : iterable of hashable, optional
        Dimensions to be considered sample dimensions and are to be reduced.
        Default ``rcParams["data.sample_dims"]``.
    joint : bool, default True
        Whether to compute Wasserstein distance for the joint distribution (True)
        or over the marginals (False)
    num_samples : int
        Number of samples to use for the distance calculation. Default is 500.
    round_to: int or str, optional
        If integer, number of decimal places to round the result. If string of the
        form '2g' number of significant digits to round the result. Defaults to '2g'.
        Use None to return raw numbers.
    random_seed : int
        Random seed for reproducibility. Use None for no seed.

    Returns
    -------
    wasserstein_distance : float

    Notes
    -----
    The computation is faster for the marginals (`joint=False`). This is equivalent to
    assume the marginals are independent, which usually is not the case.
    This function uses the :func:`scipy.stats.wasserstein_distance` for the computation of the
    marginals and :func:`scipy.stats.wasserstein_distance_nd` for the joint distribution.

    Examples
    --------
    Calculate the Wasserstein distance between the posterior distributions
    for the variable mu in the centered and non-centered eight schools models

    .. ipython::

        In [1]: from arviz_stats import wasserstein
           ...: from arviz_base import load_arviz_data
           ...: data1 = load_arviz_data('centered_eight')
           ...: data2 = load_arviz_data('non_centered_eight')
           ...: wasserstein(data1, data2, var_names="mu")

    References
    ----------

    .. [1] "Wasserstein metric",
           https://en.wikipedia.org/wiki/Wasserstein_metric
    """
    dist1, dist2 = _prepare_distribution_pair(
        data1,
        data2,
        group=group,
        var_names=var_names,
        sample_dims=sample_dims,
        num_samples=num_samples,
        random_seed=random_seed,
    )

    if joint:
        distance = wasserstein_distance_nd(dist1, dist2)

    else:
        distance = 0
        for var1, var2 in zip(dist1.T, dist2.T):
            distance += wasserstein_distance(var1, var2)
        distance = distance.item()

    if round_to is not None and round_to not in ("None", "none"):
        distance = round_num(distance, round_to)

    return distance


def _prepare_distribution_pair(
    data1, data2, group, var_names, sample_dims, num_samples, random_seed
):
    """Prepare the distribution pair for metric calculations."""
    data1 = convert_to_datatree(data1)
    data2 = convert_to_datatree(data2)
    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]

    dist1 = _extract_and_reindex(
        data1,
        group=group,
        var_names=var_names,
        sample_dims=sample_dims,
        num_samples=num_samples,
        random_seed=random_seed,
    )

    dist2 = _extract_and_reindex(
        data2,
        group=group,
        var_names=var_names,
        sample_dims=sample_dims,
        num_samples=num_samples,
        random_seed=random_seed,
    )

    shared_var_names = set(dist1.data_vars) & set(dist2.data_vars)
    if not shared_var_names:
        raise ValueError(
            "No shared variable names found between the two datasets. "
            "Ensure that both datasets contain variables with matching names."
        )

    if var_names is None:
        var_names = list(shared_var_names)
        dist1, dist2 = dist1[var_names], dist2[var_names]

    dist1 = dataset_to_dataarray(dist1, sample_dims=["sample"])
    dist2 = dataset_to_dataarray(dist2, sample_dims=["sample"])

    return dist1, dist2


def _extract_and_reindex(data, group, var_names, sample_dims, num_samples, random_seed):
    return (
        extract(
            data,
            group=group,
            sample_dims=sample_dims,
            var_names=var_names,
            num_samples=num_samples,
            random_seed=random_seed,
            keep_dataset=True,
        )
        .reset_index("sample")
        .drop_vars(sample_dims)
        .assign_coords(sample=range(num_samples))
    )


def _kld(ary0, ary1):
    """Kullback-Leibler divergence approximation.

    Compute KL-divergence using equation 14 from [1]_. Assumes both arrays
    are of the same shape.

    Parameters
    ----------
    ary0, ary1 : (N, M) array-like
        Samples of the input distributions. ``N`` represents the number of samples (e.g. posterior
        samples) and ``M`` the number of outputs (e.g. number of variables in the posterior)

    Returns
    -------
    float
        The Kullback-Leibler divergence between the two
        distributions.

    References
    ----------
    .. [1] F. Perez-Cruz, *Kullback-Leibler divergence estimation of continuous distributions*
        IEEE International Symposium on Information Theory. (2008)
        https://doi.org/10.1109/ISIT.2008.4595271.
        preprint https://www.tsc.uc3m.es/~fernando/bare_conf3.pdf

    """
    # for discrete data we need to smooth the samples to avoid numerical errors
    # here we are adding a small noise to all samples, differences should be negligible
    # but we may want to do something more sophisticated in the future
    rng = np.random.default_rng(0)
    ary0 = ary0 + rng.normal(0, ary0.std(axis=0) / 1e6, size=ary0.shape)
    ary1 = ary1 + rng.normal(0, ary1.std(axis=0) / 1e6, size=ary1.shape)

    samples, dim = ary0.shape

    # Build KD-trees for X and Y
    kd_tree_ary0 = cKDTree(ary0)
    kd_tree_ary1 = cKDTree(ary1)

    # first nearest neighbour distances of X to Y
    r_k, _ = kd_tree_ary1.query(ary0)

    # second nearest neighbour distances of X to X
    # we skip the trivial first nearest neighbour distance
    s_k = kd_tree_ary0.query(ary0, k=2)[0][:, 1]

    kl_div = (dim / samples) * np.sum(np.log(r_k / s_k)) + np.log(samples / (samples - 1))
    # Due to numerical errors and for very similar samples we can get negative values
    kl_div = max(0.0, kl_div.item())

    return kl_div


def _metrics(observed, predicted, kind, round_to):
    """Compute performance metrics.

    Parameters
    ----------
    observed: DataArray
        Observed data.
    predicted: DataArray
        Predicted data.
    kind: str
        The kind of metric to compute. Available options are:

        - 'mae': mean absolute error.
        - 'mse': mean squared error.
        - 'rmse': root mean squared error. Default.
        - 'acc': classification accuracy.
        - 'acc_balanced': balanced classification accuracy.
    round_to: int or str, optional
        If integer, number of decimal places to round the result. If string of the
        form '2g' number of significant digits to round the result. Defaults to '2g'.
        Use None to return raw numbers.

    Returns
    -------
    estimate: namedtuple
        A namedtuple with the mean of the computed metric and its standard error.
    """
    valid_kind = ["mae", "rmse", "mse", "acc", "acc_balanced"]
    if kind not in valid_kind:
        raise ValueError(f"kind must be one of {valid_kind}")

    estimate = namedtuple(kind, ["mean", "se"])
    mean, std_error = array_stats.metrics(observed, predicted, kind=kind)

    return estimate(round_num(mean, round_to), round_num(std_error, round_to))


def _summary_r2(r_squared, point_estimate, ci_kind, ci_prob, round_to):
    estimate = getattr(np, point_estimate)(r_squared).item()
    c_i = getattr(array_stats, ci_kind)(r_squared, ci_prob)

    r2_summary = namedtuple("R2", [point_estimate, f"{ci_kind}_lb", f"{ci_kind}_ub"])
    if (round_to is not None) and (round_to not in ("None", "none")):
        estimate = round_num(estimate, round_to)
        c_i = (round_num(c_i[0].item(), round_to), round_num(c_i[1].item(), round_to))

    return r2_summary(estimate, c_i[0], c_i[1])
