"""Collection of metrics for evaluating the performance of probabilistic models."""

from collections import namedtuple

import numpy as np
from arviz_base import convert_to_datatree, dataset_to_dataarray, extract, rcParams
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance, wasserstein_distance_nd

from arviz_stats.base import array_stats
from arviz_stats.utils import round_num


def r2_score(
    data,
    var_name=None,
    data_pair=None,
    summary=True,
    point_estimate=None,
    ci_kind=None,
    ci_prob=None,
    round_to="2g",
):
    """R² for Bayesian regression models.

    The R², or coefficient of determination, is defined as the proportion of variance
    in the data that is explained by the model. It is computed as the variance of the
    predicted values divided by the variance of the predicted values plus the variance
    of the residuals. For details of the Bayesian R² see [1]_.

    Parameters
    ----------
    data : DataTree or InferenceData
        Input data. It should contain the posterior and the log_likelihood groups.
    var_name : str
        Name of the variable to compute the R² for.
    data_pair : dict
        Dictionary with the first element containing the posterior predictive name
        and the second element containing the observed data variable name.
    summary: bool
        Whether to return a summary (default) or an array of R² samples.
        The summary is a Pandas' series with a point estimate and a credible interval
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
    round_to: int or str, optional
        If integer, number of decimal places to round the result. If string of the
        form '2g' number of significant digits to round the result. Defaults to '2g'.
        Use None to return raw numbers.

    Returns
    -------
    Namedtuple or array

    Examples
    --------
    Calculate R² samples for Bayesian regression models :

    .. ipython::

        In [1]: from arviz_stats import r2_score
           ...: from arviz_base import load_arviz_data
           ...: data = load_arviz_data('regression1d')
           ...: r2_score(data)

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

    if data_pair is None:
        obs_var_name = var_name
        pred_var_name = var_name
    else:
        obs_var_name = list(data_pair.keys())[0]
        pred_var_name = list(data_pair.values())[0]

    y_true = extract(data, group="observed_data", var_names=obs_var_name, combined=False).values
    y_pred = extract(data, group="posterior_predictive", var_names=pred_var_name).values.T

    r_squared = array_stats.r2_score(y_true, y_pred)

    if summary:
        estimate = getattr(np, point_estimate)(r_squared).item()
        c_i = getattr(array_stats, ci_kind)(r_squared, ci_prob)

        r2_summary = namedtuple("R2", [point_estimate, f"{ci_kind}_lb", f"{ci_kind}_ub"])
        if (round_to is not None) and (round_to not in ("None", "none")):
            estimate = round_num(estimate, round_to)
            c_i = (round_num(c_i[0].item(), round_to), round_num(c_i[1].item(), round_to))

        return r2_summary(estimate, c_i[0], c_i[1])

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
    n_obs = len(observed)

    if kind == "mae":
        abs_e = np.abs(observed - predicted)
        mean = np.mean(abs_e)
        std_error = np.std(abs_e) / n_obs**0.5

    elif kind == "mse":
        sq_e = (observed - predicted) ** 2
        mean = np.mean(sq_e)
        std_error = np.std(sq_e) / n_obs**0.5

    elif kind == "rmse":
        sq_e = (observed - predicted) ** 2
        mean_mse = np.mean(sq_e)
        var_mse = np.var(sq_e) / n_obs
        var_rmse = var_mse / mean_mse / 4  # Comes from the first order Taylor approx.
        mean = mean_mse**0.5
        std_error = var_rmse**0.5

    elif kind == "acc":
        yhat = predicted > 0.5
        acc = yhat == observed
        mean = np.mean(acc)
        std_error = (mean * (1 - mean) / n_obs) ** 0.5

    else:
        yhat = predicted > 0.5
        mask = observed == 0
        true_neg = np.mean(yhat[mask] == observed[mask])
        true_pos = np.mean(yhat[~mask] == observed[~mask])
        mean = (true_pos + true_neg) / 2
        # This approximation has quite large bias for small samples
        bls_acc_var = (true_pos * (1 - true_pos) + true_neg * (1 - true_neg)) / 4
        std_error = bls_acc_var / n_obs**0.5

    return estimate(round_num(mean, round_to), round_num(std_error, round_to))
