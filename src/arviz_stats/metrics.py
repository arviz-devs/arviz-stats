"""Collection of metrics for evaluating the performance of probabilistic models."""

from collections import namedtuple

import numpy as np
from arviz_base import convert_to_datatree, dataset_to_dataframe, extract, rcParams
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance, wasserstein_distance_nd

from arviz_stats.base import array_stats


def r2_score(
    data,
    var_name=None,
    data_pair=None,
    summary=True,
    point_estimate=None,
    ci_kind=None,
    ci_prob=None,
    round_to=2,
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
    round_to : int
        Number of decimals used to round results. Defaults to 2. Use "none" to return raw numbers.

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
            estimate = round(estimate, round_to)
            c_i = (round(c_i[0].item(), round_to), round(c_i[1].item(), round_to))

        return r2_summary(estimate, c_i[0], c_i[1])

    return r_squared


def kl_divergence(
    data1,
    data2,
    group="posterior",
    var_names=None,
    sample_dims=None,
    num_samples=500,
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
    data1 : DataArray, Dataset, DataTree, or InferenceData
    data2 : DataArray, Dataset, DataTree, or InferenceData
    group : hashable, default "posterior"
        Group on which to compute the kl-divergence.
    var_names : str or list of str, optional
        Names of the variables for which the KL-divergence should be computed.
    sample_dims : iterable of hashable, optional
        Dimensions to be considered sample dimensions and are to be reduced.
        Default ``rcParams["data.sample_dims"]``.
    num_samples : int
        Number of samples to use for the distance calculation. Default is 500.
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

    return _kld(dist1, dist2)


def wasserstein(
    data1,
    data2,
    group="posterior",
    var_names=None,
    sample_dims=None,
    joint=True,
    num_samples=500,
    random_seed=212480,
):
    """Compute the Wasserstein-1 distance.

    The Wasserstein distance, also called the Earth mover’s distance or the optimal transport
    distance, is a similarity metric between two probability distributions [1]_.

    Parameters
    ----------
    data1 : DataArray, Dataset, DataTree, or InferenceData
    data2 : DataArray, Dataset, DataTree, or InferenceData
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
        The computation is faster for the marginals, but it is equivalent to assume the
        marginals are independent, which usually is not the case.
    num_samples : int
        Number of samples to use for the distance calculation. Default is 500.
    random_seed : int
        Random seed for reproducibility. Use None for no seed.

    Returns
    -------
    wasserstein_distance : float


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

    dist1 = dataset_to_dataframe(dist1, sample_dims=["sample"]).values
    dist2 = dataset_to_dataframe(dist2, sample_dims=["sample"]).values

    return dist1, dist2


def _extract_and_reindex(data, group, var_names, sample_dims, num_samples, random_seed):
    return (
        extract(
            data,
            group=group,
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
    ary0 : array-like
        First sample of shape (n_posterior_samples, n_outputs).
    ary1 : array-like
        Second sample of shape (n_posterior_samples, n_outputs).

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
