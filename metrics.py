"""Collection of metrics for evaluating the performance of probabilistic models."""

from arviz_base import convert_to_datatree, dataset_to_dataframe, extract, rcParams
from scipy.stats import wasserstein_distance, wasserstein_distance_nd


def wasserstein(
    data_1,
    data_2,
    group="posterior",
    var_names=None,
    sample_dims=None,
    joint=True,
    num_samples=500,
    random_seed=212480,
):
    """
    Compute the Wasserstein-1 distance.

    Parameters
    ----------
    data : Dataset, DataTree, or InferenceData
    group : hashable, default "posterior"
        Group on which to compute the ESS.
    var_names : str or list of str, optional
        Names of the variables for which the ess should be computed.
    sample_dims : iterable of hashable, optional
        Dimensions to be considered sample dimensions and are to be reduced.
        Default ``rcParams["data.sample_dims"]``.
    joint : bool, default True
        If True, compute the joint Wasserstein distance. If False, compute the
        Wasserstein distance for each marginal. The computation is faster when `joint=False`
        but it is equivalent to assume that the marginals are independent, which usually
        is not the case.
    num_samples : int
        Number of samples to use for the Wasserstein distance calculation. Default is 500.
    random_seed : int
        Random seed for reproducibility. Use None for no seed.

    Returns
    -------
    wasserstein_distance : float
    """
    data_1 = convert_to_datatree(data_1)
    data_2 = convert_to_datatree(data_2)
    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]

    dist1 = _extract_and_reindex(
        data_1,
        group=group,
        var_names=var_names,
        sample_dims=sample_dims,
        num_samples=num_samples,
        random_seed=random_seed,
    )

    dist2 = _extract_and_reindex(
        data_2,
        group=group,
        var_names=var_names,
        sample_dims=sample_dims,
        num_samples=num_samples,
        random_seed=random_seed,
    )

    shared_var_names = set(dist1.data_vars) & set(dist2.data_vars)
    if not shared_var_names:
        raise ValueError("No shared variable names found.")

    if var_names is None:
        var_names = list(shared_var_names)
        dist1, dist2 = dist1[var_names], dist2[var_names]

    dist1 = dataset_to_dataframe(dist1, sample_dims=["sample"]).values
    dist2 = dataset_to_dataframe(dist2, sample_dims=["sample"]).values

    if joint:
        distance = wasserstein_distance_nd(dist1, dist2)

    else:
        distance = 0
        for var1, var2 in zip(dist1.T, dist2.T):
            distance += wasserstein_distance(var1, var2)

    return distance


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
