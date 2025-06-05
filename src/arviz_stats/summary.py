"""Summaries for various statistics and diagnostics."""

import numpy as np
import xarray as xr
from arviz_base import dataset_to_dataframe, extract, rcParams, references_to_dataset
from xarray_einstats import stats

from arviz_stats.validate import validate_dims

__all__ = ["summary", "ci_in_rope"]


def summary(
    data,
    var_names=None,
    filter_vars=None,
    group="posterior",
    coords=None,
    sample_dims=None,
    kind="all",
    ci_prob=None,
    ci_kind=None,
    round_to=2,
    skipna=False,
):
    """
    Create a data frame with summary statistics and or diagnostics.

    Parameters
    ----------
    data : DataTree, DataSet or InferenceData
    var_names : list of str, optional
        Names of variables to include in summary. If None all variables are included.
    filter_vars: {None, "like", "regex"}, default None
        Used for `var_names` only.
        If ``None`` (default), interpret var_names as the real variables names.
        If "like", interpret var_names as substrings of the real variables names.
        If "regex", interpret var_names as regular expressions on the real variables names.
    group: str
        Select a group for summary. Defaults to “posterior”.
    coords : dict, optional
        Coordinates defining a subset over the selected group.
    sample_dims : str or sequence of hashable, optional
        Defaults to ``rcParams["data.sample_dims"]``
    kind: {'all', 'stats', 'diagnostics', 'all_median', 'stats_median',
    'diagnostics_median', 'mc_diagnostics'}, default 'all'
        * ``all``: `mean`, `sd`, `ci`, `ess_bulk`, `ess_tail`, `r_hat`, `mcse_mean`, `mcse_sd`.
        * ``stats``: `mean`, `sd`, and `ci`.
        * ``diagnostics``: `ess_bulk`, `ess_tail`, `r_hat`, `mcse_mean`, `mcse_sd`.
        * ``all_median``: `median`, `mad`, `ci`, `ess_median`, `ess_tail`, `r_hat`, `mcse_median`.
        * ``stats_median``: `median`, `mad`, and `ci`.
        * ``diagnostics_median``: `ess_median`, `ess_tail`, `r_hat`, `mcse_median`.
        * ``mc_diagnostics``: `mcse_mean`, `ess_mean`, and `min_ss`.
    ci_prob : float, optional
        Probability for the credible interval. Defaults to ``rcParams["stats.ci_prob"]``.
    ci_kind : {"hdi", "eti"}, optional
        Type of credible interval. Defaults to ``rcParams["stats.ci_kind"]``.
        If `kind` is stats_median or all_median, `ci_kind` is forced to "eti".
    round_to : int
        Number of decimals used to round results. Defaults to 2. Use "none" to return raw numbers.
    skipna: bool
        If true ignores nan values when computing the summary statistics. Defaults to false.

    Returns
    -------
    pandas.DataFrame

    See Also
    --------
    arviz.rhat : Compute estimate of rank normalized split R-hat for a set of traces.
    arviz.ess : Calculate the effective sample size of a set of traces.
    arviz.mcse : Calculate Markov Chain Standard Error statistic.
    plot_ess : Plot quantile, local or evolution of effective sample sizes (ESS).
    plot_mcse : Plot quantile, local or evolution of Markov Chain Standard Error (MCSE).


    Examples
    --------
    .. ipython::

        In [1]: from arviz_base import load_arviz_data
           ...: from arviz_stats import summary
           ...: data = load_arviz_data("non_centered_eight")
           ...: summary(data, var_names=["mu", "tau"])

    You can use ``filter_vars`` to select variables without having to specify all the exact
    names. Use ``filter_vars="like"`` to select based on partial naming:

    .. ipython::

        In [1]: summary(data, var_names=["the"], filter_vars="like")

    Use ``filter_vars="regex"`` to select based on regular expressions, and prefix the variables
    you want to exclude by ``~``. Here, we exclude from the summary all the variables
    starting with the letter t:

    .. ipython::

        In [1]: summary(data, var_names=["~^t"], filter_vars="regex")

    """
    if ci_kind not in ("hdi", "eti", None):
        raise ValueError("ci_kind must be either 'hdi' or 'eti'")

    kinds = [
        "all",
        "stats",
        "diagnostics",
        "all_median",
        "stats_median",
        "diagnostics_median",
        "mc_diagnostics",
    ]
    if kind not in kinds:
        raise ValueError(
            "valid options for kind are: " + ", ".join(kinds[:-1]) + " or " + kinds[-1]
        )

    if ci_prob is None:
        ci_prob = rcParams["stats.ci_prob"]
    if ci_kind is None:
        ci_kind = rcParams["stats.ci_kind"]
    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]

    ci_perc = int(ci_prob * 100)

    dataset = extract(
        data,
        var_names=var_names,
        filter_vars=filter_vars,
        group=group,
        sample_dims=sample_dims,
        combined=False,
        keep_dataset=True,
    )
    if coords is not None:
        dataset = dataset.sel(coords)

    to_concat = []

    if kind in ["stats", "all"]:
        mean = dataset.mean(dim=sample_dims, skipna=skipna).expand_dims(summary=["mean"])
        std = dataset.std(dim=sample_dims, skipna=skipna).expand_dims(summary=["sd"])
        ci = (
            dataset.azstats.eti(prob=ci_prob, dim=sample_dims, skipna=skipna)
            .rename({"ci_bound": "summary"})
            .assign_coords(summary=[f"{ci_kind}{ci_perc}_lb", f"{ci_kind}{ci_perc}_ub"])
        )

        to_concat.extend((mean, std, ci))

    if kind in ["diagnostics", "all"]:
        ess_bulk = dataset.azstats.ess(sample_dims=sample_dims, method="bulk").expand_dims(
            summary=["ess_bulk"]
        )
        ess_tail = dataset.azstats.ess(sample_dims=sample_dims, method="tail").expand_dims(
            summary=["ess_tail"]
        )
        rhat = dataset.azstats.rhat(sample_dims=sample_dims).expand_dims(summary=["r_hat"])
        mcse_mean = dataset.azstats.mcse(sample_dims=sample_dims, method="mean").expand_dims(
            summary=["mcse_mean"]
        )
        mcse_sd = dataset.azstats.mcse(sample_dims=sample_dims, method="sd").expand_dims(
            summary=["mcse_sd"]
        )

        to_concat.extend((ess_bulk, ess_tail, rhat, mcse_mean, mcse_sd))

    if kind in ["stats_median", "all_median"]:
        median = dataset.median(dim=sample_dims, skipna=skipna).expand_dims(summary=["median"])
        mad = stats.median_abs_deviation(
            dataset, dims=("chain", "draw"), nan_policy="omit" if skipna else "propagate"
        ).expand_dims(summary=["mad"])
        ci = (
            dataset.azstats.eti(prob=ci_prob, dim=sample_dims, skipna=skipna)
            .rename({"ci_bound": "summary"})
            .assign_coords(summary=[f"eti{ci_perc}_lb", f"eti{ci_perc}_ub"])
        )

        to_concat.extend((median, mad, ci))

    if kind in ["diagnostics_median", "all_median"]:
        ess_median = dataset.azstats.ess(sample_dims=sample_dims, method="median").expand_dims(
            summary=["ess_median"]
        )
        ess_tail = dataset.azstats.ess(sample_dims=sample_dims, method="tail").expand_dims(
            summary=["ess_tail"]
        )
        rhat = dataset.azstats.rhat(sample_dims=sample_dims).expand_dims(summary=["r_hat"])
        mcse_median = dataset.azstats.mcse(sample_dims=sample_dims, method="median").expand_dims(
            summary=["mcse_median"]
        )

        to_concat.extend((ess_median, ess_tail, rhat, mcse_median))

    if kind == "mc_diagnostics":
        mcse_mean = dataset.azstats.mcse(sample_dims=sample_dims, method="mean").expand_dims(
            summary=["mcse_mean"]
        )
        ess_mean = dataset.azstats.ess(sample_dims=sample_dims, method="mean").expand_dims(
            summary=["ess_mean"]
        )
        min_ss = dataset.azstats.pareto_min_ss(sample_dims=sample_dims).expand_dims(
            summary=["min_ss"]
        )

        to_concat.extend((mcse_mean, ess_mean, min_ss))

    summary_df = dataset_to_dataframe(
        xr.concat(to_concat, dim="summary"), sample_dims=["summary"]
    ).T
    if (round_to is not None) and (round_to not in ("None", "none")):
        summary_df = summary_df.round(round_to)

    return summary_df


def ci_in_rope(
    data,
    rope,
    var_names=None,
    filter_vars=None,
    group="posterior",
    dim=None,
    ci_prob=None,
    ci_kind=None,
    rope_dim="rope_dim",
):
    """
    Compute the percentage of a credible interval that falls within a ROPE.

    A region of practical equivalence (ROPE) indicates a small range of parameter values
    that are considered to be practically equivalent to the null value for purposes of the
    particular application see [1]_ for more details.

    Parameters
    ----------
    data : DataTree, DataSet or InferenceData
    rope : (2,) array-like or dict of {hashable : (2,) array-like} or Dataset
        If tuple, the lower and upper bounds of the ROPE are the same for all variables.
        If dict, the keys are the variable names and the values are tuples with the lower
        and upper bounds of the ROPE. The keys must be in `var_names`.
    var_names : list of str, optional
        Names of variables for which the ROPE should be computed.
        If None all variables are included.
    filter_vars: {None, "like", "regex"}, default None
        Used for `var_names` only.
        If ``None`` (default), interpret var_names as the real variables names.
        If "like", interpret var_names as substrings of the real variables names.
        If "regex", interpret var_names as regular expressions on the real variables names.
    group: str
        Select a group to compute the ROPE. Defaults to “posterior”.
    coords : dict, optional
        Coordinates defining a subset over the selected group.
    dim : str or sequence of hashable, optional
        Defaults to ``rcParams["data.sample_dims"]``
    ci_prob : float, optional
        Probability for the credible interval. Defaults to ``rcParams["stats.ci_prob"]``.
    ci_kind : {"hdi", "eti"}, optional
        Type of credible interval. Defaults to ``rcParams["stats.ci_kind"]``.
        If `kind` is stats_median or all_median, `ci_kind` is forced to "eti".
    rope_dim : str, default "rope_dim"
        Name for the dimension containing the ROPE values. Only used when `rope`
        is a :class:`~xarray.Dataset`

    Returns
    -------
    xarray.Dataset

    See Also
    --------
    arviz.summary : Compute summary statistics and or diagnostics.
    arviz.hdi : Compute highest density interval (HDI).
    arviz.eti : Compute equal tail interval (ETI).

    Examples
    --------
    Apply the same ROPE to a subset of variables:

    .. ipython::

        In [1]: from arviz_base import load_arviz_data
           ...: from arviz_stats import ci_in_rope
           ...: data = load_arviz_data("centered_eight")
           ...: ci_in_rope(data, var_names=["mu", "tau"], rope=(-0.5, 0.5))

    Apply different ROPEs to each variable:

    .. ipython::

        In [1]: ci_in_rope(data, rope={"mu": (-0.5, 0.5), "tau": (0.1, 0.2), "theta": (-0.1, 0.1)})

    References
    ----------
    .. [1] Kruschke. Doing Bayesian Data Analysis, Second Edition: A Tutorial with R,
        JAGS, and Stan. Academic Press, 2014. ISBN 978-0-12-405888-0.
    """
    sample_dims = validate_dims(dim)

    dataset = extract(
        data,
        var_names=var_names,
        filter_vars=filter_vars,
        group=group,
        sample_dims=sample_dims,
        combined=False,
        keep_dataset=True,
    )

    if isinstance(rope, dict):
        if not all(var in dataset.data_vars for var in rope.keys()):
            raise ValueError("`rope` must be a subset of the variables in `data`")
        if not all(isinstance(v, tuple) and len(v) == 2 for v in rope.values()):
            raise ValueError("`rope` must be a dict of tuples of length 2")
    elif isinstance(rope, xr.Dataset):
        if rope_dim not in rope.dims:
            raise ValueError(f"{rope_dim} is not a dimension of `rope`")
        if (rope_len := rope.sizes[rope_dim]) != 2:
            raise ValueError(f"Length of {rope_dim} dim must be 2 but is {rope_len}")
    else:
        try:
            rope = np.array(rope)
        except ValueError as err:
            raise ValueError(
                "`rope` must be a dict, Dataset or array-like, failed to convert to array"
            ) from err
        if len(rope) != 2:
            raise ValueError("`rope` must be a tuple of length 2")

    if ci_kind == "eti":
        c_i = dataset.azstats.eti(prob=ci_prob, dim=sample_dims)
    else:
        c_i = dataset.azstats.hdi(prob=ci_prob, dim=sample_dims)

    ci_low = c_i.sel(ci_bound="lower")
    ci_high = c_i.sel(ci_bound="upper")

    in_ci = (dataset >= ci_low) & (dataset <= ci_high)

    rope = references_to_dataset(rope, dataset, sample_dims=sample_dims, ref_dim=rope_dim)

    ci_samples = dataset.where(in_ci)
    in_rope = (ci_samples >= rope.sel({rope_dim: 0})) & (ci_samples <= rope.sel({rope_dim: 1}))

    proportion = (in_rope.sum(dim=sample_dims) / in_ci.sum(dim=sample_dims)) * 100

    return proportion
