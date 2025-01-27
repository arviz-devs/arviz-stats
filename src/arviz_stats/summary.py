"""Summaries for various statistics and diagnostics."""

import xarray as xr
from arviz_base import extract, rcParams
from xarray_einstats import stats

__all__ = ["summary"]


def summary(
    dt,
    var_names=None,
    filter_vars=None,
    group="posterior",
    coords=None,
    sample_dims=None,
    kind="all",
    ci_prob=None,
    ci_kind=None,
):
    """
    Create a data frame with summary statistics and or diagnostics.

    Parameters
    ----------
    dt : obj
        Any object that can be converted to an :class:`arviz.InferenceData` object.
        Refer to documentation of :func:`arviz.convert_to_dataset` for details.
        For ndarray: shape = (chain, draw).
        For n-dimensional ndarray transform first to dataset with ``az.convert_to_dataset``.
    var_names : list of str, optional
        Names of posterior variables to include in the power scaling sensitivity diagnostic
    filter_vars: {None, "like", "regex"}, default None
        Used for `var_names` only.
        If ``None`` (default), interpret var_names as the real variables names.
        If "like", interpret var_names as substrings of the real variables names.
        If "regex", interpret var_names as regular expressions on the real variables names.
    group : {"prior", "likelihood"}, default "prior"
        If "likelihood", the pointsize log likelihood values are retrieved
        from the ``log_likelihood`` group and added together.
        If "prior", the log prior values are retrieved from the ``log_prior`` group.
    coords : dict, optional
        Coordinates defining a subset over the posterior. Only these variables will
        be used when computing the prior sensitivity.
    sample_dims : str or sequence of hashable, optional
        Dimensions to reduce unless mapped to an aesthetic.
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

    Returns
    -------
    pandas.DataFrame or xarray.Dataset

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

        In [1]: import arviz as az
           ...: data = az.load_arviz_data("centered_eight")
           ...: az.summary(data, var_names=["mu", "tau"])

    You can use ``filter_vars`` to select variables without having to specify all the exact
    names. Use ``filter_vars="like"`` to select based on partial naming:

    .. ipython::

        In [1]: az.summary(data, var_names=["the"], filter_vars="like")

    Use ``filter_vars="regex"`` to select based on regular expressions, and prefix the variables
    you want to exclude by ``~``. Here, we exclude from the summary all the variables
    starting with the letter t:

    .. ipython::

        In [1]: az.summary(data, var_names=["~^t"], filter_vars="regex")

    """
    if ci_kind not in ("hdi", "eti", None):
        raise ValueError("ci_kind must be either 'hdi' or 'eti'")

    if ci_prob is None:
        ci_prob = rcParams["stats.ci_prob"]
    if ci_kind is None:
        ci_kind = rcParams["stats.ci_kind"] if "stats.ci_kind" in rcParams else "eti"

    ci_perc = int(ci_prob * 100)

    dataset = extract(
        dt,
        var_names=var_names,
        filter_vars=filter_vars,
        group=group,
        combined=False,
        keep_dataset=True,
    )
    if coords is not None:
        dataset = dataset.sel(coords)

    to_concat = []

    if kind == "stats" or kind == "all":
        mean = dataset.mean(dim=sample_dims).expand_dims(summary=["mean"])
        std = dataset.std(dim=sample_dims).expand_dims(summary=["std"])
        if ci_kind == "eti":
            ci = dataset.azstats.eti(prob=ci_prob, dims=sample_dims).expand_dims(
                summary=[f"eti_{ci_perc}"]
            )
            # ci = dataset.azstats.eti(prob=ci_prob, dims=dims).assign_coords(summary=["lb", "ub"])
        elif ci_kind == "hdi":
            ci = dataset.azstats.hdi(prob=ci_prob, dims=sample_dims).expand_dims(
                summary=[f"hdi_{ci_perc}"]
            )

        to_concat.extend((mean, std, ci))

    if kind == "diagnostics" or kind == "all":
        ess_bulk = dataset.azstats.ess(dims=sample_dims, method="bulk").expand_dims(
            summary=["ess_bulk"]
        )
        ess_tail = dataset.azstats.ess(dims=sample_dims, method="tail").expand_dims(
            summary=["ess_tail"]
        )
        rhat = dataset.azstats.rhat(dims=sample_dims).expand_dims(summary=["rhat"])
        mcse_mean = dataset.azstats.mcse(dims=sample_dims, method="mean").expand_dims(
            summary=["mcse_mean"]
        )
        mcse_sd = dataset.azstats.mcse(dims=sample_dims, method="sd").expand_dims(
            summary=["mcse_sd"]
        )

        to_concat.extend((ess_bulk, ess_tail, rhat, mcse_mean, mcse_sd))

    if kind == "stats_median" or kind == "all_median":
        median = dataset.median(dim=sample_dims).expand_dims(summary=["median"])
        mad = stats.median_abs_deviation(dataset.to_dataset(), dims=("chain", "draw")).expand_dims(
            summary=["mad"]
        )
        if ci_kind == "eti":
            ci = dataset.azstats.eti(prob=ci_prob, dims=sample_dims).expand_dims(
                summary=[f"eti_{ci_perc}"]
            )
            # ci = dataset.azstats.eti(prob=ci_prob, dims=dims).assign_coords(summary=["lb", "ub"])
        elif ci_kind == "hdi":
            ci = dataset.azstats.hdi(prob=ci_prob, dims=sample_dims).expand_dims(
                summary=[f"hdi_{ci_perc}"]
            )

        to_concat.extend((median, mad, ci))

    if kind == "diagnostics_median" or kind == "all_median":
        ess_median = dataset.azstats.ess(dims=sample_dims, method="median").expand_dims(
            summary=["ess_median"]
        )
        ess_tail = dataset.azstats.ess(dims=sample_dims, method="tail").expand_dims(
            summary=["ess_tail"]
        )
        rhat = dataset.azstats.rhat(dims=sample_dims).expand_dims(summary=["rhat"])
        mcse_median = dataset.azstats.mcse(dims=sample_dims, method="median").expand_dims(
            summary=["mcse_median"]
        )

        to_concat.extend((ess_median, ess_tail, rhat, mcse_median))

    if kind == "mc_diagnostics":
        mcse_mean = dataset.azstats.mcse(dims=sample_dims, method="mean").expand_dims(
            summary=["mcse_mean"]
        )
        ess_mean = dataset.azstats.ess(dims=sample_dims, method="mean").expand_dims(
            summary=["ess_mean"]
        )
        min_ss = dataset.azstats.pareto_min_ss(dims=sample_dims).expand_dims(summary=["min_ss"])

        to_concat.extend((mcse_mean, ess_mean, min_ss))

    summaries = xr.concat(to_concat, dim="summary")

    # Uncomment once we have https://github.com/arviz-devs/arviz-base/pull/25
    # dataset_to_dataframe(summaries, sample_dims=["summary"]).T
    return summaries
