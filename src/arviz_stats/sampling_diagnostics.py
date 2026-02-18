"""Functions for sampling diagnostics."""

import sys

import numpy as np
import xarray as xr
from arviz_base import convert_to_dataset, convert_to_datatree, rcParams

from arviz_stats.utils import _apply_multi_input_function, get_array_function
from arviz_stats.validate import validate_dims


def ess(
    data,
    sample_dims=None,
    group="posterior",
    var_names=None,
    filter_vars=None,
    coords=None,
    method="bulk",
    relative=False,
    prob=None,
    chain_axis=0,
    draw_axis=1,
):
    r"""Estimate the effective sample size (ess).

    Parameters
    ----------
    data : array-like, DataArray, Dataset, DataTree, DataArrayGroupBy, DatasetGroupBy, or idata-like
        Input data. It will have different pre-processing applied to it depending on its type:

        - array-like: call array layer within ``arviz-stats``.
        - xarray object: apply dimension aware function to all relevant subsets
        - others: passed to :func:`arviz_base.convert_to_dataset`

    sample_dims : iterable of hashable, optional
        Dimensions to be considered sample dimensions and are to be reduced.
        Default ``rcParams["data.sample_dims"]``.
    group : hashable, default "posterior"
        Group on which to compute the ESS.
    var_names : str or list of str, optional
        Names of the variables for which the ess should be computed.
    filter_vars : {None, "like", "regex"}, default None
    coords : dict, optional
        Dictionary of dimension/index names to coordinate values defining a subset
        of the data for which to perform the computation.
    method : str, default "bulk"
        Valid methods are:

        - "bulk"
        - "tail"     # prob, optional
        - "quantile" # prob
        - "mean" (old ess)
        - "sd"
        - "median"
        - "mad" (mean absolute deviance)
        - "z_scale"
        - "folded"
        - "identity"
        - "local"  # prob

    relative : bool
        Return relative ess ``ress = ess / n``
    prob : float, or tuple of two floats, optional
        Probability value for "tail", "quantile" or "local" ess functions.
    chain_axis, draw_axis : int, optional
        Integer indicators of the axis that correspond to the chain and the draw dimension.
        `chain_axis` can be ``None``.

    Returns
    -------
    ndarray, DataArray, Dataset, DataTree
        Requested ESS summary of the provided input

    See Also
    --------
    arviz.rhat : Compute estimate of rank normalized split R-hat for a set of traces.
    arviz.mcse : Calculate Markov Chain Standard Error statistic.
    plot_ess : Plot quantile, local or evolution of effective sample sizes (ESS).
    arviz.summary : Create a data frame with summary statistics.

    Examples
    --------
    Calculate the effective_sample_size using the default arguments:

    .. ipython::

        In [1]: from arviz_base import load_arviz_data
           ...: import arviz_stats as azs
           ...: data = load_arviz_data('non_centered_eight')
           ...: azs.ess(data)

    Calculate ess for a subset of the variables

    .. ipython::

        In [1]: azs.ess(data, relative=True, var_names=["mu", "theta_t"])

    Calculate ess using the "tail" method, leaving the `prob` at its default value.

    .. ipython::

        In [1]: azs.ess(data, method="tail")
    """
    if isinstance(data, list | tuple | np.ndarray):
        data = np.array(data)
        return get_array_function("ess")(
            data,
            method=method,
            relative=relative,
            prob=prob,
            chain_axis=chain_axis,
            draw_axis=draw_axis,
        )

    if isinstance(data, xr.core.groupby.DataArrayGroupBy | xr.core.groupby.DatasetGroupBy):
        # Make sure the grouped dimension is added as one of the dimensions to be reduced
        sample_dims = list(set(validate_dims(sample_dims)).union(data.group1d.dims))
        return data.map(
            ess,
            sample_dims=sample_dims,
            var_names=var_names,
            coords=coords,
            method=method,
            relative=relative,
            prob=prob,
        )

    if isinstance(data, xr.DataArray):
        if coords is not None:
            data = data.sel(coords)
        return data.azstats.ess(
            sample_dims=sample_dims, method=method, relative=relative, prob=prob
        )

    if isinstance(data, xr.DataTree):
        data = data.azstats.filter_vars(
            group=group, var_names=var_names, filter_vars=filter_vars
        ).datatree
        if coords is not None:
            data = data.sel(coords)
        return data.azstats.ess(
            sample_dims=sample_dims, group=group, method=method, relative=relative, prob=prob
        )

    data = convert_to_dataset(data, group=group)

    data = data.azstats.filter_vars(var_names=var_names, filter_vars=filter_vars).dataset
    if coords is not None:
        data = data.sel(coords)

    return data.azstats.ess(sample_dims=sample_dims, method=method, relative=relative, prob=prob)


def rhat(
    data,
    sample_dims=None,
    group="posterior",
    var_names=None,
    filter_vars=None,
    coords=None,
    method="rank",
    chain_axis=0,
    draw_axis=1,
):
    r"""Compute estimate of rank normalized split R-hat for a set of traces.

    The rank normalized R-hat diagnostic tests for lack of convergence by comparing the variance
    between multiple chains to the variance within each chain. If convergence has been achieved,
    the between-chain and within-chain variances should be identical. To be most effective in
    detecting evidence for nonconvergence, each chain should have been initialized to starting
    values that are dispersed relative to the target distribution.

    Parameters
    ----------
    data : array-like, DataArray, Dataset, DataTree, DataArrayGroupBy, DatasetGroupBy, or idata-like
        Input data. It will have different pre-processing applied to it depending on its type:

        - array-like: call array layer within ``arviz-stats``.
        - xarray object: apply dimension aware function to all relevant subsets
        - others: passed to :func:`arviz_base.convert_to_dataset`

        At least 2 posterior chains are needed to compute this diagnostic of one or more
        stochastic parameters.

    sample_dims : iterable of hashable, optional
        Dimensions to be considered sample dimensions and are to be reduced.
        Default ``rcParams["data.sample_dims"]``.
    group : hashable, default "posterior"
        Group on which to compute the ESS.
    var_names : str or list of str, optional
        Names of the variables for which the Rhat should be computed.
    filter_vars : {None, "like", "regex"}, default None
    coords : dict, optional
        Dictionary of dimension/index names to coordinate values defining a subset
        of the data for which to perform the computation.
    method : str, default "rank"
        Valid methods are:
        - "rank"        # recommended by Vehtari et al. (2021)
        - "split"
        - "folded"
        - "z_scale"
        - "identity"
    chain_axis, draw_axis : int, optional
        Integer indicators of the axis that correspond to the chain and the draw dimension.
        `chain_axis` can be ``None``.

    Returns
    -------
    ndarray, DataArray, Dataset, DataTree
        Requested Rhat summary of the provided input


    See Also
    --------
    arviz.ess : Calculate estimate of the effective sample size (ess).
    arviz.mcse : Calculate Markov Chain Standard Error statistic.
    plot_forest : Forest plot to compare HDI intervals from a number of distributions.

    Notes
    -----
    The diagnostic is computed by:

      .. math:: \hat{R} = \sqrt{\frac{\hat{V}}{W}}

    where :math:`W` is the within-chain variance and :math:`\hat{V}` is the posterior variance
    estimate for the pooled rank-traces. This is the potential scale reduction factor, which
    converges to unity when each of the traces is a sample from the target posterior. Values
    greater than one indicate that one or more chains have not yet converged.

    Rank values are calculated over all the chains with ``scipy.stats.rankdata``.
    Each chain is split in two and normalized with the z-transform following [1]_.

    References
    ----------
    .. [1] Vehtari et al. *Rank-normalization, folding, and localization: An improved Rhat
           for assessing convergence of MCMC*. Bayesian Analysis, 16, 2 (2021).
           https://doi.org/10.1214/20-BA1221

    Examples
    --------
    Calculate the R-hat using the default arguments:

    .. ipython::

        In [1]: from arviz_base import load_arviz_data
           ...: import arviz_stats as azs
           ...: data = load_arviz_data('non_centered_eight')
           ...: azs.rhat(data)

    Calculate the R-hat of some variables using the folded method:

    .. ipython::

        In [1]: azs.rhat(data, var_names=["mu", "theta_t"], method="folded")

    """
    if isinstance(data, list | tuple | np.ndarray):
        data = np.array(data)
        return get_array_function("rhat")(
            data,
            method=method,
            chain_axis=chain_axis,
            draw_axis=draw_axis,
        )

    if isinstance(data, xr.core.groupby.DataArrayGroupBy | xr.core.groupby.DatasetGroupBy):
        # Make sure the grouped dimension is added as one of the dimensions to be reduced
        sample_dims = list(set(validate_dims(sample_dims)).union(data.group1d.dims))
        return data.map(
            rhat,
            sample_dims=sample_dims,
            var_names=var_names,
            coords=coords,
            method=method,
        )

    if isinstance(data, xr.DataArray):
        if coords is not None:
            data = data.sel(coords)
        return data.azstats.rhat(sample_dims=sample_dims, method=method)

    if isinstance(data, xr.DataTree):
        data = data.azstats.filter_vars(
            group=group, var_names=var_names, filter_vars=filter_vars
        ).datatree
        if coords is not None:
            data = data.sel(coords)
        return data.azstats.rhat(sample_dims=sample_dims, group=group, method=method)

    data = convert_to_dataset(data, group=group)

    data = data.azstats.filter_vars(var_names=var_names, filter_vars=filter_vars).dataset
    if coords is not None:
        data = data.sel(coords)

    return data.azstats.rhat(sample_dims=sample_dims, method=method)


def rhat_nested(
    data,
    sample_dims=None,
    group="posterior",
    var_names=None,
    filter_vars=None,
    method="rank",
    coords=None,
    superchain_ids=None,
    chain_axis=0,
    draw_axis=1,
):
    """Compute nested R-hat.

    Nested R-hat is a convergence diagnostic useful when running many short chains.
    It is calculated on superchains, which are groups of chains that have been
    initialized at the same point.

    Note that there is a slight difference in the calculation of R-hat and nested R-hat,
    as nested R-hat is lower bounded by 1. This means that nested R-hat with one chain per
    superchain will not be exactly equal to basic R-hat see [1]_ for details.

    Parameters
    ----------
    data : array-like, DataArray, Dataset, DataTree, DataArrayGroupBy, DatasetGroupBy, or idata-like
        Input data. It will have different pre-processing applied to it depending on its type:

        - array-like: call array layer within ``arviz-stats``.
        - xarray object: apply dimension aware function to all relevant subsets
        - others: passed to :func:`arviz_base.convert_to_dataset`

        At least 2 posterior chains are needed to compute this diagnostic of one or more
        stochastic parameters.

    sample_dims : iterable of hashable, optional
        Dimensions to be considered sample dimensions and are to be reduced.
        Default ``rcParams["data.sample_dims"]``.
    group : hashable, default "posterior"
        Group on which to compute the R-hat.
    var_names : str or list of str, optional
        Names of the variables for which the Rhat should be computed.
    filter_vars : {None, "like", "regex"}, default None
    method : str, default "rank"
        Valid methods are:

        - "rank"        # recommended by Vehtari et al. (2021) [2]_
        - "split"
        - "folded"
        - "z_scale"
        - "identity"
    coords : dict, optional
        Dictionary of dimension/index names to coordinate values defining a subset
        of the data for which to perform the computation.
    superchain_ids : list
        Lisf ot length ``chains`` specifying which superchain each chain belongs to.
        There should be equal numbers of chains in each superchain. All chains within
        the same superchain are assumed to have been initialized at the same point.
    chain_axis, draw_axis : int, optional
        Integer indicators of the axis that correspond to the chain and the draw dimension.
        `chain_axis` can be ``None``.

    See Also
    --------
    arviz.rhat : Calculate estimate of the effective sample size (ess).
    arviz.ess : Calculate Markov Chain Standard Error statistic.
    plot_forest : Forest plot to compare HDI intervals from a number of distributions.

    References
    ----------
    .. [1] Margossian et al. *Nested R-hat: Assessing the convergence of
        Markov Chain Monte Carlo when running many short chains*.
        Bayesian Analysis, (2024). https://doi.org/10.1214/24-BA1453
    .. [2] Vehtari et al. *Rank-normalization, folding, and localization: An improved Rhat
        for assessing convergence of MCMC*. Bayesian Analysis, 16, 2 (2021).
        https://doi.org/10.1214/20-BA1221
    """
    if isinstance(data, list | tuple | np.ndarray):
        data = np.array(data)
        return get_array_function("rhat_nested")(
            data,
            method=method,
            chain_axis=chain_axis,
            draw_axis=draw_axis,
            superchain_ids=superchain_ids,
        )

    if isinstance(data, xr.core.groupby.DataArrayGroupBy | xr.core.groupby.DatasetGroupBy):
        # Make sure the grouped dimension is added as one of the dimensions to be reduced
        sample_dims = list(set(validate_dims(sample_dims)).union(data.group1d.dims))
        return data.map(
            rhat_nested,
            sample_dims=sample_dims,
            var_names=var_names,
            coords=coords,
            method=method,
            superchain_ids=superchain_ids,
        )

    if isinstance(data, xr.DataArray):
        if coords is not None:
            data = data.sel(coords)
        return data.azstats.rhat_nested(
            sample_dims=sample_dims, method=method, superchain_ids=superchain_ids
        )

    if isinstance(data, xr.DataTree):
        data = data.azstats.filter_vars(
            group=group, var_names=var_names, filter_vars=filter_vars
        ).datatree
        if coords is not None:
            data = data.sel(coords)
        return data.azstats.rhat_nested(
            sample_dims=sample_dims, group=group, method=method, superchain_ids=superchain_ids
        )

    data = convert_to_dataset(data, group=group)

    data = data.azstats.filter_vars(var_names=var_names, filter_vars=filter_vars).dataset
    if coords is not None:
        data = data.sel(coords)

    return data.azstats.rhat_nested(
        sample_dims=sample_dims, method=method, superchain_ids=superchain_ids
    )


def mcse(
    data,
    sample_dims=None,
    group="posterior",
    var_names=None,
    filter_vars=None,
    coords=None,
    method="mean",
    prob=None,
    chain_axis=0,
    draw_axis=1,
):
    """Calculate Markov Chain Standard Error statistic.

    Parameters
    ----------
    data : array-like, DataArray, Dataset, DataTree, DataArrayGroupBy, DatasetGroupBy, or idata-like
        Input data. It will have different pre-processing applied to it depending on its type:

        - array-like: call array layer within ``arviz-stats``.
        - xarray object: apply dimension aware function to all relevant subsets
        - others: passed to :func:`arviz_base.convert_to_dataset`

    sample_dims : iterable of hashable, optional
        Dimensions to be considered sample dimensions and are to be reduced.
        Default ``rcParams["data.sample_dims"]``.
    group : hashable, default "posterior"
        Group on which to compute the MCSE.
    var_names : str or list of str, optional
        Names of the variables for which the mcse should be computed.
    filter_vars : {None, "like", "regex"}, default None
    coords : dict, optional
        Dictionary of dimension/index names to coordinate values defining a subset
        of the data for which to perform the computation.
    method : str, default "mean"
        Valid methods are:

        - "mean"
        - "sd"
        - "median"
        - "quantile"
    prob : float, or tuple of two floats, optional
        Probability value "quantile".
    chain_axis, draw_axis : int, optional
        Integer indicators of the axis that correspond to the chain and the draw dimension.
        `chain_axis` can be ``None``.

    Returns
    -------
    ndarray, DataArray, Dataset, DataTree
        Requested mcse summary of the provided input

    See Also
    --------
    arviz.ess : Compute autocovariance estimates for every lag for the input array.
    arviz.summary : Create a data frame with summary statistics.
    arviz_plots.plot_mcse : Plot quantile or local Monte Carlo Standard Error.

    Examples
    --------
    Calculate the Markov Chain Standard Error using the default arguments:

    .. ipython::

        In [1]: from arviz_base import load_arviz_data
           ...: import arviz_stats as azs
           ...: data = load_arviz_data('non_centered_eight')
           ...: azs.mcse(data)

    Calculate the Markov Chain Standard Error using the quantile method:

    .. ipython::

        In [1]: azs.mcse(data, method="quantile", prob=0.7)

    """
    if isinstance(data, list | tuple | np.ndarray):
        data = np.array(data)
        return get_array_function("mcse")(
            data,
            method=method,
            prob=prob,
            chain_axis=chain_axis,
            draw_axis=draw_axis,
        )

    if isinstance(data, xr.core.groupby.DataArrayGroupBy | xr.core.groupby.DatasetGroupBy):
        # Make sure the grouped dimension is added as one of the dimensions to be reduced
        sample_dims = list(set(validate_dims(sample_dims)).union(data.group1d.dims))
        return data.map(
            mcse,
            sample_dims=sample_dims,
            var_names=var_names,
            coords=coords,
            method=method,
            prob=prob,
        )

    if isinstance(data, xr.DataArray):
        if coords is not None:
            data = data.sel(coords)
        return data.azstats.mcse(sample_dims=sample_dims, method=method, prob=prob)

    if isinstance(data, xr.DataTree):
        data = data.azstats.filter_vars(
            group=group, var_names=var_names, filter_vars=filter_vars
        ).datatree
        if coords is not None:
            data = data.sel(coords)
        return data.azstats.mcse(sample_dims=sample_dims, group=group, method=method, prob=prob)

    data = convert_to_dataset(data, group=group)

    data = data.azstats.filter_vars(var_names=var_names, filter_vars=filter_vars).dataset
    if coords is not None:
        data = data.sel(coords)

    return data.azstats.mcse(sample_dims=sample_dims, method=method, prob=prob)


def bfmi(
    data,
    sample_dims=None,
    group="sample_stats",
    var_names="energy",
    filter_vars=None,
    coords=None,
    **kwargs,
):
    """Calculate the estimated Bayesian fraction of missing information (BFMI).

    BFMI quantifies how well momentum resampling matches the marginal energy distribution as
    explained in [1]_. As a rule of thumb, BFMI values smaller than 0.3 are indicative of poor
    sampling.

    See the EABM chapter on MCMC diagnostic `of gradient-based algorithms <https://arviz-devs.github.io/EABM/Chapters/MCMC_diagnostics.html#diagnosis-of-gradient-based-algorithms>`_
    for more details.

    Parameters
    ----------
    data : array-like, DataArray, Dataset, DataTree, DataArrayGroupBy, DatasetGroupBy, or idata-like
        Input data. It will have different pre-processing applied to it depending on its type:

        - array-like: call array layer within ``arviz-stats``.
        - xarray object: apply dimension aware function to all relevant subsets
        - others: passed to :func:`arviz_base.convert_to_dataset`

    sample_dims : iterable of hashable, optional
        Dimensions to be considered sample dimensions and are to be reduced.
        Default ``rcParams["data.sample_dims"]``.
    group : hashable, default "sample_stats"
        Group on which to compute the BFMI.
    var_names : str or list of str, optional
        Names of the variables for which the BFMI should be computed. Defaults to "energy".
    filter_vars : {None, "like", "regex"}, default None
    coords : dict, optional
        Dictionary of dimension/index names to coordinate values defining a subset
        of the data for which to perform the computation.
    **kwargs : any, optional
        Forwarded to the array or dataarray interface for additional options.

    Returns
    -------
    ndarray, DataArray, Dataset, DataTree
        Requested BFMI summary of the provided input

    Examples
    --------
    Compute the BFMI for the 'radon' example:

    .. ipython::

        In [1]: from arviz_base import load_arviz_data
           ...: import arviz_stats as azs
           ...: data = load_arviz_data('radon')
           ...: azs.bfmi(data)

    References
    ----------
    .. [1] Betancourt. Diagnosing Suboptimal Cotangent Disintegrations in
        Hamiltonian Monte Carlo. (2016) https://arxiv.org/abs/1604.00695
    """
    return _apply_multi_input_function(
        "bfmi",
        data,
        sample_dims,
        "sample_dims",
        group=group,
        var_names=var_names,
        filter_vars=filter_vars,
        coords=coords,
        **kwargs,
    )


def diagnose(
    data,
    *,
    var_names=None,
    filter_vars=None,
    coords=None,
    sample_dims=None,
    group="posterior",
    rhat_max=1.01,
    ess_min_ratio=0.001,
    bfmi_threshold=0.3,
    show_diagnostics=True,
    return_diagnostics=False,
):
    """Run comprehensive diagnostic checks for MCMC sampling.

    This function performs diagnostic checks on MCMC samples similar to CmdStan's diagnose
    utility. It checks for:

    - Divergent transitions
    - Maximum tree depth saturation
    - Low E-BFMI (Energy Bayesian Fraction of Missing Information)
    - Low effective sample size (ESS)
    - High R-hat values

    See [1]_ and [2]_ for more details. You can also check https://arviz-devs.github.io/EABM/Chapters/MCMC_diagnostics.html
    for a more practical overview.

    Parameters
    ----------
    data : DataTree, Dataset, or InferenceData-like
        Input data. To be able to compute all diagnostics, the data should contain MCMC
        posterior samples (see ``group`` argument) and sampler statistics (in "sample_stats" group).
    var_names : str or list of str, optional
        Names of variables to check for R-hat and ESS diagnostics.
        If None, checks all variables.
    filter_vars : {None, "like", "regex"}, default None
        How to filter variable names. See :func:`filter_vars` for details.
    coords : dict, optional
        Coordinates to select a subset of the data.
    sample_dims : iterable of hashable, optional
        Dimensions to be considered sample dimensions.
        Default from ``rcParams["data.sample_dims"]``.
    group : str, default "posterior"
        Group to check for convergence diagnostics (R-hat, ESS).
    rhat_max : float, default 1.01
        Maximum acceptable R-hat value. Parameters with R-hat > rhat_max
        will be flagged.
    ess_min_ratio : float, default 0.001
        Minimum acceptable ratio of ESS to total samples. Parameters with
        ESS/N < ess_min_ratio will be flagged.
        A flag is also emitted if ESS is lower than 100 * number of chains.
    bfmi_threshold : float, default 0.3
        Minimum acceptable E-BFMI value. Values below this threshold indicate
        potential issues with the sampler's exploration.
    show_diagnostics : bool, default True
        If True, print diagnostic messages to stdout. If False, return results silently.
    return_diagnostics : bool, default False
        If True, return a dictionary with detailed diagnostic results in addition
        to the boolean has_errors flag.

    Returns
    -------
    has_errors : bool
        True if any diagnostic checks failed, False otherwise.
    diagnostics : dict, optional
        Only returned if return_diagnostics=True.

        - "divergent": dict with keys "n_divergent", "pct", "total_samples"
        - "treedepth": dict with keys "n_max", "pct", "total_samples"
        - "bfmi": dict with keys "bfmi_values", "failed_chains", "threshold"
        - "ess": dict with keys "bad_params", "ess_values", "threshold_ratio", "total_samples"
        - "rhat": dict with keys "bad_params", "rhat_values", "threshold"

    Examples
    --------
    Get diagnostics printted to stdout:

    .. ipython::

        In [1]: import arviz_stats as azs
           ...: from arviz_base import load_arviz_data
           ...: data = load_arviz_data('centered_eight')
           ...: azs.diagnose(data)

    Get detailed diagnostic information without printing messages:

    .. ipython::

        In [1]: _, diagnostics = azs.diagnose(data, return_diagnostics=True, show_diagnostics=False)
           ...: diagnostics

    See Also
    --------
    rhat : Compute R-hat convergence diagnostic
    ess : Compute effective sample size
    bfmi : Compute Bayesian fraction of missing information
    summary : Create a data frame with summary statistics, including diagnostics.

    References
    ----------
    .. [1] Vehtari et al. *Rank-normalization, folding, and localization: An improved Rhat for
        assessing convergence of MCMC*. Bayesian Analysis. 16(2) (2021)
        https://doi.org/10.1214/20-BA1221. arXiv preprint https://arxiv.org/abs/1903.08008
    .. [2] Betancourt. Diagnosing Suboptimal Cotangent Disintegrations in
        Hamiltonian Monte Carlo. (2016) https://arxiv.org/abs/1604.00695
    """
    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]
    if isinstance(sample_dims, str):
        sample_dims = [sample_dims]

    dt = convert_to_datatree(data)

    has_errors = False
    diagnostics_results = {}
    messages = []

    sample_stats = dt.get("sample_stats")
    if sample_stats is None:
        messages.append("No sample_stats group found. Skipping sampler-specific diagnostics.")
        sample_stats_available = False
    else:
        sample_stats_available = True

    posterior = dt[group]

    total_samples = np.prod([posterior.sizes[dim] for dim in sample_dims if dim in posterior.sizes])

    # Check divergences
    if sample_stats_available and "diverging" in sample_stats:
        diverging = sample_stats["diverging"]
        n_divergent = int(diverging.sum().values)

        diagnostics_results["divergent"] = {
            "n_divergent": n_divergent,
            "pct": 100 * n_divergent / total_samples,
            "total_samples": total_samples,
        }

        messages.append("Divergences")
        if n_divergent > 0:
            has_errors = True
            pct = diagnostics_results["divergent"]["pct"]
            messages.append(
                f"{n_divergent} of {total_samples} ({pct:.2f}%) transitions ended with a "
                "divergence.\n"
                "These divergent transitions indicate that HMC is not fully able to explore "
                "the posterior distribution.\n"
                "Try increasing adapt delta closer to 1.\n"
                "If this doesn't remove all divergences, try to reparameterize the model."
            )
        else:
            messages.append("No divergent transitions found.")

    # Check tree depth
    if sample_stats_available and "reached_max_treedepth" in sample_stats:
        reached_max_treedepth = sample_stats["reached_max_treedepth"]
        n_max = int((reached_max_treedepth).sum().values)

        diagnostics_results["treedepth"] = {
            "n_max": n_max,
            "pct": 100 * n_max / total_samples,
            "total_samples": total_samples,
        }

        messages.append("\nTree depth")
        if n_max:
            has_errors = True
            pct = diagnostics_results["treedepth"]["pct"]
            messages.append(
                f"{n_max} of {total_samples} ({pct:.2f}%) transitions hit the maximum treedepth "
                "limit.\n"
                "Trajectories that are prematurely terminated due to this limit will result "
                "in slow exploration.\nFor optimal performance, increase this limit."
            )
        else:
            messages.append("Treedepth satisfactory for all transitions.")

    # Check E-BFMI
    if sample_stats_available and "energy" in sample_stats:
        bfmi_values = bfmi(dt, sample_dims=sample_dims)["energy"]

        low_bfmi = bfmi_values < 0.3
        chain_indices = low_bfmi.where(low_bfmi, drop=True).coords["chain"].values.tolist()

        diagnostics_results["bfmi"] = {
            "bfmi_values": bfmi_values.values,
            "failed_chains": chain_indices,
            "threshold": bfmi_threshold,
        }

        messages.append("\nE-BFMI")
        if chain_indices:
            has_errors = True
            for chain_idx in chain_indices:
                bfmi_val = bfmi_values.sel(chain=chain_idx).item()
                messages.append(f"Chain {chain_idx}: E-BFMI = {bfmi_val:.3f}")
            messages.append(
                f"E-BFMI values are below the threshold {bfmi_threshold:.2f} which suggests that "
                "HMC may have trouble exploring the target distribution.\n"
                "If possible, try to reparameterize the model."
            )
        else:
            messages.append("E-BFMI satisfactory for all chains.")

    # Check ESS
    ess_bulk = ess(
        dt,
        sample_dims=sample_dims,
        group=group,
        var_names=var_names,
        filter_vars=filter_vars,
        coords=coords,
        method="bulk",
    )
    ess_tail = ess(
        dt,
        sample_dims=sample_dims,
        group=group,
        var_names=var_names,
        filter_vars=filter_vars,
        coords=coords,
        method="tail",
    )

    ess_min = np.minimum(ess_bulk.ds, ess_tail.ds)
    ess_ratio = ess_min / total_samples
    bad_ess_params = [var for var in ess_ratio.data_vars if (ess_ratio[var] < ess_min_ratio).any()]

    ess_threshold = 100 * len(posterior.coords["chain"])
    below_minimum_params = [
        var for var in ess_bulk.ds.data_vars if (ess_bulk.ds[var] < ess_threshold).any()
    ]

    diagnostics_results["ess"] = {
        "bad_params": bad_ess_params,
        "ess_values": ess_min[bad_ess_params],
        "threshold_ratio": ess_min_ratio,
        "total_samples": total_samples,
    }

    messages.append("\nESS")
    if bad_ess_params:
        has_errors = True
        messages.append(
            f"The following parameters has fewer than {ess_min_ratio:.3f} effective draws per "
            f"transition:\n  {', '.join(bad_ess_params)}\n"
            "Such low values indicate that the effective sample size estimators may be "
            "biased high and actual performance may be substantially lower than quoted."
        )

    if below_minimum_params:
        has_errors = True
        messages.append(
            f"The following parameters has fewer than {ess_threshold} effective samples:\n"
            f"  {', '.join(below_minimum_params)}\n"
            "This suggests that the sampler may not have fully explored the posterior "
            "distribution for this parameter.\nConsider reparameterizing the model or "
            "increasing the number of samples."
        )

    if not bad_ess_params and not below_minimum_params:
        messages.append("Effective sample size satisfactory for all parameters.")

    # Check R-hat
    rhat_rank = rhat(
        dt,
        sample_dims=sample_dims,
        group=group,
        var_names=var_names,
        filter_vars=filter_vars,
        coords=coords,
        method="rank",
    )
    rhat_folded = rhat(
        dt,
        sample_dims=sample_dims,
        group=group,
        var_names=var_names,
        filter_vars=filter_vars,
        coords=coords,
        method="folded",
    )

    rhat_max_vals = np.maximum(rhat_rank.dataset, rhat_folded.dataset)
    bad_rhat_params = [
        var for var in rhat_max_vals.data_vars if (rhat_max_vals[var] > rhat_max).any()
    ]

    diagnostics_results["rhat"] = {
        "bad_params": bad_rhat_params,
        "rhat_values": rhat_max_vals[bad_rhat_params],
        "threshold": rhat_max,
    }

    messages.append("\nR-hat")
    if bad_rhat_params:
        has_errors = True
        messages.append(
            f"The following parameters has R-hat values greater than {rhat_max:.2f}:\n"
            f"  {', '.join(bad_rhat_params)}\n"
            "Such high values indicate incomplete mixing and biased estimation.\n"
            "You should consider regularizing your model with additional prior information or "
            "a more effective parameterization."
        )
    else:
        messages.append("R-hat values satisfactory for all parameters.")

    if not has_errors:
        messages.append("\nProcessing complete, no problems detected.")

    if show_diagnostics:
        print("\n".join(messages), file=sys.stdout)

    if return_diagnostics:
        return has_errors, diagnostics_results

    return has_errors
