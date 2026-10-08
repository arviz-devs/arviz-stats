"""Textual interface for posterior predictive checks based on PIT uniformity."""

import sys
import warnings

import numpy as np
import xarray as xr
from arviz_base import convert_to_datatree
from arviz_base.utils import _var_names
from arviz_base.validate import validate_or_use_rcparam, validate_sample_dims

from arviz_stats.base import array_stats
from arviz_stats.loo import loo_pit

__all__ = ["text_loo_pit", "text_ppc_pit"]

VALID_METHODS = {"pot_c", "prit_c", "piet_c"}


def _validate_method(method):
    """Raise an informative error if ``method`` is not one of the valid uniformity tests."""
    if method not in VALID_METHODS:
        raise ValueError(
            f"Method {method!r} not supported. Choose from 'pot_c', 'prit_c' or 'piet_c'."
        )
    return method


def warn_if_binary(observed_dist, predictive_dist):
    """Warn if data is binary."""
    for dist, name in zip([observed_dist, predictive_dist], ["observed_data", "predictive"]):
        if dist is None:
            continue
        binary_vars = [
            var for var, da in dist.items() if (np.isclose(da, 0) | np.isclose(da, 1)).all()
        ]
        if binary_vars:
            warnings.warn(
                f"Variables {', '.join(binary_vars)} in '{name}' look binary. "
                "For binary outcomes, plot_ppc_pava may be more appropriate.",
                UserWarning,
                stacklevel=2,
            )


def warn_if_prior_predictive(group):
    """Warn if group is prior_predictive."""
    if group == "prior_predictive":
        warnings.warn(
            "This plot always uses the `observed_data` group."
            "\nBe cautious when using it for prior predictive checks.",
            UserWarning,
            stacklevel=2,
        )


def _process_groups(dt, group, var_names, filter_vars, coords):
    """Extract and subset the predictive and observed groups."""
    predictive_dist = dt[group].dataset
    observed_dist = dt["observed_data"].dataset

    if var_names is not None:
        selected = _var_names(var_names, [predictive_dist, observed_dist], filter_vars)
        predictive_dist = predictive_dist[[var for var in predictive_dist if var in selected]]
        observed_dist = observed_dist[[var for var in observed_dist if var in selected]]

    if coords is not None:
        predictive_dist = predictive_dist.sel(coords)
        observed_dist = observed_dist.sel(coords)

    return predictive_dist, observed_dist


def _top_points(pit, shapley, suspicious, max_points):
    """Select the most suspicious observations, sorted by decreasing Shapley contribution."""
    flat_mask = suspicious.values.ravel()
    flat_shapley = shapley.values.ravel()
    flat_pit = pit.values.ravel()

    indices = np.flatnonzero(flat_mask)
    if indices.size == 0:
        return []
    order = np.argsort(-flat_shapley[indices], kind="stable")
    if max_points is not None:
        order = order[:max_points]
    indices = indices[order]

    top_points = []
    for index in indices:
        position = np.unravel_index(index, pit.shape)
        coords = {}
        for dim, i in zip(pit.dims, position, strict=True):
            if dim in pit.coords:
                value = pit.coords[dim].values[i]
                coords[dim] = value.item() if isinstance(value, np.generic) else value
            else:
                coords[dim] = int(i)
        top_points.append(
            {
                "coords": coords,
                "pit": float(flat_pit[index]),
                "shapley": float(flat_shapley[index]),
            }
        )
    return top_points


def _interpret_pit(pit_values):
    """Describe the deviation from uniformity through the shape of the Δ-ECDF curve.

    The signs of Δ-ECDF(u) = ECDF(u) - u at u = 0.25 and u = 0.75 give the four canonical
    shapes: both positive (inverted U), both negative (U), negative then positive (S below
    0 first) and positive then negative (S above 0 first).

    Parameters
    ----------
    pit_values : array-like
        PIT values of a single variable, over all its observations.

    Returns
    -------
    str
        Short interpretation of the deviation.
    """
    pit_values = np.asarray(pit_values, dtype=float).ravel()
    n_obs = pit_values.size
    low = np.count_nonzero(pit_values <= 0.25) / n_obs - 0.25
    high = np.count_nonzero(pit_values <= 0.75) / n_obs - 0.75

    if low > 0 and high > 0:
        return (
            "the Δ-ECDF curve forms an inverted U (mostly above 0), so the observations lie "
            "below most predictions (overprediction)."
        )
    if low < 0 and high < 0:
        return (
            "the Δ-ECDF curve forms a U (mostly below 0), so the observations lie above "
            "most predictions (underprediction)."
        )
    if low < 0:
        return (
            "the Δ-ECDF curve forms an S (below 0 first, then above 0), so the predictions "
            "are too wide compared with the observations (underconfident)."
        )
    return (
        "the Δ-ECDF curve forms an S (above 0 first, then below 0), so the predictions "
        "are too narrow compared with the observations (overconfident)."
    )


def _pit_uniformity_check(
    pit_ds,
    *,
    method,
    alpha,
    gamma,
    max_points,
    show_diagnostics,
    return_diagnostics,
):
    """Run the PIT uniformity test on all variables and build the textual report."""
    messages = []
    diagnostics = {}
    has_errors = False

    for var, pit in pit_ds.data_vars.items():
        obs_dims = list(pit.dims)
        n_obs = int(np.prod([pit.sizes[dim] for dim in obs_dims], dtype=int))

        p_value_da, _, shapley_flat = pit.azstats.uniformity_test(dim=obs_dims, method=method)
        p_value = float(p_value_da)
        shapley = xr.DataArray(
            shapley_flat.values.reshape(pit.shape),
            dims=obs_dims,
            coords=pit.coords,
            name=var,
        )
        suspicious = (shapley > gamma) & (p_value < alpha)
        top_points = _top_points(pit, shapley, suspicious, max_points)
        flagged = p_value < alpha
        if flagged:
            has_errors = True

        diagnostics[var] = {
            "p_value": p_value,
            "alpha": alpha,
            "method": method,
            "n_obs": n_obs,
            "pit": pit,
            "shapley": shapley,
            "suspicious": suspicious,
            "top_points": top_points,
        }

        messages.append(f"p-value: {p_value:.4g}, alpha: {alpha:.4g}, observations: {n_obs}")
        if not flagged:
            messages.append("No deviation from uniformity detected")
            continue

        n_flagged = int(np.count_nonzero(np.asarray(suspicious)))
        messages.append("Deviation from uniformity detected")
        messages.append(f"{n_flagged} of {n_obs} observations flagged.")
        messages.append(f"Interpretation: {_interpret_pit(np.asarray(pit))}")
        if top_points:
            messages.append("Top flagged observations:")
            n_digits = len(str(n_obs))
            for point in top_points:
                coords_str = ", ".join(
                    f"{key} = {value:0{n_digits}d}"
                    if isinstance(value, int)
                    else f"{key} = {value}"
                    for key, value in point["coords"].items()
                )
                messages.append(f"  {coords_str}: percentile={point['pit']:>6.1%}")

    if show_diagnostics:
        print("\n".join(messages), file=sys.stdout)

    if return_diagnostics:
        return has_errors, diagnostics

    return has_errors


def get_suspicious_mask_ds(observed_dist, pit_dt, alpha, gamma, method):
    """Compute most suspicious observations based on PIT uniformity test results.

    Parameters
    ----------
    observed_dist : xarray.Dataset
        The observed data. Only used to select the variables to test.
    pit_dt : xarray.DataTree
        DataTree with an "ecdf_pit" group holding the PIT values, as returned by
        :func:`ppc_pit`.
    alpha : float
        Significance level of the uniformity test. Observations are only flagged when the
        p-value is below ``alpha``.
    gamma : float
        Minimum Shapley contribution for an observation to be flagged.
    method : {"pot_c", "prit_c", "piet_c"}
        Method used for the uniformity test.

    Returns
    -------
    xarray.Dataset
        Boolean mask, True for the observations flagged as suspicious.
    """
    pit_ds = pit_dt["ecdf_pit"].dataset[list(observed_dist.data_vars)]
    p_values, _, shapley_vals = pit_ds.azstats.uniformity_test(dim=pit_ds.dims, method=method)

    highlight = (shapley_vals > gamma) & (p_values < alpha)
    return xr.Dataset(
        {
            var: highlight[var].rename(
                {dim: next(iter(pit_ds[var].dims)) for dim in da.dims if "pit_dim" in dim}
            )
            for var, da in highlight.items()
        }
    )


def get_ppc_pit(predictive_dist, observed_dist, sample_dims, method):
    """Compute PIT values, with optional Pareto tail refinement.

    The probability of the posterior predictive being less than or equal to the observed data
    should be uniformly distributed. This function computes the PIT values with
    Generalized Pareto Distribution tail refinement.

    Parameters
    ----------
    predictive_dist : xarray.Dataset
        The posterior predictive distribution.
    observed_dist : xarray.Dataset
        The observed data.
    sample_dims : str or sequence of hashable, optional
        Dimensions to reduce.
    method : {"pot_c", "prit_c", "piet_c"}
        The method to use for PIT computation.
    """
    rng = np.random.default_rng(214)

    pareto_pit = method in ("pot_c", "piet_c")

    dictio = {}
    for var in observed_dist.data_vars:
        if pareto_pit:
            pred_stacked = predictive_dist[var].stack(__sample__=sample_dims)
            vals = xr.apply_ufunc(
                array_stats._pareto_pit_vec,  # pylint: disable=protected-access
                pred_stacked,
                observed_dist[var],
                input_core_dims=[["__sample__"], []],
                output_core_dims=[[]],
                vectorize=False,
                kwargs={"rng": rng},
            )
        else:
            n_below = (predictive_dist[var] < observed_dist[var]).sum(sample_dims)
            n_equal = (predictive_dist[var] == observed_dist[var]).sum(sample_dims)
            n_samples = int(np.prod([predictive_dist[var].sizes[dim] for dim in sample_dims]))
            vals = (n_below + (n_equal + 1) * rng.uniform(size=n_below.values.shape)) / (
                n_samples + 1
            )

        dictio[var] = vals

    return xr.DataTree.from_dict({"ecdf_pit": xr.Dataset(dictio)})


def text_ppc_pit(
    data,
    *,
    var_names=None,
    filter_vars=None,
    coords=None,
    sample_dims=None,
    group="posterior_predictive",
    method="pot_c",
    envelope_prob=None,
    gamma=0,
    max_points=10,
    show_diagnostics=True,
    return_diagnostics=False,
):
    r"""Run posterior predictive checks based on the PIT Δ-ECDF uniformity test.

    For a calibrated model the Probability Integral Transform (PIT) values,
    :math:`p(\tilde{y}_i \le y_i \mid y)`, should be uniformly distributed, where
    :math:`y_i` is the observed data and :math:`\tilde y_i` the posterior predictive
    sample at index :math:`i`. This function computes the p-value of a uniformity test on
    the PIT values and reports the observations contributing the most to the deviation, as
    described in [1]_ and [2]_.

    A variable is reported as deviating from uniformity when its p-value is below
    ``alpha = 1 - envelope_prob``. Within such a variable, the individual observations with
    a Shapley contribution above ``gamma`` are flagged as suspicious. The report includes a
    short interpretation of the deviation, based on the shape of the Δ-ECDF curve.

    For more details on how to interpret the results,
    see https://arviz-devs.github.io/EABM/Chapters/Prior_posterior_predictive_checks.html#pit-ecdfs.

    Parameters
    ----------
    data : DataTree or InferenceData-like
        Input data. It must contain the group to check (``group``, "posterior_predictive"
        by default) and the "observed_data" group.
    var_names : str or list of str, optional
        Names of the variables to check. If None, all variables are checked.
    filter_vars : {None, "like", "regex"}, default None
        How to filter variable names. See :func:`filter_vars` for details.
    coords : dict, optional
        Coordinates to select a subset of the data.
    sample_dims : iterable of hashable, optional
        Dimensions to be considered sample dimensions.
        Default from ``rcParams["data.sample_dims"]``.
    group : str, default "posterior_predictive"
        Group to compute the PIT values from. It can also be "prior_predictive".
    method : {"pot_c", "prit_c", "piet_c"}, default "pot_c"
        Method used for the uniformity test.
    envelope_prob : float, optional
        Probability used to compute the significance level ``alpha = 1 - envelope_prob``.
        Defaults to ``rcParams["stats.envelope_prob"]``.
    gamma : float, default 0
        Minimum Shapley contribution for an observation to be flagged as suspicious.
    max_points : int, default 10
        Maximum number of suspicious observations to report per variable.
    show_diagnostics : bool, default True
        If True, print the diagnostic messages to stdout.
    return_diagnostics : bool, default False
        If True, return a dictionary with detailed results in addition to the boolean
        has_errors flag.

    Returns
    -------
    has_errors : bool
        True if any variable has a p-value below alpha, False otherwise.
    diagnostics : dict, optional
        Only returned if ``return_diagnostics=True``. Mapping of variable name to a dict with:

        - "p_value": float, p-value of the uniformity test
        - "alpha": float, significance level used to flag observations
        - "method": str, method used for the uniformity test
        - "n_obs": int, number of observations tested
        - "pit": xarray.DataArray with the PIT values, on the observation dimensions
        - "shapley": xarray.DataArray with the unsorted Shapley contributions, on the
          observation dimensions
        - "suspicious": xarray.DataArray of bool, True for flagged observations
        - "top_points": list of dicts with "coords", "pit" and "shapley" of the most
          suspicious observations, sorted by decreasing Shapley contribution and limited
          to ``max_points`` entries

    See Also
    --------
    text_loo_pit : Textual PIT uniformity check for leave-one-out cross-validation.
    ppc_pit : Compute posterior predictive check (PPC) PIT values.
    loo_pit : Compute leave one out (PSIS-LOO) probability integral transform (PIT) values.

    Examples
    --------
    Print the PPC PIT diagnostics for the radon dataset:

    .. ipython::

        In [1]: from arviz_base import load_arviz_data
           ...: import arviz_stats as azs
           ...: data = load_arviz_data("radon")
           ...: azs.text_ppc_pit(data)

    Get the diagnostics as a dictionary, without printing anything:

    .. ipython::

        In [2]: has_errors, diagnostics = azs.text_ppc_pit(
           ...:     data, var_names="obs", show_diagnostics=False, return_diagnostics=True
           ...: )
           ...: has_errors, diagnostics["obs"]["p_value"]

    References
    ----------
    .. [1] Tesso et al. *LOO-PIT predictive model checking* arXiv:2603.02928 (2026).

    .. [2] Säilynoja et al. *Graphical test for discrete uniformity and its applications in
        goodness-of-fit evaluation and multiple sample comparison*. Statistics and Computing
        32(32). (2022) https://doi.org/10.1007/s11222-022-10090-6
    """
    _validate_method(method)
    alpha = 1 - validate_or_use_rcparam(envelope_prob, "stats.envelope_prob")

    dt = convert_to_datatree(data)
    predictive_dist, observed_dist = _process_groups(dt, group, var_names, filter_vars, coords)

    warn_if_binary(observed_dist, predictive_dist)
    warn_if_prior_predictive(group)

    sample_dims = validate_sample_dims(sample_dims, data=predictive_dist)
    pit_dt = get_ppc_pit(predictive_dist, observed_dist, sample_dims=sample_dims, method=method)

    return _pit_uniformity_check(
        pit_dt["ecdf_pit"].dataset,
        method=method,
        alpha=alpha,
        gamma=gamma,
        max_points=max_points,
        show_diagnostics=show_diagnostics,
        return_diagnostics=return_diagnostics,
    )


def text_loo_pit(
    data,
    *,
    var_names=None,
    method="pot_c",
    envelope_prob=None,
    gamma=0,
    max_points=10,
    show_diagnostics=True,
    return_diagnostics=False,
):
    r"""Run predictive checks based on the LOO-PIT Δ-ECDF uniformity test.

    For a calibrated model the LOO Probability Integral Transform (PIT) values,
    :math:`p(\tilde{y}_i \le y_i \mid y_{-i})`, should be uniformly distributed, where
    :math:`y_i` is the observed data and :math:`\tilde y_i` the posterior predictive
    sample at index :math:`i`. LOO-PIT values are computed using the PSIS-LOO-CV method
    described in [1]_ and [2]_.

    This function applies the same uniformity test, reporting and interpretation as
    :func:`text_ppc_pit` to the LOO-PIT values, following [3]_.

    Parameters
    ----------
    data : DataTree or InferenceData-like
        Input data. It should contain posterior, posterior_predictive and log_likelihood
        groups.
    var_names : str or list of str, optional
        Names of the variables to check. If None, all variables are checked.
    method : {"pot_c", "prit_c", "piet_c"}, default "pot_c"
        Method used for the uniformity test. "pot_c" and "piet_c" compute the LOO-PIT
        values with Pareto tail refinement, "prit_c" without it.
    envelope_prob : float, optional
        Probability used to compute the significance level ``alpha = 1 - envelope_prob``.
        Defaults to ``rcParams["stats.envelope_prob"]``.
    gamma : float, default 0
        Minimum Shapley contribution for an observation to be flagged as suspicious.
    max_points : int, default 10
        Maximum number of suspicious observations to report per variable.
    show_diagnostics : bool, default True
        If True, print the diagnostic messages to stdout.
    return_diagnostics : bool, default False
        If True, return a dictionary with detailed results in addition to the boolean
        has_errors flag.

    Returns
    -------
    has_errors : bool
        True if any variable has a p-value below alpha, False otherwise.
    diagnostics : dict, optional
        Only returned if ``return_diagnostics=True``. Same structure as the diagnostics
        returned by :func:`text_ppc_pit`.

    See Also
    --------
    text_ppc_pit : Text-based PIT uniformity check for posterior predictive values.
    plot_loo_pit : Plot-based LOO-PIT uniformity check for posterior predictive values.

    Examples
    --------
    Print the LOO-PIT diagnostics for the radon dataset:

    .. ipython::

        In [1]: from arviz_base import load_arviz_data
           ...: import arviz_stats as azs
           ...: data = load_arviz_data("radon")
           ...: azs.text_loo_pit(data)

    Get the diagnostics as a dictionary, without printing anything:

    .. ipython::

        In [2]: has_errors, diagnostics = azs.text_loo_pit(
           ...:     data, show_diagnostics=False, return_diagnostics=True
           ...: )
           ...: diagnostics["obs"]["n_obs"]

    References
    ----------
    .. [1] Vehtari et al. *Practical Bayesian model evaluation using leave-one-out
        cross-validation and WAIC*. Statistics and Computing. 27(5) (2017)
        https://doi.org/10.1007/s11222-016-9696-4

    .. [2] Vehtari et al. *Pareto Smoothed Importance Sampling*. Journal of Machine Learning
        Research, 25(72) (2024) https://jmlr.org/papers/v25/19-556.html

    .. [3] Tesso et al. *LOO-PIT predictive model checking* arXiv:2603.02928 (2026).
    """
    _validate_method(method)
    alpha = 1 - validate_or_use_rcparam(envelope_prob, "stats.envelope_prob")
    pareto_pit = method in ("pot_c", "piet_c")

    dt = convert_to_datatree(data)
    observed_dist = dt["observed_data"]
    observed_dist = observed_dist.azstats.filter_vars(var_names=var_names).dataset
    warn_if_binary(observed_dist, None)

    pit_ds = loo_pit(data, var_names=var_names, pareto_pit=pareto_pit)

    return _pit_uniformity_check(
        pit_ds,
        method=method,
        alpha=alpha,
        gamma=gamma,
        max_points=max_points,
        show_diagnostics=show_diagnostics,
        return_diagnostics=return_diagnostics,
    )
