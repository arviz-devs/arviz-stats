"""Utility functions for computing the PIT ECDF and the simultaneous confidence bands."""

import math
import warnings

import numpy as np
from arviz_base import dict_to_dataset
from scipy.special import bdtr, bdtrik  # pylint: disable=no-name-in-module

try:
    from numba import vectorize

    NUMBA_FLAG = True
except ImportError:
    NUMBA_FLAG = False


def difference_ecdf_pit(dt, data_pairs, group, ci_prob, coverage, randomized, n_simulations):
    """Compute the difference PIT ECDF values.

    The probability of the posterior predictive being less than or equal to the observed data.
    should be uniformly distributed. This function computes the PIT ECDF and the simultaneous
    confidence bands.

    Parameters
    ----------
    dt: DataTree
        DataTree with "posterior_predictive" and "observed_data" groups
    data_pairs : tuple
        Tuple with first element contains the posterior predictive name (or list of names)
        and the second element contains the observed data variable name (or list of names).
    group : str
        The group from which to get the unique values.
    ci_prob : float, optional
        The probability for the credible interval.
    randomized : list of bool
        Whether to randomize the PIT values. Randomization is needed for discrete data.
    n_simulations : int
        The number of simulations to use with method `simulation`.
    """
    dictio = {}
    rng = np.random.default_rng(214)

    for idx, (var_predictive, var_obs) in enumerate(data_pairs.items()):
        vals = (dt[group][var_predictive] <= dt.observed_data[var_obs]).mean(("chain", "draw"))
        if randomized[idx]:
            vals_less = (dt[group][var_predictive] < dt.observed_data[var_obs]).mean(
                ("chain", "draw")
            )

            urvs = rng.uniform(size=vals.shape)
            vals = urvs * vals_less + (1 - urvs) * vals

        if coverage:
            vals = 2 * np.abs(vals - 0.5)

        eval_points, ecdf, ci_lb, ci_ub = ecdf_pit(vals, ci_prob, n_simulations, rng=rng)
        dictio[var_predictive] = np.stack(
            [eval_points, ecdf - eval_points, ci_lb - eval_points, ci_ub - eval_points]
        )

    return (
        dict_to_dataset(dictio)
        .rename({"draw": "x_values", "chain": "plot_axis"})
        .assign_coords({"plot_axis": ["x", "y", "y_bottom", "y_top"]})
    )


def ecdf_pit(vals, ci_prob, n_simulations, n_chains=1, rng=None):
    """
    Compute the PIT ECDF and the pointwise confidence bands.

    Parameters
    ----------
    vals : array-like
        The values to compute the PIT ECDF.
    ci_prob : float, optional
        The probability for the credible interval.
    n_simulations : int
        The number of simulations to use with method `simulation`.
    n_chains : int, optional
        The number of chains.
    rng : Generator, optional
        The random number generator to use with the simulation method.

    Returns
    -------
    eval_points : array-like
        The evaluation points.
    ecdf : array-like
        The PIT ECDF.
    lower : array-like
        The lower confidence band.
    upper : array-like
        The upper confidence band.
    """
    if rng is None:
        rng = np.random.default_rng(214)

    n_draws = len(vals)
    eval_points = np.arange(1, n_draws) / n_draws
    ecdf = compute_ecdf(vals, eval_points)

    prob_pointwise = simulate_confidence_bands(
        n_draws,
        n_chains,
        eval_points,
        ci_prob,
        n_simulations,
        rng,
    )

    lower, upper = get_pointwise_confidence_band(prob_pointwise, n_draws, eval_points)

    return eval_points, ecdf, lower, upper


def compute_ecdf(sample, eval_points):
    """Compute the empirical CDF at the evaluation points."""
    sample = np.sort(sample)
    return np.searchsorted(sample, eval_points, side="right") / len(sample)


def get_pointwise_confidence_band(prob, ndraws, eval_points):
    """Compute the `prob`-level pointwise confidence band."""
    lower_ci = (1 - prob) / 2
    upper_ci = 1 - lower_ci
    # We use the bdtrik function instead of instantiating a binomial distribution
    # and computing the quantiles because it is faster
    count_lower = np.ceil(bdtrik(lower_ci, ndraws, eval_points))
    count_upper = np.ceil(bdtrik(upper_ci, ndraws, eval_points))

    prob_lower = count_lower / ndraws
    prob_upper = count_upper / ndraws
    return prob_lower, prob_upper


def simulate_confidence_bands(n_draws, n_chains, eval_points, prob, n_simulations, rng):
    """Simulate method for simultaneous confidence bands.

    Compute the smallest marginal probability of a pointwise confidence band that
    contains the ECDF using the simulation method.

    Parameters
    ----------
    n_draws : int
        The total number of draws.
    n_chains : int
        The number of chains.
    eval_points : array-like
        The evaluation points.
    prob : float
        The probability for the credible interval.
    n_simulations : int
        The number of trials to use.
    rng : Generator
        The random number generator to use.
    """
    if n_chains > 1:
        total_draws = n_draws * n_chains
        evaluated_counts = np.linspace(1, total_draws, n_draws - 1, dtype=int, endpoint=False)
        if NUMBA_FLAG:
            func = hypergeometric_cdf
        else:
            warnings.warn(
                "Numba is not available, using slower implementation for hypergeometric CDF."
                " Please consider installing numba for better performance.",
            )
            from scipy.special._ufuncs import _hypergeom_cdf

            def func(x, population, draws, successes):
                return _hypergeom_cdf(x, draws, successes, population)

        gamma = np.empty(n_simulations)
        for i in range(n_simulations):
            gamma_chains = np.empty(n_chains)
            for j in range(n_chains):
                sample = rng.uniform(0, 1, size=n_draws)
                ecdf_at_eval_points = (compute_ecdf(sample, eval_points) * n_draws).astype(int)
                prob_lower_tail = np.nanmin(
                    func(ecdf_at_eval_points, total_draws, n_draws, evaluated_counts)
                )
                prob_upper_tail = np.nanmin(
                    1 - func(ecdf_at_eval_points - 1, total_draws, n_draws, evaluated_counts)
                )
                gamma_chains[j] = 1 - 2 * min(prob_lower_tail, prob_upper_tail)

            gamma[i] = np.min(gamma_chains)

    else:
        gamma = np.empty(n_simulations)
        for i in range(n_simulations):
            sample = rng.uniform(0, 1, size=n_draws)
            ecdf_at_eval_points = (compute_ecdf(sample, eval_points) * n_draws).astype(int)
            prob_lower_tail = np.min(bdtr(ecdf_at_eval_points, n_draws, eval_points))
            prob_upper_tail = np.min(1 - bdtr(ecdf_at_eval_points - 1, n_draws, eval_points))
            gamma[i] = 1 - 2 * min(prob_lower_tail, prob_upper_tail)

    return np.quantile(gamma, prob)


@vectorize()
def ln_binomial(n, k):
    """Compute the natural logarithm of the binomial coefficient."""
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


@vectorize(["float64(int64, int64, int64, int64)"], target="parallel", cache=True)
def hypergeometric_cdf(x_val, population, draws, successes):
    """Compute the hypergeometric cumulative distribution function."""
    k_min = max(0, draws + successes - population)
    k_max = min(successes, draws)

    if x_val < k_min:
        return 0.0
    if x_val >= k_max:
        return 1.0

    log_p = (
        ln_binomial(successes, k_min)
        + ln_binomial(population - successes, draws - k_min)
        - ln_binomial(population, draws)
    )
    log_cdf = log_p

    for k in range(k_min + 1, x_val + 1):
        denom = k * (population - successes - draws + k)
        if denom > 0:
            log_ratio = np.log(successes - k + 1) + np.log(draws - k + 1) - np.log(denom)
            log_p += log_ratio

        if log_cdf > log_p:
            log_cdf = log_cdf + np.log1p(np.exp(log_p - log_cdf))
        else:
            log_cdf = log_p + np.log1p(np.exp(log_cdf - log_p))

    cdf = np.exp(log_cdf)
    if cdf < 0.0:
        return 0.0
    if cdf > 1.0:
        return 1.0
    return cdf
