"""Utility functions for computing the PIT ECDF and the simultaneous confidence bands."""

import math

import numpy as np
from arviz_base import dict_to_dataset
from numba import vectorize
from scipy.optimize import minimize_scalar
from scipy.special import bdtr, bdtrik  # pylint: disable=no-name-in-module
from scipy.stats import uniform


def difference_ecdf_pit(dt, data_pairs, ci_prob, randomized, method, n_simulations):
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
    ci_prob : float, optional
        The probability for the credible interval.
    randomized : bool
        Whether to randomize the PIT values. Randomization is needed for discrete data.
    n_simulations : int
        The number of simulations to use with method `simulation`.
    method : str
        The method to use to compute the confidence bands. Either `simulation` or `optimized`.
    """
    dictio = {}
    rng = np.random.default_rng(214)

    for var_predictive, var_obs in data_pairs.items():
        vals = (dt.posterior_predictive[var_predictive] <= dt.observed_data[var_obs]).mean(
            ("chain", "draw")
        )
        if randomized:
            vals_less = (dt.posterior_predictive[var_predictive] < dt.observed_data[var_obs]).mean(
                ("chain", "draw")
            )

            urvs = rng.uniform(size=vals.shape)
            vals = urvs * vals_less + (1 - urvs) * vals

        eval_points, ecdf, ci_lb, ci_ub = ecdf_pit(vals, ci_prob, method, n_simulations, rng)
        dictio[var_predictive] = np.stack(
            [eval_points, ecdf - eval_points, ci_lb - eval_points, ci_ub - eval_points]
        )

    return (
        dict_to_dataset(dictio)
        .rename({"draw": "x_values", "chain": "plot_axis"})
        .assign_coords({"plot_axis": ["x", "y", "y_bottom", "y_top"]})
    )


def ecdf_pit(vals, ci_prob, method, n_simulations, rng=None):
    """
    Compute the PIT ECDF and the pointwise confidence bands.

    Parameters
    ----------
    vals : array-like
        The values to compute the PIT ECDF.
    ci_prob : float, optional
        The probability for the credible interval.
    method : str
        The method to use to compute the confidence bands. Either `simulation` or `optimized`.
    n_simulations : int
        The number of simulations to use with method `simulation`.
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
    n_draws = len(vals)
    eval_points = np.linspace(1 / n_draws, 1, n_draws)
    vals = np.sort(vals)
    ecdf = compute_ecdf(vals, eval_points)
    cdf_at_eval_points = uniform(0, 1).cdf(eval_points)

    if method == "simulation":
        prob_pointwise = simulate_confidence_bands(
            n_draws,
            eval_points,
            cdf_at_eval_points,
            ci_prob,
            n_simulations,
            rng,
        )

    elif method == "optimized":
        prob_pointwise = optimize_confidence_bands(n_draws, cdf_at_eval_points, ci_prob)
    else:
        raise ValueError(f"Invalid method: {method}")

    lower, upper = get_pointwise_confidence_band(prob_pointwise, n_draws, cdf_at_eval_points)

    return eval_points, ecdf, lower, upper


def compute_ecdf(sample, eval_points):
    """Compute the empirical CDF at the evaluation points."""
    sample = np.sort(sample)
    return np.searchsorted(sample, eval_points, side="right") / len(sample)


def get_pointwise_confidence_band(prob, ndraws, cdf_at_eval_points):
    """Compute the `prob`-level pointwise confidence band."""
    lower_ci = (1 - prob) / 2
    upper_ci = 1 - lower_ci

    # We use the bdtrik function instead of instantiating a binomial distribution
    # and computing the quantiles because it is faster
    count_lower = np.ceil(bdtrik(lower_ci, ndraws, cdf_at_eval_points))
    count_upper = np.ceil(bdtrik(upper_ci, ndraws, cdf_at_eval_points))

    prob_lower = count_lower / ndraws
    prob_upper = count_upper / ndraws
    return prob_lower, prob_upper


### Simulation method
def simulate_confidence_bands(n_draws, eval_points, cdf_at_eval_points, prob, n_simulations, rng):
    """Simulate method for simultaneous confidence bands.

    Compute the smallest marginal probability of a pointwise confidence band that
    contains the ECDF using the simulation method.

    Parameters
    ----------
    n_draws : int
        The number of draws.
    eval_points : array-like
        The evaluation points.
    cdf_at_eval_points : array-like
        The CDF at the evaluation points.
    prob : float
        The probability for the credible interval.
    num_trials : int
        The number of trials to use.
    rng : Generator
        The random number generator to use.
    """
    probs_pointwise = np.empty(n_simulations)
    for i in range(n_simulations):
        sample = rng.uniform(0, 1, size=n_draws)
        ecdf_at_eval_points = compute_ecdf(sample, eval_points)
        prob_pointwise = fit_pointwise_band_probability(
            n_draws, ecdf_at_eval_points, cdf_at_eval_points
        )
        probs_pointwise[i] = prob_pointwise

    return np.quantile(probs_pointwise, prob)


def fit_pointwise_band_probability(
    n_draws,
    ecdf_at_eval_points,
    cdf_at_eval_points,
):
    """Return the smallest marginal prob of confidence band that contains the ECDF.

    Parameters
    ----------
    n_draws : int
        The number of draws.
    ecdf_at_eval_points : array-like
        The ECDF at the evaluation points.
    cdf_at_eval_points : array-like
        The CDF at the evaluation points.
    """
    ecdf_scaled = n_draws * ecdf_at_eval_points
    # We use the bdtr function instead of instantiating a binomial distribution
    # and computing the cdf because it is faster
    prob_lower_tail = np.min(bdtr(ecdf_scaled, n_draws, cdf_at_eval_points))
    prob_upper_tail = np.min(1 - bdtr(ecdf_scaled - 1, n_draws, cdf_at_eval_points))
    prob_pointwise = 1 - 2 * min(prob_lower_tail, prob_upper_tail)
    return prob_pointwise


### Optimization method
### The optimization method should be faster than the simulation method
### but it is not the case with the current implementation
### Leave it here for future reference
def optimize_confidence_bands(ndraws, cdf_at_eval_points, prob):
    """Estimate probability for simultaneous confidence band using optimization.

    This function simulates the pointwise probability needed to construct pointwise confidence bands
    that form a `prob`-level confidence envelope for the ECDF of a sample.
    """
    # cdf_at_eval_points = np.unique(cdf_at_eval_points)
    prob_pointwise = minimize_scalar(
        ecdf_band_optimization_objective,
        bounds=(prob, 1),
        method="bounded",
        options={"xatol": 0.001},
        args=(cdf_at_eval_points, ndraws, prob),
    ).x
    return prob_pointwise


@vectorize()
def binomial_pmf(k, n, p, loc):
    """Compute the binomial probability mass function."""
    k -= loc
    if k < 0 or k > n:
        return 0.0
    if p == 0:
        return 1.0 if k == 0 else 0.0
    if p == 1:
        return 1.0 if k == n else 0.0
    if k == 0:
        return (1 - p) ** n
    if k == n:
        return p**n
    lbinom = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    return np.exp(lbinom + k * np.log(p) + (n - k) * np.log1p(-p))


def ecdf_band_optimization_objective(prob_pointwise, cdf_at_eval_points, ndraws, prob_target):
    """Objective function for optimizing the simultaneous confidence band probability."""
    lower, upper = get_pointwise_confidence_band(prob_pointwise, ndraws, cdf_at_eval_points)
    lower_count = (lower * ndraws).astype(int)
    upper_count = (upper * ndraws).astype(int) + 1
    cdf_with_zero = np.insert(cdf_at_eval_points[:-1], 0, 0)
    prob_between_points = (cdf_at_eval_points - cdf_with_zero) / (1 - cdf_with_zero)
    prob_interior = ecdf_band_interior_probability(
        prob_between_points, ndraws, lower_count, upper_count
    )
    return abs(prob_interior - prob_target)


def ecdf_band_interior_probability(prob_between_points, ndraws, lower_count, upper_count):
    """Compute the probability of the ECDF being inside the pointwise confidence bands."""
    interval_left = np.zeros(1)
    prob_interior = np.ones(1)
    for i in range(prob_between_points.shape[0]):
        interval_right = np.arange(lower_count[i], upper_count[i])
        prob_interior = update_ecdf_band_interior_probabilities(
            prob_interior, interval_left, interval_right, prob_between_points[i], ndraws
        )
        interval_left = interval_right
    return prob_interior.sum()


def update_ecdf_band_interior_probabilities(prob_left, interval_left, interval_right, p, ndraws):
    """Update the probability of the ECDF being inside the pointwise confidence bands."""
    interval_left = interval_left[:, None]
    prob_conditional = binomial_pmf(interval_right, ndraws - interval_left, p, interval_left)
    prob_right = np.dot(prob_left, prob_conditional)
    return prob_right
