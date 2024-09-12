"""Pareto k-hat diagnostics."""

import warnings

import numpy as np


def pareto_khat(ary, r_eff=1, tail="both", log_weights=False):
    """
    Compute Pareto k-hat diagnostic.

    See details in Vehtari et al., 2024 (https://doi.org/10.48550/arXiv.1507.02646)

    Parameters
    ----------
    ary : Array
    r_eff : float, optional
        Relative efficiency. Effective sample size divided the number of samples.
    tail : srt, optional
        Which tail to fit. Can be 'right', 'left', or 'both'.
    log_weights : bool, optional
        Whether dt represents log-weights.

    Returns
    -------
    khat : float
        Pareto k-hat value.
    """
    if log_weights:
        tail = "right"

    n_draws = len(ary)

    if n_draws > 255:
        n_draws_tail = np.ceil(3 * (n_draws / r_eff) ** 0.5).astype(int)
    else:
        n_draws_tail = int(n_draws / 5)

    if tail == "both":
        if n_draws_tail > n_draws / 2:
            warnings.warn(
                "Number of tail draws cannot be more than half "
                "the total number of draws if both tails are fit, "
                f"changing to {n_draws / 2}"
            )
            n_draws_tail = n_draws / 2

        if n_draws_tail < 5:
            warnings.warn("Number of tail draws cannot be less than 5. Changing to 5")
            n_draws_tail = 5

        khat = max(
            ps_tail(ary, n_draws, n_draws_tail, smooth_draws=False, tail=t)[1]
            for t in ("left", "right")
        )
    else:
        _, khat = ps_tail(ary, n_draws, n_draws_tail, smooth_draws=False, tail=tail)

    return khat


def ps_tail(ary, n_draws, n_draws_tail, smooth_draws=False, tail="both", log_weights=False):
    """
    Estimate the tail of a distribution using the Generalized Pareto Distribution.

    Parameters
    ----------
    x : array
        1D array.
    n_draws : int
        Number of draws.
    n_draws_tail : int
        Number of draws in the tail.
    smooth_draws : bool, optional
        Whether to smooth the tail.
    tail : str, optional
        Which tail to fit. Can be 'right', 'left', or 'both'.
    log_weights : bool, optional
        Whether x represents log-weights.

    Returns
    -------
    ary : array
        Array with smoothed tail values.
    k : float
        Estimated shape parameter.
    """
    if log_weights:
        ary = ary - np.max(ary)

    if tail not in ["right", "left", "both"]:
        raise ValueError('tail must be one of "right", "left", or "both"')

    tail_ids = np.arange(n_draws - n_draws_tail, n_draws)

    if tail == "left":
        ary = -ary

    ordered = np.argsort(ary)
    draws_tail = ary[ordered[tail_ids]]

    cutoff = ary[ordered[tail_ids[0] - 1]]  # largest value smaller than tail values

    max_tail = np.max(draws_tail)
    min_tail = np.min(draws_tail)

    if n_draws_tail >= 5:
        if abs(max_tail - min_tail) < np.finfo(float).tiny:
            raise ValueError("All tail values are the same")

        if log_weights:
            draws_tail = np.exp(draws_tail)
            cutoff = np.exp(cutoff)

        khat, sigma = _gpdfit(draws_tail - cutoff)

        if np.isfinite(khat) and smooth_draws:
            p = (np.arange(0.5, n_draws_tail)) / n_draws_tail
            smoothed = _gpinv(p, khat, sigma, cutoff)

            if log_weights:
                smoothed = np.log(smoothed)
        else:
            smoothed = None
    else:
        raise ValueError("n_draws_tail must be at least 5")

    if smoothed is not None:
        smoothed[smoothed > max_tail] = max_tail
        ary[ordered[tail_ids]] = smoothed

    if tail == "left":
        ary = -ary

    return ary, khat


def _gpdfit(ary):
    """Estimate the parameters for the Generalized Pareto Distribution (GPD).

    Empirical Bayes estimate for the parameters (kappa, sigma) of the generalized Pareto
    distribution given the data.

    The fit uses a prior for kappa to stabilize estimates for very small (effective)
    sample sizes. The weakly informative prior is a Gaussian  centered at 0.5.
    See details in Vehtari et al., 2024 (https://doi.org/10.48550/arXiv.1507.02646)


    Parameters
    ----------
    ary: array
        sorted 1D data array

    Returns
    -------
    kappa: float
        estimated shape parameter
    sigma: float
        estimated scale parameter
    """
    prior_bs = 3
    prior_k = 10
    n = len(ary)
    m_est = 30 + int(n**0.5)

    b_ary = 1 - np.sqrt(m_est / (np.arange(1, m_est + 1, dtype=float) - 0.5))
    b_ary /= prior_bs * ary[int(n / 4 + 0.5) - 1]
    b_ary += 1 / ary[-1]

    k_ary = np.log1p(-b_ary[:, None] * ary).mean(axis=1)  # pylint: disable=no-member
    len_scale = n * (np.log(-(b_ary / k_ary)) - k_ary - 1)
    weights = 1 / np.exp(len_scale - len_scale[:, None]).sum(axis=1)

    # remove negligible weights
    real_idxs = weights >= 10 * np.finfo(float).eps
    if not np.all(real_idxs):
        weights = weights[real_idxs]
        b_ary = b_ary[real_idxs]
    # normalise weights
    weights /= weights.sum()

    # posterior mean for b
    b_post = np.sum(b_ary * weights)
    # estimate for k
    kappa = np.log1p(-b_post * ary).mean()  # pylint: disable=invalid-unary-operand-type,no-member
    # add prior for kappa
    sigma = -kappa / b_post
    kappa = (n * kappa + prior_k * 0.5) / (n + prior_k)

    return kappa, sigma


def _gpinv(probs, kappa, sigma, mu):
    """Quantile function for generalized pareto distribution."""
    if sigma <= 0:
        return np.full_like(probs, np.nan)

    probs = 1 - probs
    if kappa == 0:
        q = mu - sigma * np.log1p(-probs)
    else:
        q = mu + sigma * np.expm1(-kappa * np.log1p(-probs)) / kappa

    return q
