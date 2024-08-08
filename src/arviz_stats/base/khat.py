import warnings
import numpy as np
from arviz import ess

def pareto_khat(x, r_eff=None, tail="both", log_weights=False):
    """
    
    parameters
    ----------
    x : DataArray
    """
    ary = x.values.flatten()

    if log_weights:
        tail = "right"

    ndraws = len(ary)

    if r_eff is None:
        r_eff = ess(x.values, method="tail") / ndraws

    if ndraws > 255:
        ndraws_tail = np.ceil(3 * (ndraws / r_eff)**0.5).astype(int)
    else:
        ndraws_tail = int(ndraws / 5)

    if tail == "both":
        if ndraws_tail > ndraws / 2:
            warnings.warn("Number of tail draws cannot be more than half "
                   "the total number of draws if both tails are fit, "
                   f"changing to {ndraws / 2}")
            ndraws_tail = ndraws / 2
        

        if ndraws_tail < 5:
            warnings.warn("Number of tail draws cannot be less than 5. "
                          "Changing to 5")
            ndraws_tail = 5

        k = max([pareto_smooth_tail(ary, ndraws, ndraws_tail, smooth_draws=False, tail=t)[1] for t in ("left", "right")])
    else:
        _, k = pareto_smooth_tail(ary, ndraws, ndraws_tail, smooth_draws=False, tail=tail)


    return k

def ps_min_ss(k):
    if k < 1:
        return 10**(1 / (1 - max(0, k)))
    else:
        return np.inf

def pareto_smooth_tail(x, ndraws, ndraws_tail, smooth_draws=False, tail='both', log_weights=False):
    if log_weights:
        x = x - np.max(x)

    if tail not in ['right', 'left', 'both']:
        raise ValueError('tail must be one of "right", "left", or "both"')

    tail_ids = np.arange(ndraws - ndraws_tail, ndraws)

    if tail == 'left':
        x = -x

    ordered = np.argsort(x)
    draws_tail = x[ordered[tail_ids]]

    cutoff = x[ordered[tail_ids[0] - 1]]  # largest value smaller than tail values

    max_tail = np.max(draws_tail)
    min_tail = np.min(draws_tail)

    if ndraws_tail >= 5:
        if abs(max_tail - min_tail) < np.finfo(float).tiny:
            raise ValueError('All tail values are the same')

        if log_weights:
            draws_tail = np.exp(draws_tail)
            cutoff = np.exp(cutoff)

        k, sigma = _gpdfit(draws_tail - cutoff)

        if np.isfinite(k) and smooth_draws:
            p = (np.arange(0.5, ndraws_tail)) / ndraws_tail
            smoothed = _gpinv(p, k, sigma, cutoff)
        
            if log_weights:
                smoothed = np.log(smoothed)
        else:
            smoothed = None
    else:
        raise ValueError('ndraws_tail must be at least 5')
    
    if smoothed is not None:
        smoothed[smoothed > max_tail] = max_tail
        x[ordered[tail_ids]] = smoothed 

    if tail == 'left':
        x = -x
    
    return x, k


def _gpdfit(ary):
    """Estimate the parameters for the Generalized Pareto Distribution (GPD).

    Empirical Bayes estimate for the parameters of the generalized Pareto
    distribution given the data.

    Parameters
    ----------
    ary: array
        sorted 1D data array

    Returns
    -------
    k: float
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
    k_post = np.log1p(-b_post * ary).mean()  # pylint: disable=invalid-unary-operand-type,no-member
    # add prior for k_post
    sigma = -k_post / b_post
    k_post = (n * k_post + prior_k * 0.5) / (n + prior_k)

    return k_post, sigma

def _gpinv(probs, kappa, sigma, mu):
    """
    """
    if sigma <= 0:
        return np.full_like(probs, np.nan)

    probs = 1 - probs
    if kappa == 0:
        q = mu - sigma * np.log1p(-probs)
    else:
        q = mu + sigma * np.expm1(-kappa * np.log1p(-probs)) / kappa

    return q


