# pylint: disable=too-many-lines, too-many-function-args, redefined-outer-name
"""Diagnostic functions for ArviZ."""

import warnings
from collections.abc import Sequence

import numpy as np
from scipy import stats
from scipy.special import logsumexp

from arviz_stats.base.circular_utils import circular_diff, circular_mean, circular_sd, circular_var
from arviz_stats.base.core import _CoreBase
from arviz_stats.base.stats_utils import not_valid as _not_valid


class _DiagnosticsBase(_CoreBase):
    """Class with numpy.scipy only diagnostic related functions."""

    def _backtransform_ranks(self, arr, c=3 / 8):  # pylint: disable=invalid-name, no-self-use
        """Backtransformation of ranks.

        Parameters
        ----------
        arr : np.ndarray
            Ranks array
        c : float
            Fractional offset. Defaults to c = 3/8 as recommended by Blom (1958).

        Returns
        -------
        np.ndarray

        References
        ----------
        Blom, G. (1958). Statistical Estimates and Transformed Beta-Variables. Wiley; New York.
        """
        arr = np.asarray(arr)
        size = arr.size
        return (arr - c) / (size - 2 * c + 1)

    def _z_scale(self, ary):
        """Calculate z_scale.

        Parameters
        ----------
        ary : np.ndarray

        Returns
        -------
        np.ndarray
        """
        ary = np.asarray(ary)
        rank = stats.rankdata(ary, method="average", nan_policy="omit")
        rank = self._backtransform_ranks(rank)
        z = stats.norm.ppf(rank)
        z = z.reshape(ary.shape)
        return z

    def _split_chains(self, ary):  # pylint: disable=no-self-use
        """Split and stack chains."""
        ary = np.asarray(ary)
        if len(ary.shape) <= 1:
            ary = np.atleast_2d(ary)
        _, n_draw = ary.shape
        half = n_draw // 2
        return np.vstack((ary[:, :half], ary[:, -half:]))

    def _z_fold(self, ary):
        """Fold and z-scale values."""
        ary = np.asarray(ary)
        ary = abs(ary - np.median(ary))
        ary = self._z_scale(ary)
        return ary

    def _rhat(self, ary):  # pylint: disable=no-self-use
        """Compute the rhat for a 2d array."""
        ary = np.asarray(ary, dtype=float)
        _, num_samples = ary.shape

        # Calculate chain mean
        chain_mean = np.mean(ary, axis=1)
        # Calculate chain variance
        chain_var = np.var(ary, axis=1, ddof=1)
        # Calculate between-chain variance
        between_chain_variance = num_samples * np.var(chain_mean, axis=None, ddof=1)
        # Calculate within-chain variance
        within_chain_variance = np.mean(chain_var)
        # Estimate of marginal posterior variance
        rhat_value = np.sqrt(
            (between_chain_variance / within_chain_variance + num_samples - 1) / (num_samples)
        )
        return rhat_value

    def _rhat_rank(self, ary):
        """Compute the rank normalized rhat for 2d array.

        Computation follows https://arxiv.org/abs/1903.08008
        """
        ary = np.asarray(ary)
        if _not_valid(ary, shape_kwargs={"min_draws": 4, "min_chains": 2}):
            return np.nan
        split_ary = self._split_chains(ary)
        rhat_bulk = self._rhat(self._z_scale(split_ary))

        split_ary_folded = abs(split_ary - np.median(split_ary))
        rhat_tail = self._rhat(self._z_scale(split_ary_folded))

        rhat_rank = max(rhat_bulk, rhat_tail)
        return rhat_rank

    def _rhat_folded(self, ary):
        """Calculate split-Rhat for folded z-values."""
        ary = np.asarray(ary)
        if _not_valid(ary, shape_kwargs={"min_draws": 4, "min_chains": 2}):
            return np.nan
        ary = self._z_fold(self._split_chains(ary))
        return self._rhat(ary)

    def _rhat_z_scale(self, ary):
        ary = np.asarray(ary)
        if _not_valid(ary, shape_kwargs={"min_draws": 4, "min_chains": 2}):
            return np.nan
        return self._rhat(self._z_scale(self._split_chains(ary)))

    def _rhat_split(self, ary):
        ary = np.asarray(ary)
        if _not_valid(ary, shape_kwargs={"min_draws": 4, "min_chains": 2}):
            return np.nan
        return self._rhat(self._split_chains(ary))

    def _rhat_identity(self, ary):
        ary = np.asarray(ary)
        if _not_valid(ary, shape_kwargs={"min_draws": 4, "min_chains": 2}):
            return np.nan
        return self._rhat(ary)

    def _ess(self, ary, relative=False):
        """Compute the effective sample size for a 2D array."""
        ary = np.asarray(ary, dtype=float)
        if (np.max(ary) - np.min(ary)) < np.finfo(float).resolution:  # pylint: disable=no-member
            return ary.size
        n_chain, n_draw = ary.shape
        acov = self.autocov(ary, axis=1)
        chain_mean = ary.mean(axis=1)
        mean_var = np.mean(acov[:, 0]) * n_draw / (n_draw - 1.0)
        var_plus = mean_var * (n_draw - 1.0) / n_draw
        if n_chain > 1:
            var_plus += np.var(chain_mean, axis=None, ddof=1)

        rho_hat_t = np.zeros(n_draw)
        rho_hat_even = 1.0
        rho_hat_t[0] = rho_hat_even
        rho_hat_odd = 1.0 - (mean_var - np.mean(acov[:, 1])) / var_plus
        rho_hat_t[1] = rho_hat_odd

        # Geyer's initial positive sequence
        t = 1
        while t < (n_draw - 3) and (rho_hat_even + rho_hat_odd) > 0.0:
            rho_hat_even = 1.0 - (mean_var - np.mean(acov[:, t + 1])) / var_plus
            rho_hat_odd = 1.0 - (mean_var - np.mean(acov[:, t + 2])) / var_plus
            if (rho_hat_even + rho_hat_odd) >= 0:
                rho_hat_t[t + 1] = rho_hat_even
                rho_hat_t[t + 2] = rho_hat_odd
            t += 2

        max_t = t - 2
        # improve estimation
        if rho_hat_even > 0:
            rho_hat_t[max_t + 1] = rho_hat_even
        # Geyer's initial monotone sequence
        t = 1
        while t <= max_t - 2:
            if (rho_hat_t[t + 1] + rho_hat_t[t + 2]) > (rho_hat_t[t - 1] + rho_hat_t[t]):
                rho_hat_t[t + 1] = (rho_hat_t[t - 1] + rho_hat_t[t]) / 2.0
                rho_hat_t[t + 2] = rho_hat_t[t + 1]
            t += 2

        ess = n_chain * n_draw
        tau_hat = (
            -1.0 + 2.0 * np.sum(rho_hat_t[: max_t + 1]) + np.sum(rho_hat_t[max_t + 1 : max_t + 2])
        )
        tau_hat = max(tau_hat, 1 / np.log10(ess))
        ess = (1 if relative else ess) / tau_hat
        if np.isnan(rho_hat_t).any():
            ess = np.nan
        return ess

    def _ess_bulk(self, ary, relative=False):
        """Compute the effective sample size for the bulk."""
        ary = np.asarray(ary)
        if _not_valid(ary, shape_kwargs={"min_draws": 4, "min_chains": 1}):
            return np.nan
        z_scaled = self._z_scale(self._split_chains(ary))
        ess_bulk = self._ess(z_scaled, relative=relative)
        return ess_bulk

    def _ess_tail(self, ary, prob, relative=False):
        """Compute the effective sample size for the tail.

        If `prob` defined, ess = min(qess(prob), qess(1-prob))
        """
        if not isinstance(prob, Sequence):
            prob = sorted((prob, 1 - prob))

        ary = np.asarray(ary)
        if _not_valid(ary, shape_kwargs={"min_draws": 4, "min_chains": 1}):
            return np.nan

        prob_low, prob_high = prob
        quantile_low_ess = self._ess_quantile(ary, prob_low, relative=relative)
        quantile_high_ess = self._ess_quantile(ary, prob_high, relative=relative)
        return min(quantile_low_ess, quantile_high_ess)

    def _ess_mean(self, ary, relative=False):
        """Compute the effective sample size for the mean."""
        ary = np.asarray(ary)
        if _not_valid(ary, shape_kwargs={"min_draws": 4, "min_chains": 1}):
            return np.nan
        return self._ess(self._split_chains(ary), relative=relative)

    def _ess_sd(self, ary, relative=False):
        """Compute the effective sample size for the sd."""
        ary = np.asarray(ary)
        if _not_valid(ary, shape_kwargs={"min_draws": 4, "min_chains": 1}):
            return np.nan
        ary = (ary - ary.mean()) ** 2
        return self._ess(self._split_chains(ary), relative=relative)

    def _ess_quantile(self, ary, prob, relative=False):
        """Compute the effective sample size for the specific residual."""
        ary = np.asarray(ary)
        if _not_valid(ary, shape_kwargs={"min_draws": 4, "min_chains": 1}):
            return np.nan
        if prob is None:
            raise TypeError("Prob not defined.")
        quantile = self.quantile(ary, prob, axis=None)
        iquantile = ary <= quantile
        return self._ess(self._split_chains(iquantile), relative=relative)

    def _ess_local(self, ary, prob, relative=False):
        """Compute the effective sample size for the specific residual."""
        ary = np.asarray(ary)
        if _not_valid(ary, shape_kwargs={"min_draws": 4, "min_chains": 1}):
            return np.nan
        if prob is None:
            raise TypeError("Prob not defined.")
        if len(prob) != 2:
            raise ValueError("Prob argument in ess local must be upper and lower bound")
        quantile = self.quantile(ary, prob, axis=None)
        iquantile = (quantile[0] <= ary) & (ary <= quantile[1])
        return self._ess(self._split_chains(iquantile), relative=relative)

    def _ess_z_scale(self, ary, relative=False):
        """Calculate ess for z-scaLe."""
        ary = np.asarray(ary)
        if _not_valid(ary, shape_kwargs={"min_draws": 4, "min_chains": 1}):
            return np.nan
        return self._ess(self._z_scale(self._split_chains(ary)), relative=relative)

    def _ess_folded(self, ary, relative=False):
        """Calculate split-ess for folded data."""
        ary = np.asarray(ary)
        if _not_valid(ary, shape_kwargs={"min_draws": 4, "min_chains": 1}):
            return np.nan
        return self._ess(self._z_fold(self._split_chains(ary)), relative=relative)

    def _ess_median(self, ary, relative=False):
        """Calculate split-ess for median."""
        ary = np.asarray(ary)
        if _not_valid(ary, shape_kwargs={"min_draws": 4, "min_chains": 1}):
            return np.nan
        return self._ess_quantile(ary, 0.5, relative=relative)

    def _ess_mad(self, ary, relative=False):
        """Calculate split-ess for mean absolute deviance."""
        ary = np.asarray(ary)
        if _not_valid(ary, shape_kwargs={"min_draws": 4, "min_chains": 1}):
            return np.nan
        ary = abs(ary - np.median(ary))
        ary = ary <= np.median(ary)
        ary = self._z_scale(self._split_chains(ary))
        return self._ess(ary, relative=relative)

    def _ess_identity(self, ary, relative=False):
        """Calculate ess."""
        ary = np.asarray(ary)
        if _not_valid(ary, shape_kwargs={"min_draws": 4, "min_chains": 1}):
            return np.nan
        return self._ess(ary, relative=relative)

    def _mcse_mean(self, ary):
        """Compute the Markov Chain mean error."""
        ary = np.asarray(ary)
        if _not_valid(ary, shape_kwargs={"min_draws": 4, "min_chains": 1}):
            return np.nan
        ess = self._ess_mean(ary)
        sd = np.std(ary, ddof=1)
        mcse_mean_value = sd / np.sqrt(ess)
        return mcse_mean_value

    def _mcse_sd(self, ary):
        """Compute the Markov Chain sd error."""
        ary = np.asarray(ary)
        if _not_valid(ary, shape_kwargs={"min_draws": 4, "min_chains": 1}):
            return np.nan
        sims_c2 = (ary - ary.mean()) ** 2
        ess = self._ess_mean(ary)
        evar = sims_c2.mean()
        varvar = ((sims_c2**2).mean() - evar**2) / ess
        varsd = varvar / evar / 4
        mcse_sd_value = np.sqrt(varsd)
        return mcse_sd_value

    def _mcse_median(self, ary):
        """Compute the Markov Chain median error."""
        return self._mcse_quantile(ary, 0.5)

    def _mcse_quantile(self, ary, prob):
        """Compute the Markov Chain quantile error at quantile=prob."""
        ary = np.asarray(ary)
        if _not_valid(ary, shape_kwargs={"min_draws": 4, "min_chains": 1}):
            return np.nan
        ess = self._ess_quantile(ary, prob)
        probability = [0.1586553, 0.8413447]
        with np.errstate(invalid="ignore"):
            ppf = stats.beta.ppf(probability, ess * prob + 1, ess * (1 - prob) + 1)
        sorted_ary = np.sort(ary.ravel())
        size = sorted_ary.size
        ppf_size = ppf * size - 1
        th1 = sorted_ary[int(np.floor(np.nanmax((ppf_size[0], 0))))]
        th2 = sorted_ary[int(np.ceil(np.nanmin((ppf_size[1], size - 1))))]
        return (th2 - th1) / 2

    def _pareto_min_ss(self, ary):
        """Compute the minimum effective sample size."""
        ary = np.asarray(ary)
        if _not_valid(ary, check_nan=False, shape_kwargs={"min_draws": 4, "min_chains": 1}):
            warnings.warn(
                "Pareto minimum effective sample size requieres "
                "at least 4 draws and at least 1 chain."
            )
            return np.nan
        if not np.all(np.isfinite(ary)):
            warnings.warn(
                "Pareto minimum effective sample size cannot be computed with non-finite values."
            )
            return np.nan

        _, kappa = self._pareto_khat(ary, tail="both", log_weights=False)

        # This should be 1, but to avoid overflow we use 0.99
        # we could even use a lower value as this will give
        # a ridiculously large number of samples needed
        if kappa < 0.99:
            return 10 ** (1 / (1 - max(0, kappa)))

        return np.inf

    def _psislw(self, ary, r_eff):
        """Compute Pareto smoothed importance sampling (PSIS) log weights."""
        ary_shape = ary.shape
        ary = ary.flatten()
        n_draws = len(ary)
        n_draws_tail = self._get_ps_tails(n_draws, r_eff, tail="right")
        ary, khat = self._ps_tail(
            -ary, n_draws, n_draws_tail, smooth_draws=True, tail="right", log_weights=True
        )

        return ary.reshape(ary_shape), khat

    def _bfmi(self, ary):  # pylint: disable=no-self-use
        """Calculate the estimated Bayesian fraction of missing information."""
        ary_mat = np.atleast_2d(ary)
        num = np.square(np.diff(ary_mat, axis=1)).mean(axis=1)
        den = np.var(ary, axis=1, ddof=1)
        return num / den

    def _loo(self, ary, r_eff=1.0, log_weights=None, pareto_k=None, log_jacobian=None):
        """
        Compute PSIS-LOO-CV for a 2D array (chain, draw).

        Parameters
        ----------
        ary : np.ndarray
            2D array with shape (n_chains, n_draws)
        r_eff : float, default 1.0
            Relative effective sample size
        log_weights : np.ndarray, optional
            Pre-computed PSIS-LOO-CV log weights (same shape as ary)
        pareto_k : float, optional
            Pre-computed Pareto k value
        log_jacobian : float, optional
            Log-Jacobian adjustment for variable transformations

        Returns
        -------
        elpd_i : float
            Expected log pointwise predictive density
        pareto_k : float
            Pareto k diagnostic value
        p_loo_i : float
            Effective number of parameters
        """
        ary = np.asarray(ary)
        n_samples = ary.size

        if log_weights is None:
            log_weights, pareto_k = self._psislw(ary, r_eff)
        elif pareto_k is None:
            raise ValueError("If log_weights is provided, pareto_k must also be provided.")

        log_weights_sum = log_weights + ary
        elpd_i = logsumexp(log_weights_sum)
        lppd_i = logsumexp(ary, b=1 / n_samples)
        p_loo_i = lppd_i - elpd_i

        if log_jacobian is not None:
            elpd_i = elpd_i + log_jacobian

        return elpd_i, pareto_k, p_loo_i

    def _loo_approximate_posterior(self, ary, log_p, log_q, log_jacobian=None):
        """
        Compute PSIS-LOO-CV with approximate posterior correction.

        Parameters
        ----------
        ary : np.ndarray
            2D array (chain, draw) of log-likelihood values for one observation
        log_p : np.ndarray
            2D array (chain, draw) of target log-densities
        log_q : np.ndarray
            2D array (chain, draw) of proposal log-densities
        log_jacobian : float, optional
            Log-Jacobian adjustment for this observation

        Returns
        -------
        elpd_i : float
            Expected log pointwise predictive density
        pareto_k : float
            Pareto k diagnostic value
        p_loo_i : float
            Effective number of parameters
        """
        ary = np.asarray(ary)
        log_p = np.asarray(log_p)
        log_q = np.asarray(log_q)

        approx_correction = log_p - log_q
        approx_correction = approx_correction - approx_correction.max()

        corrected_log_ratios = -ary + approx_correction
        corrected_log_ratios = corrected_log_ratios - corrected_log_ratios.max()

        psis_input = -corrected_log_ratios
        log_weights, pareto_k = self._psislw(psis_input, r_eff=1.0)

        return self._loo(
            ary, r_eff=1.0, log_weights=log_weights, pareto_k=pareto_k, log_jacobian=log_jacobian
        )

    @staticmethod
    def _loo_score(ary, y_obs, log_weights, kind):
        """
        Compute CRPS or SCRPS for a single observation.

        Parameters
        ----------
        ary : np.ndarray
            1D array of posterior predictive samples (flattened chain*draw).
        y_obs : float
            Observed value.
        log_weights : np.ndarray
            1D array of pre-computed PSIS log weights.
        kind : str
            Score type, either "crps" or "scrps".

        Returns
        -------
        score : float
            Score value (negative orientation for maximization).
        """
        ary = np.asarray(ary).ravel()
        log_weights = np.asarray(log_weights).ravel()
        y_obs = np.asarray(y_obs).flat[0]
        abs_error = np.abs(ary - y_obs)

        log_den = logsumexp(log_weights)
        loo_weighted_abs_error = np.exp(logsumexp(log_weights, b=abs_error) - log_den)
        loo_weighted_mean_prediction = np.exp(logsumexp(log_weights, b=ary) - log_den)

        weights = np.exp(log_weights - log_weights.max())
        weights /= np.sum(weights)

        idx = np.argsort(ary, kind="mergesort")
        values_sorted = ary[idx]
        weights_sorted = weights[idx]

        cumulative_weights = np.cumsum(weights_sorted)
        f_minus = cumulative_weights - weights_sorted
        f_mid = f_minus + weights_sorted / 2
        pwm_first_moment_b1 = np.sum(weights_sorted * values_sorted * f_mid)

        crps = loo_weighted_abs_error + loo_weighted_mean_prediction - 2.0 * pwm_first_moment_b1

        if kind == "crps":
            return -crps

        bracket = 2.0 * f_minus + weights_sorted - 1.0
        gini_mean_difference = 2.0 * np.sum(weights_sorted * values_sorted * bracket)
        return -(loo_weighted_abs_error / gini_mean_difference) - 0.5 * np.log(gini_mean_difference)

    @staticmethod
    def _loo_pit(ary, y_obs, log_weights, rng=None):
        """
        Compute LOO-PIT value for a single observation.

        Parameters
        ----------
        ary : np.ndarray
            1D array of posterior predictive samples (flattened chain*draw).
        y_obs : float
            Observed value.
        log_weights : np.ndarray
            1D array of pre-computed PSIS log weights.
        rng : np.random.Generator, optional
            Random number generator for tie-breaking. If None, uses midpoint.

        Returns
        -------
        pit : float
            LOO-PIT value in [0, 1].
        """
        ary = np.asarray(ary).ravel()
        log_weights = np.asarray(log_weights).ravel()
        y_obs_val = np.asarray(y_obs).ravel()[0]
        log_norm = logsumexp(log_weights)
        weights = np.exp(log_weights - log_norm)

        sel_below = ary < y_obs_val
        pit_lower = np.sum(weights[sel_below])

        sel_equal = ary == y_obs_val
        if np.any(sel_equal):
            pit_at_obs = np.sum(weights[sel_equal])
            pit_upper = pit_lower + pit_at_obs

            if rng is not None:
                return rng.uniform(pit_lower, pit_upper)
            return (pit_lower + pit_upper) / 2.0

        return pit_lower

    def _loo_expectation(self, ary, log_weights, kind):
        """
        Compute weighted expectation for a single observation.

        Parameters
        ----------
        ary : np.ndarray
            1D array of posterior predictive samples (flattened chain*draw).
        log_weights : np.ndarray
            1D array of pre-computed PSIS log weights.
        kind : str
            Type of expectation: "mean", "median", "var", "sd",
            "circular_mean", "circular_var", "circular_sd".

        Returns
        -------
        expectation : float
            Weighted expectation value.
        """
        ary = np.asarray(ary).ravel()
        log_weights = np.asarray(log_weights).ravel()
        log_norm = logsumexp(log_weights)
        weights = np.exp(log_weights - log_norm)

        if kind == "mean":
            result = np.sum(weights * ary)
        elif kind == "median":
            result = self._weighted_quantile(ary, weights, 0.5)
        elif kind in ("var", "sd"):
            mean_val = np.sum(weights * ary)
            ess = 1.0 / np.sum(weights**2)
            correction = ess / (ess - 1) if ess > 1 else 1.0
            var_val = np.sum(weights * (ary - mean_val) ** 2) * correction
            result = np.sqrt(var_val) if kind == "sd" else var_val
        elif kind in ("circular_mean", "circular_var", "circular_sd"):
            angles_2d = ary.reshape(1, -1)
            weights_2d = weights.reshape(1, -1)
            if kind == "circular_mean":
                result = circular_mean(angles_2d, weights_2d)[0]
            elif kind == "circular_var":
                result = circular_var(angles_2d, weights_2d)[0]
            else:  # circular_sd
                result = circular_sd(angles_2d, weights_2d)[0]
        else:
            raise ValueError(f"Unknown kind: {kind}")

        return result

    def _loo_quantile(self, ary, log_weights, prob):
        """
        Compute weighted quantile for a single observation.

        Parameters
        ----------
        ary : np.ndarray
            1D array of posterior predictive samples (flattened chain*draw).
        log_weights : np.ndarray
            1D array of pre-computed PSIS log weights.
        prob : float
            Quantile probability in [0, 1].

        Returns
        -------
        quantile : float
            Weighted quantile value.
        """
        ary = np.asarray(ary).ravel()
        log_weights = np.asarray(log_weights).ravel()
        log_norm = logsumexp(log_weights)
        weights = np.exp(log_weights - log_norm)

        return self._weighted_quantile(ary, weights, prob)

    @staticmethod
    def _loo_r2(y_obs, ypred_loo, n_simulations=4000, circular=False, random_state=42):
        """
        Compute LOO-adjusted :math:`R^2` using Dirichlet-weighted bootstrap.

        Parameters
        ----------
        y_obs : np.ndarray
            1D array of observed values.
        ypred_loo : np.ndarray
            1D array of LOO predictions (same length as y_obs).
        n_simulations : int, default 4000
            Number of Dirichlet-weighted bootstrap samples.
        circular : bool, default False
            Whether the variable is circular (angles in radians).
        random_state : int, default 42
            Random seed for reproducibility.

        Returns
        -------
        loo_r_squared : np.ndarray
            Array of :math:`R^2` samples with shape (n_simulations,).
        """
        y_obs = np.asarray(y_obs).ravel()
        ypred_loo = np.asarray(ypred_loo).ravel()

        if circular:
            eloo = circular_diff(ypred_loo, y_obs)
        else:
            eloo = ypred_loo - y_obs

        n = len(y_obs)
        rd = stats.dirichlet.rvs(np.ones(n), size=n_simulations, random_state=random_state)

        if circular:
            loo_r_squared = 1 - circular_var(eloo, rd)
        else:
            vary = (np.sum(rd * y_obs**2, axis=1) - np.sum(rd * y_obs, axis=1) ** 2) * (n / (n - 1))
            vareloo = (np.sum(rd * eloo**2, axis=1) - np.sum(rd * eloo, axis=1) ** 2) * (
                n / (n - 1)
            )

            loo_r_squared = 1 - vareloo / vary
            loo_r_squared = np.clip(loo_r_squared, -1, 1)

        return loo_r_squared

    @staticmethod
    def _mixture(ary, obs_axes, sample_axes, log_jacobian=None):
        """
        Compute mixture importance sampling LOO (Mix-IS-LOO).

        Parameters
        ----------
        ary : np.ndarray
            Full log-likelihood array
        obs_axes : tuple of int
            Axes corresponding to observation dimensions
        sample_axes : tuple of int
            Axes corresponding to sample dimensions (chain, draw)
        log_jacobian : np.ndarray, optional
            Log-Jacobian adjustment

        Returns
        -------
        elpd_i : np.ndarray
            Pointwise expected log predictive density
        p_loo_i : np.ndarray
            Pointwise effective number of parameters
        mix_log_weights : np.ndarray
            Mixture log weights
        """
        ary = np.asarray(ary)
        n_samples = np.prod([ary.shape[ax] for ax in sample_axes])

        l_common = logsumexp(-ary, axis=obs_axes, keepdims=True)
        mix_log_weights = -ary - l_common
        l_common_squeezed = l_common.squeeze(axis=obs_axes)

        log_norm = logsumexp(-l_common_squeezed, axis=sample_axes)
        elpd_i = log_norm - logsumexp(mix_log_weights, axis=sample_axes)
        lppd_i = logsumexp(ary, axis=sample_axes, b=1 / n_samples)

        if log_jacobian is not None:
            log_jacobian = np.asarray(log_jacobian)
            elpd_i = elpd_i + log_jacobian
            lppd_i = lppd_i + log_jacobian

        p_loo_i = lppd_i - elpd_i
        return elpd_i, p_loo_i, mix_log_weights

    @staticmethod
    def _srs_estimator(y_sample, n_data_points, subsample_size):
        """
        Calculate the SRS estimator for PSIS-LOO-CV with sub-sampling.

        Parameters
        ----------
        y_sample : 1D array
            Values of the statistic (e.g., p_loo) for the subsample, shape (m,).
        n_data_points : int
            Total number of data points (N).
        subsample_size : int
            Number of observations in the subsample (m).

        Returns
        -------
        y_hat : float
            The estimated statistic using simple random sampling.
        var_y_hat : float
            The variance of the estimator (sampling uncertainty).
        hat_var_y : float
            The estimated variance of the statistic.
        """
        y_sample = np.asarray(y_sample).ravel()
        y_sample_mean = np.mean(y_sample)
        y_hat = n_data_points * y_sample_mean

        if subsample_size > 1:
            y_sample_var = np.var(y_sample, ddof=1)
            var_y_hat = (
                (n_data_points**2)
                * (1 - subsample_size / n_data_points)
                * y_sample_var
                / subsample_size
            )
            hat_var_y = n_data_points * y_sample_var
        else:
            var_y_hat = np.inf
            hat_var_y = np.inf

        return y_hat, var_y_hat, hat_var_y

    @staticmethod
    def _diff_srs_estimator(
        elpd_loo_i_sample,
        lpd_approx_sample,
        lpd_approx_all,
        n_data_points,
        subsample_size,
    ):
        """
        Calculate the difference estimator for PSIS-LOO-CV with sub-sampling.

        Parameters
        ----------
        elpd_loo_i_sample : 1D array
            Pointwise ELPD values for the subsample, shape (m,).
        lpd_approx_sample : 1D array
            LPD approximation values for the subsample, shape (m,).
        lpd_approx_all : 1D array
            LPD approximation values for the full dataset, shape (N,).
        n_data_points : int
            Total number of data points (N).
        subsample_size : int
            Number of observations in the subsample (m).

        Returns
        -------
        elpd_loo_hat : float
            The estimated ELPD using the difference estimator.
        subsampling_se : float
            The standard error due to subsampling uncertainty.
        total_se : float
            The total standard error (approximation + sampling uncertainty).
        """
        elpd_sample = np.asarray(elpd_loo_i_sample, dtype=float).ravel()
        approx_sample = np.asarray(lpd_approx_sample, dtype=float).ravel()
        approx_all = np.asarray(lpd_approx_all, dtype=float).ravel()

        if elpd_sample.size != approx_sample.size:
            raise ValueError("Subsampled ELPD and approximation must have matching shapes.")

        finite_mask = np.isfinite(elpd_sample) & np.isfinite(approx_sample)
        valid_count = int(np.count_nonzero(finite_mask))

        if valid_count == 0:
            return np.nan, np.inf, np.inf

        elpd_valid = elpd_sample[finite_mask]
        approx_valid = approx_sample[finite_mask]

        pointwise_diff = elpd_valid - approx_valid

        lpd_approx_sum_all = np.nansum(approx_all)
        scaled_mean_pointwise_diff = n_data_points * np.nanmean(pointwise_diff)
        elpd_loo_estimate = lpd_approx_sum_all + scaled_mean_pointwise_diff

        subsampling_se = np.inf
        total_se = np.inf

        effective_m = min(valid_count, subsample_size)
        if effective_m > 1 and n_data_points > 0:
            variance = np.nanvar(pointwise_diff, ddof=1)

            if np.isfinite(variance):
                finite_pop_correction = max(0.0, 1 - effective_m / n_data_points)
                subsampling_variance = (
                    (n_data_points**2) * finite_pop_correction * variance / effective_m
                )
                subsampling_variance = np.nan_to_num(subsampling_variance, nan=np.inf)

                subsampling_se = (
                    np.sqrt(subsampling_variance) if np.isfinite(subsampling_variance) else np.inf
                )

                lpd_approx_sq_sum_all = np.nansum(approx_all**2)
                mean_sq_diff = np.nanmean(elpd_valid**2 - approx_valid**2)
                scaled_mean_sq_diff = n_data_points * mean_sq_diff

                total_variance_estimate = (
                    lpd_approx_sq_sum_all
                    + scaled_mean_sq_diff
                    - (1 / n_data_points)
                    * (
                        scaled_mean_pointwise_diff**2
                        - subsampling_variance
                        + 2 * lpd_approx_sum_all * elpd_loo_estimate
                        - lpd_approx_sum_all**2
                    )
                )

                if np.isfinite(total_variance_estimate):
                    total_variance_estimate = max(0.0, total_variance_estimate)
                    total_se = np.sqrt(total_variance_estimate)
                else:
                    total_se = np.inf

        return elpd_loo_estimate, subsampling_se, total_se

    @staticmethod
    def _weighted_quantile(ary, weights, prob):
        """
        Compute weighted quantile.

        Parameters
        ----------
        ary : np.ndarray
            1D array of values
        weights : np.ndarray
            1D array of normalized weights (must sum to 1)
        prob : float
            Quantile probability in [0, 1]

        Returns
        -------
        quantile : float
            Weighted quantile value
        """
        nonzero = weights != 0
        ary = ary[nonzero]
        weights = weights[nonzero]
        n = ary.size

        if n == 0:
            return np.nan
        nw = weights.sum() ** 2 / (weights**2).sum()

        idx = np.argsort(ary)
        sorted_ary = ary[idx]
        sorted_weights = weights[idx]

        sorted_weights = sorted_weights / sorted_weights.sum()
        weights_cum = np.concatenate([[0], np.cumsum(sorted_weights)])

        h = (nw - 1) * prob + 1
        h = np.clip(h, 1, nw)

        u = np.maximum((h - 1) / nw, np.minimum(h / nw, weights_cum))
        v = u * nw - h + 1
        w = np.diff(v)

        return np.sum(sorted_ary * w)

    @staticmethod
    def _summary(elpd_i, p_loo_i):
        """
        Aggregate pointwise LOO values.

        Parameters
        ----------
        elpd_i : np.ndarray
            Pointwise expected log predictive density values
        p_loo_i : np.ndarray
            Pointwise effective number of parameters

        Returns
        -------
        elpd : float
            Total expected log predictive density
        se : float
            Standard error of elpd
        p_loo : float
            Total effective number of parameters
        lppd : float
            Log pointwise predictive density
        """
        elpd_i = np.asarray(elpd_i).ravel()
        p_loo_i = np.asarray(p_loo_i).ravel()
        n = len(elpd_i)
        elpd = np.sum(elpd_i)
        se = np.sqrt(n * np.var(elpd_i))
        p_loo = np.sum(p_loo_i)
        lppd = elpd + p_loo
        return elpd, se, p_loo, lppd

    def _pareto_khat(self, ary, r_eff=None, tail="both", log_weights=False):
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

        if r_eff is None:
            r_eff = self._ess_tail(ary, prob=0.05, relative=True)

        ary = ary.flatten()
        n_draws = len(ary)

        n_draws_tail = self._get_ps_tails(n_draws, r_eff, tail=tail)

        if tail == "both":
            khat = max(
                self._ps_tail(ary, n_draws, n_draws_tail, smooth_draws=False, tail=t)[1]
                for t in ("left", "right")
            )
        else:
            ary, khat = self._ps_tail(ary, n_draws, n_draws_tail, smooth_draws=False, tail=tail)

        return ary, khat

    @staticmethod
    def _get_ps_tails(n_draws, r_eff, tail):
        if n_draws * r_eff > 225:
            n_draws_tail = np.floor(3 * (n_draws / r_eff) ** 0.5)
        else:
            n_draws_tail = np.floor(n_draws / 5)

        if tail == "both":
            half_n_draws = n_draws // 2
            if n_draws_tail > half_n_draws:
                warnings.warn(
                    "Number of tail draws cannot be more than half "
                    "the total number of draws if both tails are fit, "
                    f"changing to {half_n_draws}"
                )
                n_draws_tail = half_n_draws

            if n_draws_tail < 5:
                warnings.warn("Number of tail draws cannot be less than 5. Changing to 5")
                n_draws_tail = 5

        return n_draws_tail

    def _ps_tail(
        self, ary, n_draws, n_draws_tail, smooth_draws=False, tail="both", log_weights=False
    ):
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

        tail_ids = np.arange(n_draws - n_draws_tail, n_draws, dtype=int)

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

            khat, sigma = self._gpdfit(draws_tail - cutoff)

            if np.isfinite(khat) and smooth_draws:
                p = np.arange(0.5, n_draws_tail) / n_draws_tail
                smoothed = self._gpinv(p, khat, sigma, cutoff)

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

        # normalise weights
        if log_weights:
            ary -= logsumexp(ary)
        else:
            ary = ary / np.sum(ary)

        return ary, khat

    @staticmethod
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

        k_ary = np.mean(np.log1p(-b_ary[:, None] * ary), axis=1)
        len_scale = n * (np.log(-b_ary / k_ary) - k_ary - 1)
        weights = np.exp(len_scale - logsumexp(len_scale))

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
        kappa = np.mean(np.log1p(-b_post * ary))
        # add prior for kappa
        sigma = -kappa / b_post
        kappa = (n * kappa + prior_k * 0.5) / (n + prior_k)

        return kappa, sigma

    @staticmethod
    def _gpinv(probs, kappa, sigma, mu):
        """Quantile function for generalized pareto distribution."""
        if sigma <= 0:
            return np.full_like(probs, np.nan)

        if kappa == 0:
            q = mu - sigma * np.log1p(-probs)
        else:
            q = mu + sigma * np.expm1(-kappa * np.log1p(-probs)) / kappa

        return q

    def _power_scale_sense(self, ary, lower_w, upper_w, lower_alpha, upper_alpha):
        """Compute power-scaling sensitivity by finite difference second derivative of CJS."""
        ary = np.ravel(ary)
        lower_w = np.ravel(lower_w)
        upper_w = np.ravel(upper_w)
        lower_cjs = max(self._cjs_dist(ary, lower_w), self._cjs_dist(-1 * ary, lower_w))
        upper_cjs = max(self._cjs_dist(ary, upper_w), self._cjs_dist(-1 * ary, upper_w))
        lower_grad = -1 * lower_cjs / np.log2(lower_alpha)
        upper_grad = upper_cjs / np.log2(upper_alpha)
        return (lower_grad + upper_grad) / 2

    def _power_scale_lw(self, ary, alpha):
        """Compute log weights for power-scaling component by alpha."""
        shape = ary.shape
        ary = np.ravel(ary)
        log_weights = (alpha - 1) * ary
        n_draws = len(log_weights)
        r_eff = self._ess_tail(ary, prob=0.05, relative=True)
        n_draws_tail = self._get_ps_tails(n_draws, r_eff, tail="both")
        log_weights, _ = self._ps_tail(
            log_weights,
            n_draws,
            n_draws_tail,
            smooth_draws=True,
            log_weights=True,
        )

        return log_weights.reshape(shape)

    @staticmethod
    def _cjs_dist(ary, weights):
        """Calculate the cumulative Jensen-Shannon distance between original and weighted draws."""
        # sort draws and weights
        order = np.argsort(ary)
        ary = ary[order]
        weights = weights[order]

        binwidth = np.diff(ary)

        # ecdfs
        cdf_p = np.linspace(1 / len(ary), 1 - 1 / len(ary), len(ary) - 1)
        cdf_q = np.cumsum(weights / np.sum(weights))[:-1]

        # integrals of ecdfs
        cdf_p_int = np.dot(cdf_p, binwidth)
        cdf_q_int = np.dot(cdf_q, binwidth)

        # cjs calculation
        pq_numer = np.log2(cdf_p, out=np.zeros_like(cdf_p), where=cdf_p != 0)
        qp_numer = np.log2(cdf_q, out=np.zeros_like(cdf_q), where=cdf_q != 0)

        denom = 0.5 * (cdf_p + cdf_q)
        denom = np.log2(denom, out=np.zeros_like(denom), where=denom != 0)

        cjs_pq = np.sum(binwidth * (cdf_p * (pq_numer - denom))) + 0.5 / np.log(2) * (
            cdf_q_int - cdf_p_int
        )

        cjs_qp = np.sum(binwidth * (cdf_q * (qp_numer - denom))) + 0.5 / np.log(2) * (
            cdf_p_int - cdf_q_int
        )

        cjs_pq = max(0, cjs_pq)
        cjs_qp = max(0, cjs_qp)

        bound = cdf_p_int + cdf_q_int

        return np.sqrt((cjs_pq + cjs_qp) / bound)

    def _rhat_nested_rank(self, ary, superchain_ids):
        """Compute the rank normalized rhat for 2d array.

        Computation follows https://arxiv.org/abs/1903.08008
        """
        ary = np.asarray(ary)
        if _not_valid(ary, shape_kwargs={"min_draws": 4, "min_chains": 2}):
            return np.nan
        split_ary = self._split_chains(ary)
        # splitting we "duplicate the number of chains so we need to update
        # superchain_ids accordingly if we had 3 chains (0, 1, 2) we end up
        # with (0, 1, 2, 0, 1, 2) half-chains
        superchain_ids = np.hstack((superchain_ids, superchain_ids))
        rhat_bulk = self._rhat_nested(self._z_scale(split_ary), superchain_ids)

        split_ary_folded = abs(split_ary - np.median(split_ary))
        rhat_tail = self._rhat_nested(self._z_scale(split_ary_folded), superchain_ids)

        rhat_rank = max(rhat_bulk, rhat_tail)
        return rhat_rank

    def _rhat_nested_folded(self, ary, superchain_ids):
        """Calculate split-Rhat for folded z-values."""
        ary = np.asarray(ary)
        if _not_valid(ary, shape_kwargs={"min_draws": 4, "min_chains": 2}):
            return np.nan
        ary = self._z_fold(self._split_chains(ary))
        superchain_ids = np.hstack((superchain_ids, superchain_ids))
        return self._rhat_nested(ary, superchain_ids)

    def _rhat_nested_z_scale(self, ary, superchain_ids):
        ary = np.asarray(ary)
        if _not_valid(ary, shape_kwargs={"min_draws": 4, "min_chains": 2}):
            return np.nan
        ary = self._z_scale(self._split_chains(ary))
        superchain_ids = np.hstack((superchain_ids, superchain_ids))
        return self._rhat_nested(ary, superchain_ids)

    def _rhat_nested_split(self, ary, superchain_ids):
        ary = np.asarray(ary)
        if _not_valid(ary, shape_kwargs={"min_draws": 4, "min_chains": 2}):
            return np.nan
        superchain_ids = np.hstack((superchain_ids, superchain_ids))
        return self._rhat_nested(self._split_chains(ary), superchain_ids)

    def _rhat_nested_identity(self, ary, superchain_ids):
        ary = np.asarray(ary)
        if _not_valid(ary, shape_kwargs={"min_draws": 4, "min_chains": 2}):
            return np.nan
        return self._rhat_nested(ary, superchain_ids)

    @staticmethod
    def _rhat_nested(ary, superchain_ids):
        ary = np.asarray(ary)
        nchains, niterations = ary.shape

        # Check that all chains are assigned a superchain
        if len(superchain_ids) != nchains:
            raise ValueError("Length of superchain_ids not equal to number of chains")

        # Check that superchains have equal length
        superchain_counts = np.bincount(superchain_ids)
        nchains_per_superchain = np.max(superchain_counts)

        if nchains_per_superchain != np.min(superchain_counts):
            raise ValueError("Number of chains per superchain is not the same for each superchain")

        superchains = np.unique(superchain_ids)

        # Compute chain means and variances
        chain_mean = np.mean(ary, axis=1)
        chain_var = np.var(ary, axis=1, ddof=1)

        # mean of superchains calculated by only including specified chains
        # (equation 4 in Margossian et al. 2024)
        superchain_mean = np.array([np.mean(chain_mean[superchain_ids == k]) for k in superchains])

        # between-chain variance estimate (Bhat_k in equation 7 in Margossian et al. 2024)
        if nchains_per_superchain == 1:
            var_between_chain = np.zeros(len(superchains))
        else:
            var_between_chain = np.array(
                [np.var(chain_mean[superchain_ids == k], ddof=1) for k in superchains]
            )

        #  within-chain variance estimate (What_k in equation 7 in Margossian et al. 2024)
        if niterations == 1:
            var_within_chain = np.zeros(len(superchains))
        else:
            var_within_chain = np.array(
                [np.mean(chain_var[np.where(superchain_ids == k)[0]]) for k in superchains]
            )

        # between-superchain variance (Bhat_nu in equation 6 in Margossian et al. 2024)
        var_between_superchain = np.var(superchain_mean, ddof=1)

        # within-superchain variance (What_nu in equation 7 in Margossian et al. 2024)
        var_within_superchain = np.mean(var_within_chain + var_between_chain)

        # nested Rhat (Rhat_nu in equation 8 in Margossian et al. 2024)
        return np.sqrt(1 + var_between_superchain / var_within_superchain)

    @staticmethod
    def _bayesian_r2(mu_pred, scale, scale_kind, circular):
        """Compute the Bayesian R².

        Parameters
        ----------
        mu_pred: array-like of shape = (n_posterior_samples, n_outputs)
            Estimated mean for the response variable.
        var : array-like of shape (n_posterior_samples,), optional
            Posterior draws of the variance or pseudo-variance.
            - If provided: treated as the model-implied residual variance.
            - If None: assumes Bernoulli-like model and computes pseudo-variance
            as mean(mu_pred) * (1 - mean(mu_pred)) per posterior draw.
        scale_kind : str, optional
            Kind of scale for the variance. Options are 'sd' (standard deviation) or
            'var' (variance). Default is 'sd'.
        circular: bool, optional
            Whether the response variable is circular. For circular response,
            the scale should be the circular variance and scale_kind should be 'var'.

        Returns
        -------
        array-like  (sample, dims)

        """
        if scale is None:
            if circular:
                raise ValueError("scale must be provided for circular response.")

            # Bernoulli-like models: use Tjur’s pseudo-variance
            scale = np.mean(mu_pred * (1 - mu_pred), axis=1)
        else:
            if scale_kind not in ("sd", "var"):
                raise ValueError("scale_kind must be either 'sd' or 'var'")

            if np.any(scale < 0):
                raise ValueError("Scale must be non-negative.")

            if scale_kind == "sd":
                if circular:
                    raise ValueError("scale_kind cannot be 'sd' for circular response.")
                scale = scale**2

        if circular:
            return 1 - scale

        var_y_est = np.var(mu_pred, axis=1, ddof=1)
        return var_y_est / (var_y_est + scale)

    @staticmethod
    def _residual_r2(y_obs, mu_pred, circular):
        """Compute the residual R².

        Parameters
        ----------
        y_obs: array-like of shape = (n_outputs,)
            Observed values for the response variable.
        mu_pred: array-like of shape = (n_posterior_samples, n_outputs)
            Estimated mean for the response variable.
        circular: bool, optional
            Whether the response variable is circular.

        Returns
        -------
        array-like  (sample, dims)
        """
        if circular:
            return 1 - circular_var(circular_diff(y_obs, mu_pred))

        var_y_est = np.var(mu_pred, axis=1, ddof=1)
        var_e = np.var(y_obs - mu_pred, axis=1, ddof=1)

        r_squared = var_y_est / (var_y_est + var_e)
        return r_squared

    @staticmethod
    def _mae(observed, predicted):
        """Compute the Mean Absolute Error (MAE).

        Parameters
        ----------
        observed: array-like of shape = (n_outputs,)
            Ground truth (correct) target values.
        predicted: array-like of shape = (n_outputs)
            Estimated target values.

        Returns
        -------
        mean: float
            Mean absolute error.
        std_error: float
            Standard error of the mean absolute error.
        """
        n_obs = len(observed)
        abs_e = np.abs(observed - predicted)
        mean = np.mean(abs_e)
        std_error = np.std(abs_e) / n_obs**0.5
        return mean, std_error

    @staticmethod
    def _mse(observed, predicted):
        """Compute the Mean Squared Error (MSE).

        Parameters
        ----------
        observed: array-like of shape = (n_outputs,)
            Ground truth (correct) target values.
        predicted: array-like of shape = (n_outputs)
            Estimated target values.

        Returns
        -------
        mean: float
            Mean squared error.
        std_error: float
            Standard error of the mean squared error.
        """
        n_obs = len(observed)
        sq_e = (observed - predicted) ** 2
        mean = np.mean(sq_e)
        std_error = np.std(sq_e) / n_obs**0.5
        return mean, std_error

    @staticmethod
    def _rmse(observed, predicted):
        """Compute the Root Mean Squared Error (RMSE).

        Parameters
        ----------
        observed: array-like of shape = (n_outputs,)
            Ground truth (correct) target values.
        predicted: array-like of shape = (n_outputs)
            Estimated target values.

        Returns
        -------
        mean: float
            Root mean squared error.
        std_error: float
            Standard error of the root mean squared error.
        """
        n_obs = len(observed)
        sq_e = (observed - predicted) ** 2
        mean_mse = np.mean(sq_e)
        var_mse = np.var(sq_e) / n_obs
        var_rmse = var_mse / mean_mse / 4  # Comes from the first order Taylor approx.
        mean = mean_mse**0.5
        std_error = var_rmse**0.5

        return mean, std_error

    @staticmethod
    def _acc(observed, predicted):
        """Compute the Accuracy.

        Parameters
        ----------
        observed: array-like of shape = (n_outputs,)
            Ground truth (correct) target values.
        predicted: array-like of shape = (n_outputs)
            Estimated target values.

        Returns
        -------
        mean: float
            Accuracy.
        std_error: float
            Standard error of the accuracy.
        """
        n_obs = len(observed)
        yhat = predicted > 0.5
        acc = yhat == observed
        mean = np.mean(acc)
        std_error = (mean * (1 - mean) / n_obs) ** 0.5
        return mean, std_error

    @staticmethod
    def _acc_balanced(observed, predicted):
        """Compute the Balanced Accuracy.

        Parameters
        ----------
        observed: array-like of shape = (n_outputs,)
            Ground truth (correct) target values.
        predicted: array-like of shape = (n_outputs)
            Estimated target values.

        Returns
        -------
        mean: float
            Balanced accuracy.
        std_error: float
            Standard error of the balanced accuracy.
        """
        n_obs = len(observed)
        yhat = predicted > 0.5
        mask = observed == 0
        true_neg = np.mean(yhat[mask] == observed[mask])
        true_pos = np.mean(yhat[~mask] == observed[~mask])
        mean = (true_pos + true_neg) / 2
        # This approximation has quite large bias for small samples
        bls_acc_var = (true_pos * (1 - true_pos) + true_neg * (1 - true_neg)) / 4
        std_error = bls_acc_var / n_obs**0.5
        return mean, std_error
