# pylint: disable=too-many-lines, too-many-function-args, redefined-outer-name
"""Diagnostic functions for ArviZ."""

from collections.abc import Sequence

import numpy as np
from scipy import stats

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

    def _ess_tail(self, ary, prob=None, relative=False):
        """Compute the effective sample size for the tail.

        If `prob` defined, ess = min(qess(prob), qess(1-prob))
        """
        if prob is None:
            prob = (0.05, 0.95)
        elif not isinstance(prob, Sequence):
            prob = (prob, 1 - prob)

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
        ary = self._split_chains(ary)
        return min(self._ess(ary, relative=relative), self._ess(ary**2, relative=relative))

    def _ess_quantile(self, ary, prob, relative=False):
        """Compute the effective sample size for the specific residual."""
        ary = np.asarray(ary)
        if _not_valid(ary, shape_kwargs={"min_draws": 4, "min_chains": 1}):
            return np.nan
        if prob is None:
            raise TypeError("Prob not defined.")
        quantile = self.quantile(ary, prob)
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
        quantile = self.quantile(ary, prob)
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
        ess = self._ess_sd(ary)
        sd = np.std(ary, ddof=1)
        fac_mcse_sd = np.sqrt(np.exp(1) * (1 - 1 / ess) ** (ess - 1) - 1)
        mcse_sd_value = sd * fac_mcse_sd
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
