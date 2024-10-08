# pylint: disable=too-many-lines, too-many-function-args, redefined-outer-name
"""Diagnostic functions for ArviZ."""

import warnings
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

    def _pareto_min_ss(self, ary):
        """Compute the minimum effective sample size."""
        ary = np.asarray(ary)
        if _not_valid(ary, shape_kwargs={"min_draws": 4, "min_chains": 1}):
            return np.nan
        ary_flatten = ary.flatten()
        r_eff = self._ess_tail(ary, relative=True)

        kappa = self._pareto_khat(ary_flatten, r_eff=r_eff, tail="both", log_weights=False)

        if kappa < 1:
            return 10 ** (1 / (1 - max(0, kappa)))

        return np.inf

    def _pareto_khat(self, ary, r_eff=1, tail="both", log_weights=False):
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

        n_draws_tail = self._get_ps_tails(n_draws, r_eff, tail=tail)

        if tail == "both":
            khat = max(
                self._ps_tail(ary, n_draws, n_draws_tail, smooth_draws=False, tail=t)[1]
                for t in ("left", "right")
            )
        else:
            _, khat = self._ps_tail(ary, n_draws, n_draws_tail, smooth_draws=False, tail=tail)

        return khat

    @staticmethod
    def _get_ps_tails(n_draws, r_eff, tail):
        if n_draws > 255:
            n_draws_tail = np.ceil(3 * (n_draws / r_eff) ** 0.5).astype(int)
        else:
            n_draws_tail = int(n_draws / 5)

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
                p = (np.arange(0.5, n_draws_tail)) / n_draws_tail
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

    @staticmethod
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

    def _power_scale_sense(self, ary, lower_w, upper_w, delta=0.01):
        """Compute power-scaling sensitivity by finite difference second derivative of CJS."""
        ary = np.ravel(ary)
        lower_w = np.ravel(lower_w)
        upper_w = np.ravel(upper_w)
        lower_cjs = max(self._cjs_dist(ary, lower_w), self._cjs_dist(-1 * ary, lower_w))
        upper_cjs = max(self._cjs_dist(ary, upper_w), self._cjs_dist(-1 * ary, upper_w))
        grad = (lower_cjs + upper_cjs) / (2 * np.log2(1 + delta))
        return grad

    def _power_scale_lw(self, ary, alpha):
        """Compute log weights for power-scaling component by alpha."""
        shape = ary.shape
        ary = np.ravel(ary)
        log_weights = (alpha - 1) * ary
        n_draws = len(log_weights)
        r_eff = self._ess_tail(ary, relative=True)
        n_draws_tail = self._get_ps_tails(n_draws, r_eff, tail="both")
        log_weights, _ = self._ps_tail(
            log_weights,
            n_draws,
            n_draws_tail,
            smooth_draws=False,
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
