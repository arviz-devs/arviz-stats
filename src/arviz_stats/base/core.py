"""Core stats functions.

Functions that are needed by multiple "organization classes"
should go here. e.g. fft is used for kde bandwidth estimation and for ess.
"""

import warnings

import numpy as np
from scipy.fftpack import next_fast_len
from scipy.interpolate import CubicSpline
from scipy.stats import circmean


class _CoreBase:
    def fft(self, x):  # pylint: disable=no-self-use
        return np.fft.fft(x)

    def rfft(self, ary, n, axis=-1):  # pylint: disable=no-self-use
        return np.fft.rfft(ary, n=n, axis=axis)

    def irfft(self, ary, n, axis=-1):  # pylint: disable=no-self-use
        return np.fft.irfft(ary, n=n, axis=axis)

    def autocov(self, ary, axis=-1):
        """Compute autocovariance estimates for every lag for the input array.

        Parameters
        ----------
        ary : array-like
        axis : int, default -1
        """
        if not isinstance(axis, int):
            raise ValueError("Only integer values are allowed for `axis` in autocov.")
        axis = axis if axis > 0 else len(ary.shape) + axis
        n = ary.shape[axis]
        m = next_fast_len(2 * n)

        ary = ary - ary.mean(axis, keepdims=True)

        # added to silence tuple warning for a submodule
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            ifft_ary = self.rfft(ary, n=m, axis=axis)
            ifft_ary *= np.conjugate(ifft_ary)

            shape = tuple(
                slice(None) if dim_len != axis else slice(0, n)
                for dim_len, _ in enumerate(ary.shape)
            )
            cov = self.irfft(ifft_ary, n=m, axis=axis)[shape]
            cov /= n

        return cov

    def autocorr(self, ary, axis=-1):
        """Compute autocorrelation using FFT for every lag for the input array.

        See https://en.wikipedia.org/wiki/autocorrelation#Efficient_computation

        Parameters
        ----------
        ary : array-like
        axis : int, default -1
        """
        if not isinstance(axis, int):
            raise ValueError("Only integer values are allowed for `axis` in autocorr.")
        corr = self.autocov(ary, axis=axis)
        axis = axis = axis if axis > 0 else len(corr.shape) + axis
        norm = tuple(
            slice(None, None) if dim != axis else slice(None, 1) for dim, _ in enumerate(corr.shape)
        )
        with np.errstate(invalid="ignore"):
            corr /= corr[norm]
        return corr

    def circular_mean(self, ary):  # pylint: disable=no-self-use
        """Compute mean of circular variable measured in radians.

        The result is between -pi and pi.
        """
        return circmean(ary, high=np.pi, low=-np.pi)

    def _circular_standardize(self, ary):  # pylint: disable=no-self-use
        """Standardize circular data to the interval [-pi, pi]."""
        return np.mod(ary + np.pi, 2 * np.pi) - np.pi

    def quantile(self, ary, quantile, axis=-1, method="linear", skipna=False, weights=None):  # pylint: disable=no-self-use
        """Compute the quantile of an array of samples.

        Implementation wise, the version in `arviz_stats.base.array_stats`
        calls either :func:`numpy.quantile` or :func:`numpy.nanquantile`
        and ensures the added dimension is the last one for compatibility
        with :func:`xarray.apply_ufunc` when using it as part of the ``DataArray`` interface.

        Parameters
        ----------
        ary : array-like
        quantile : float or array-like
        axis : int or sequence of int or None, default -1
        method : str, default "linear"
        skipna : bool, default False
        weights : array-like, optional
            Array with the same shape as `a` with the weights associated to the values of `a`.

        Notes
        -----
        Should default to NumPy's "linear" method, which corresponds to
        type 7 from H&F [1]_.

        References
        ----------
        .. [1] R. J. Hyndman and Y. Fan,
           "Sample quantiles in statistical packages,"
           The American Statistician, 50(4), pp. 361-365, 1996
        """
        if skipna:
            result = np.nanquantile(ary, quantile, axis=axis, method=method, weights=weights)
        else:
            result = np.quantile(ary, quantile, axis=axis, method=method, weights=weights)
        if np.ndim(quantile) == 0:
            return result
        return np.moveaxis(result, 0, -1)

    def eti(self, ary, prob, axis=-1, method="linear", skipna=False, weights=None):
        """Compute the equal tail interval (ETI) of an array of samples.

        Parameters
        ----------
        ary : array-like
        prob : float
            Probability to be contained within the returned interval.
        axis : int or sequence of int or None, default -1
        method : str, optional
        skipna : bool, default False
        weights : array-like, optional
        """
        edge_prob = (1 - prob) / 2
        return self.quantile(
            ary,
            [edge_prob, 1 - edge_prob],
            axis=axis,
            method=method,
            skipna=skipna,
            weights=weights,
        )

    def _float_rankdata(self, ary):  # pylint: disable=no-self-use
        """Compute ranks on continuous data, assuming there are no ties.

        Notes
        -----
        :func:`scipy.stats.rankdata` is focused on discrete data and different ways
        to resolve ties. However, our most common use is converting all data to continuous
        to get rid of the ties, the call rankdata which is not very efficient nor
        numba compatible.
        """
        ranks = np.empty(len(ary), dtype=int)
        ranks[np.argsort(ary, axis=None)] = np.arange(1, ary.size + 1)
        return ranks

    def _compute_ranks(self, ary, relative=False):
        """Compute ranks for continuous and discrete variables."""
        ary_shape = ary.shape
        ary = ary.flatten()
        if ary.dtype.kind == "i":
            min_ary, max_ary = min(ary), max(ary)
            x = np.linspace(min_ary, max_ary, len(ary))
            csi = CubicSpline(x, ary)
            ary = csi(np.linspace(min_ary + 0.001, max_ary - 0.001, len(ary))).reshape(ary_shape)
        out = self._float_rankdata(ary).reshape(ary_shape)
        if relative:
            return out / out.size
        return out

    def _get_bininfo(self, values, bins="arviz"):
        dtype = values.dtype.kind

        if isinstance(bins, str) and bins != "arviz":
            bins = np.histogram_bin_edges(values, bins=bins)

        if isinstance(bins, np.ndarray):
            return bins[0], bins[-1], bins[1] - bins[0]

        if dtype == "i":
            x_min = values.min().astype(int)
            x_max = values.max().astype(int)
        else:
            x_min = values.min().astype(float)
            x_max = values.max().astype(float)

        if isinstance(bins, int):
            width = (x_max - x_min) / bins
            if dtype == "i":
                width = max(1, width)
            return x_min, x_max, width

        # Sturges histogram bin estimator
        width_sturges = (x_max - x_min) / (np.log2(values.size) + 1)

        # The Freedman-Diaconis histogram bin estimator.
        iqr = np.subtract(*self.quantile(values, [0.75, 0.25]))  # pylint: disable=assignment-from-no-return
        width_fd = 2 * iqr * values.size ** (-1 / 3)

        if dtype == "i":
            width = np.round(np.max([1, width_sturges, width_fd])).astype(int)
        else:
            width = np.max([width_sturges, width_fd])

        return x_min, x_max, width

    def _get_bins(self, values, bins="arviz"):
        """
        Automatically compute the number of bins for histograms.

        Parameters
        ----------
        values : array_like
        bins : int, str or array_like, default "arviz"
            If `bins` "arviz", use ArviZ default rule (explained in detail in notes),
            if it is a different string it is passed to :func:`numpy.histogram_bin_edges`.
            If `bins` is an integer it is interpreted as the number of bins, however,
            if `values` holds discrete data, there is an extra check to prevent
            the width of the bins to be smaller than ``1``.
            If it is an array it is returned as it.

        Returns
        -------
        array with the bins

        Notes
        -----
        Computes the width of the bins by taking the maximum of the Sturges and
        the Freedman-Diaconis estimators.
        According to numpy `np.histogram` this provides good all around performance.

        The Sturges is a very simplistic estimator based on the assumption of normality of the data.
        This estimator has poor performance for non-normal data, which becomes especially obvious
        for large data sets. The estimate depends only on size of the data.

        The Freedman-Diaconis rule uses interquartile range (IQR) to estimate the binwidth.
        It is considered a robust version of the Scott rule as the IQR is less affected by outliers
        than the standard deviation. However, the IQR depends on fewer points than the standard
        deviation, so it is less accurate, especially for long tailed distributions.
        """
        dtype = values.dtype.kind

        x_min, x_max, width = self._get_bininfo(values, bins)

        if dtype == "i":
            bins = np.arange(x_min, x_max + width + 1, width)
        else:
            if np.isclose(x_min, x_max):
                width = 1e-3
            bins = np.arange(x_min, x_max + width, width)

        return bins

    # pylint: disable=redefined-builtin
    def _histogram(self, ary, bins=None, range=None, weights=None, density=None):
        if bins is None:
            bins = self._get_bins(ary)
        return np.histogram(ary, bins=bins, range=range, weights=weights, density=density)

    def _hdi_linear_nearest_common(self, ary, prob):  # pylint: disable=no-self-use
        n = len(ary)

        ary = np.sort(ary)
        interval_idx_inc = int(np.floor(prob * n))
        n_intervals = n - interval_idx_inc
        interval_width = np.subtract(ary[interval_idx_inc:], ary[:n_intervals], dtype=np.float64)

        if len(interval_width) == 0:
            raise ValueError("Too few elements for interval calculation. ")

        min_idx = np.argmin(interval_width)
        hdi_interval = ary[[min_idx, min_idx + interval_idx_inc]]

        return hdi_interval

    def _hdi_nearest(self, ary, prob, circular, skipna):
        """Compute HDI over the flattened array as closest samples that contain the given prob."""
        ary = ary.flatten()
        if skipna:
            nans = np.isnan(ary)
            if not nans.all():
                ary = ary[~nans]

        if circular:
            mean = self.circular_mean(ary)
            ary = self._circular_standardize(ary - mean)

        hdi_interval = self._hdi_linear_nearest_common(ary, prob)

        if circular:
            hdi_interval = self._circular_standardize(hdi_interval + mean)

        return hdi_interval

    def _hdi_multimodal_continuous(
        self, ary, prob, skipna, max_modes, circular, from_sample=False, **kwargs
    ):
        """Compute HDI if the distribution is multimodal."""
        ary = ary.flatten()
        if skipna:
            ary = ary[~np.isnan(ary)]

        bins, density, _ = self.kde(ary, circular=circular, **kwargs)
        if from_sample:
            ary_density = np.interp(ary, bins, density)
            hdi_intervals, interval_probs = self._hdi_from_point_densities(
                ary, ary_density, prob, circular
            )
        else:
            dx = (bins[-1] - bins[0]) / (len(bins) - 1)
            bin_probs = density * dx

            hdi_intervals, interval_probs = self._hdi_from_bin_probabilities(
                bins, bin_probs, prob, circular, dx
            )

        return self._pad_hdi_to_maxmodes(hdi_intervals, interval_probs, max_modes)

    def _hdi_multimodal_discrete(self, ary, prob, max_modes, bins=None):
        """Compute HDI if the distribution is multimodal."""
        ary = ary.flatten()

        if bins is None:
            bins, counts = np.unique(ary, return_counts=True)
            bin_probs = counts / len(ary)
            dx = 1
        else:
            counts, edges = self._histogram(ary, bins=bins)
            bins = 0.5 * (edges[1:] + edges[:-1])
            bin_probs = counts / counts.sum()
            dx = bins[1] - bins[0]

        hdi_intervals, interval_probs = self._hdi_from_bin_probabilities(
            bins, bin_probs, prob, False, dx
        )

        return self._pad_hdi_to_maxmodes(hdi_intervals, interval_probs, max_modes)

    def _hdi_from_point_densities(self, points, densities, prob, circular):
        if circular:
            points = self._circular_standardize(points)

        sorted_idx = np.argsort(points)
        points = points[sorted_idx]
        densities = densities[sorted_idx]

        # find idx of points in the interval
        interval_size = int(np.ceil(prob * len(points)))
        sorted_idx = np.argsort(densities)[::-1]
        idx_in_interval = sorted_idx[:interval_size]
        idx_in_interval.sort()

        # find idx of interval bounds
        probs_in_interval = np.full(idx_in_interval.shape, 1 / len(points))
        interval_bounds_idx, interval_probs = self._interval_points_to_bounds(
            idx_in_interval, probs_in_interval, 1, circular, period=len(points)
        )

        return points[interval_bounds_idx], interval_probs

    def _hdi_from_bin_probabilities(self, bins, bin_probs, prob, circular, dx):
        if circular:
            bins = self._circular_standardize(bins)
            sorted_idx = np.argsort(bins)
            bins = bins[sorted_idx]
            bin_probs = bin_probs[sorted_idx]

        # find idx of bins in the interval
        sorted_idx = np.argsort(bin_probs)[::-1]
        cum_probs = bin_probs[sorted_idx].cumsum()
        interval_size = np.searchsorted(cum_probs, prob, side="left") + 1
        idx_in_interval = sorted_idx[:interval_size]
        idx_in_interval.sort()

        # get points in intervals
        intervals = bins[idx_in_interval]
        probs_in_interval = bin_probs[idx_in_interval]

        return self._interval_points_to_bounds(intervals, probs_in_interval, dx, circular)

    def _interval_points_to_bounds(self, points, probs, dx, circular, period=2 * np.pi):  # pylint: disable=no-self-use
        cum_probs = probs.cumsum()

        is_bound = np.diff(points) > dx * 1.01
        is_lower_bound = np.insert(is_bound, 0, True)
        is_upper_bound = np.append(is_bound, True)
        interval_bounds = np.column_stack([points[is_lower_bound], points[is_upper_bound]])
        interval_probs = (
            cum_probs[is_upper_bound] - cum_probs[is_lower_bound] + probs[is_lower_bound]
        )

        if (
            circular
            and np.mod(dx * 1.01 + interval_bounds[-1, -1] - interval_bounds[0, 0], period)
            <= dx * 1.01
        ):
            interval_bounds[-1, 1] = interval_bounds[0, 1]
            interval_bounds = interval_bounds[1:, :]
            interval_probs[-1] += interval_probs[0]
            interval_probs = interval_probs[1:]

        return interval_bounds, interval_probs

    def _pad_hdi_to_maxmodes(self, hdi_intervals, interval_probs, max_modes):  # pylint: disable=no-self-use
        if hdi_intervals.shape[0] > max_modes:
            warnings.warn(
                f"found more modes than {max_modes}, returning only the {max_modes} highest "
                "probability modes"
            )
            hdi_intervals = hdi_intervals[np.argsort(interval_probs)[::-1][:max_modes], :]
        elif hdi_intervals.shape[0] < max_modes:
            hdi_intervals = np.vstack(
                [hdi_intervals, np.full((max_modes - hdi_intervals.shape[0], 2), np.nan)]
            )
        return hdi_intervals

    def _mode(self, ary):  # pylint: disable=no-self-use
        ary = ary.flatten()

        if ary.size == 0:
            return np.nan
        if ary.size == 1:
            return ary.item()

        ary = ary[~np.isnan(ary)]

        if ary.dtype.kind == "f":
            # For continuous data, we use the half-sample mode algorithm.
            x = np.sort(ary)
            while len(x) > 2:
                n = (len(x) + 1) // 2
                widths = x[n:] - x[:-n]
                min_idx = np.argmin(widths)
                x = x[min_idx : min_idx + n]

            return (x[0] + x[1]) / 2
        # For discrete data, we use the most frequent value.
        vals, cnts = np.unique(ary, return_counts=True)
        return vals[cnts.argmax()]
