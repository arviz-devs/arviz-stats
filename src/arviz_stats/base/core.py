"""Core stats functions.

Functions that are needed by multiple "organization classes"
should go here. e.g. fft is used for kde bandwidth estimation and for ess.
"""
import warnings

import numpy as np
from scipy.fftpack import next_fast_len
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
        ary : Numpy array
            An array containing MCMC samples

        Returns
        -------
        acov: Numpy array same size as the input array
        """
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
        ary : Numpy array
            An array containing MCMC samples

        Returns
        -------
        acorr: Numpy array same size as the input array
        """
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

    def quantile(self, ary, quantile, **kwargs):  # pylint: disable=no-self-use
        """Compute the quantile of an array of samples.

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
        result = np.quantile(ary, quantile, **kwargs)
        if np.ndim(quantile) == 0:
            return result
        return np.moveaxis(result, 0, -1)

    def eti(self, ary, prob, **kwargs):
        edge_prob = (1 - prob) / 2
        return self.quantile(ary, [edge_prob, 1 - edge_prob], **kwargs)

    def _get_bininfo(self, values):
        dtype = values.dtype.kind

        if dtype == "i":
            x_min = values.min().astype(int)
            x_max = values.max().astype(int)
        else:
            x_min = values.min().astype(float)
            x_max = values.max().astype(float)

        # Sturges histogram bin estimator
        width_sturges = (x_max - x_min) / (np.log2(values.size) + 1)

        # The Freedman-Diaconis histogram bin estimator.
        iqr = np.subtract(
            *self.quantile(values, [0.75, 0.25])
        )  # pylint: disable=assignment-from-no-return
        width_fd = 2 * iqr * values.size ** (-1 / 3)

        if dtype == "i":
            width = np.round(np.max([1, width_sturges, width_fd])).astype(int)
        else:
            width = np.max([width_sturges, width_fd])

        return x_min, x_max, width

    def _get_bins(self, values):
        """
        Automatically compute the number of bins for histograms.

        Parameters
        ----------
        values = array_like

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

        x_min, x_max, width = self._get_bininfo(values)

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

    def _hdi_linear_nearest_common(self, ary, prob, skipna, circular):
        ary = ary.flatten()
        if skipna:
            nans = np.isnan(ary)
            if not nans.all():
                ary = ary[~nans]
        n = len(ary)

        mean = None
        if circular:
            mean = self.circular_mean(ary)
            ary = ary - mean
            ary = np.arctan2(np.sin(ary), np.cos(ary))

        ary = np.sort(ary)
        interval_idx_inc = int(np.floor(prob * n))
        n_intervals = n - interval_idx_inc
        interval_width = np.subtract(ary[interval_idx_inc:], ary[:n_intervals], dtype=np.float64)

        if len(interval_width) == 0:
            raise ValueError("Too few elements for interval calculation. ")

        min_idx = np.argmin(interval_width)

        return ary, mean, min_idx, interval_idx_inc

    def _hdi_nearest(self, ary, prob, circular, skipna):
        """Compute HDI over the flattened array as closest samples that contain the given prob."""
        ary, mean, min_idx, interval_idx_inc = self._hdi_linear_nearest_common(
            ary, prob, skipna, circular
        )

        hdi_min = ary[min_idx]
        hdi_max = ary[min_idx + interval_idx_inc]

        if circular:
            hdi_min = hdi_min + mean
            hdi_max = hdi_max + mean
            hdi_min = np.arctan2(np.sin(hdi_min), np.cos(hdi_min))
            hdi_max = np.arctan2(np.sin(hdi_max), np.cos(hdi_max))

        hdi_interval = np.array([hdi_min, hdi_max])

        return hdi_interval

    def _hdi_multimodal(self, ary, prob, skipna, max_modes):
        """Compute HDI if the distribution is multimodal."""
        ary = ary.flatten()
        if skipna:
            ary = ary[~np.isnan(ary)]

        if ary.dtype.kind == "f":
            bins, density, _ = self.kde(ary)
            lower, upper = bins[0], bins[-1]
            range_x = upper - lower
            dx = range_x / len(density)
        else:
            bins = self._get_bins(ary)
            density, _ = self._histogram(ary, bins=bins, density=True)
            dx = np.diff(bins)[0]

        density *= dx

        idx = np.argsort(-density)
        intervals = bins[idx][density[idx].cumsum() <= prob]
        intervals.sort()

        intervals_splitted = np.split(intervals, np.where(np.diff(intervals) >= dx * 1.1)[0] + 1)

        hdi_intervals = np.full((max_modes, 2), np.nan)
        for i, interval in enumerate(intervals_splitted):
            if i == max_modes:
                warnings.warn(
                    f"found more modes than {max_modes}, returning only the first {max_modes} modes"
                )
                break
            if interval.size == 0:
                hdi_intervals[i] = np.asarray([bins[0], bins[0]])
            else:
                hdi_intervals[i] = np.asarray([interval[0], interval[-1]])

        return np.array(hdi_intervals)
