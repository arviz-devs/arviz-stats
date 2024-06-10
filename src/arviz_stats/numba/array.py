"""Array numba functions."""

import numpy as np
from numba import guvectorize, jit

from arviz_stats.base.array import BaseArray


def process_ary_axes(ary, axes):
    """Process input array and axes to ensure input core dims are flattened into the last axis.

    Parameters
    ----------
    ary : array_like
    axes : int or sequence of int
    """
    if isinstance(axes, int):
        axes = [axes]
    axes = [ax if ax >= 0 else ary.ndim + ax for ax in axes]
    reordered_axes = [i for i in range(ary.ndim) if i not in axes] + list(axes)
    ary = np.transpose(ary, axes=reordered_axes)
    ary = ary.reshape((*ary.shape[: -len(axes)], -1))
    return ary, axes


@guvectorize(
    "void(float64[:], float64[:], float64[:])",
    "(n),(m)->(m)",
    target="parallel",
    cache="true",
    nopython=True,
)
def _quantile_ufunc(a, q, out):
    out[:] = 0
    aux = np.quantile(a, q)
    for i in np.ndindex(out.shape):
        out[i] = aux[i]


@jit(nopython=True)
def _histogram_jit(ary, bins, range=None):  # pylint: disable=redefined-builtin
    return np.histogram(ary, bins, range)


class NumbaArray(BaseArray):
    """Class with numba accelerated/guvectorized functions that take array inputs."""

    def __init__(self):
        super().__init__()
        self._kde_ufunc = None

    def quantile(self, ary, quantile, method="linear", **kwargs):
        """Compute the quantile."""
        if method != "linear":
            return super().quantile(ary, quantile, method=method, **kwargs)

        axes = kwargs.pop("axis", None)
        if axes is not None:
            ary, axes = process_ary_axes(ary, axes)
            kwargs["axes"] = [(-1,), (0,), (0,)]
        else:
            ary = ary.ravel()

        result = _quantile_ufunc(ary, quantile, **kwargs)
        if np.ndim(quantile) == 0:
            return result
        return np.moveaxis(result, 0, -1)

    def histogram(
        self, ary, bins=None, range=None, weights=None, density=None
    ):  # pylint: disable=redefined-builtin
        """Compute the histogram of the data."""
        if bins is None:
            bins = self.get_bins(ary)
        if weights is not None:
            raise NotImplementedError("numba doesn't support the weights argument")
        hist, bin_edges = _histogram_jit(ary, bins, range)
        if density:
            hist = hist / (len(ary) * (bin_edges[1:] - bin_edges[:-1]))
        return hist, bin_edges

    @property
    def kde_ufunc(self):
        """Property to cache the guvectorized kde function.

        Notes
        -----
        The function being guvectorized is the kde method of the class,
        so it can't be done until the class is initialized.
        """
        if self._kde_ufunc is None:
            # pylint: disable=line-too-long, unused-argument
            @guvectorize(
                [
                    "void(float64[:],float64[:],boolean,UnicodeCharSeq(1),boolean,float64[:],float64[:],float64)",
                    "void(float64[:],float64[:],boolean,float64,boolean,float64[:],float64[:],float64)",
                ],
                "(n),(m),(),(),()->(m),(m),()",
                cache=True,
                target="parallel",
                nopython=False,
                forceobj=True,
            )
            def kde_gufunc(a, grid_in, circular, bw, adaptive, grid, pdf, bw_out):
                grid[:] = 0
                pdf[:] = 0
                bw_out = 0
                bw = {"t": "scott", "e": "experimental", "i": "isj", "s": "silverman"}.get(bw, bw)
                grid_aux, pdf_aux, bw_aux = self.kde_linear(
                    a, bw=bw, adaptive=adaptive, grid_len=len(grid_in)
                )
                for i in np.ndindex(grid.shape):
                    grid[i] = grid_aux[i]
                    pdf[i] = pdf_aux[i]
                bw_out = bw_aux

            self._kde_ufunc = kde_gufunc
        return self._kde_ufunc

    def kde(self, ary, axes=-1, circular=False, grid_len=512, **kwargs):
        """Compute the guvectorized kde."""
        if axes is not None:
            ary, axes = process_ary_axes(ary, axes)
            kwargs["axes"] = [(-1,), (0,), (), (), ()]
        else:
            ary = ary.ravel()
        bw = kwargs.get("bw", "experimental")
        bw = {"scott": "t", "experimental": "e", "isj": "i", "silverman": "s"}.get(bw, bw)
        adaptive = kwargs.get("adaptive", False)

        return self.kde_ufunc(ary, np.empty(grid_len), circular, bw, adaptive)


array_stats = NumbaArray()
