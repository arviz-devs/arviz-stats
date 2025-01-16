"""Class with array functions.

"array" functions work on any dimension array,
batching as necessary.
"""

import numpy as np

from arviz_stats.base.density import _DensityBase
from arviz_stats.base.diagnostics import _DiagnosticsBase
from arviz_stats.base.stats_utils import make_ufunc


def process_ary_axes(ary, axes):
    """Process input array and axes to ensure input core dims are the last ones.

    Parameters
    ----------
    ary : array_like
    axes : int or sequence of int
    """
    if axes is None:
        axes = list(range(ary.ndim))
    if isinstance(axes, int):
        axes = [axes]
    axes = [ax if ax >= 0 else ary.ndim + ax for ax in axes]
    reordered_axes = [i for i in range(ary.ndim) if i not in axes] + list(axes)
    ary = np.transpose(ary, axes=reordered_axes)
    return ary, np.arange(-len(axes), 0, dtype=int)


class BaseArray(_DensityBase, _DiagnosticsBase):
    """Class with numpy+scipy only functions that take array inputs.

    Notes
    -----
    If a new dimension is created by the function it must be added at the end of the array.
    Otherwise the functions won't be compatible with :func:`xarray.apply_ufunc`.
    """

    def hdi(
        self,
        ary,
        prob,
        axes=-1,
        method="nearest",
        circular=False,
        max_modes=10,
        skipna=False,
        **kwargs,
    ):
        """Compute HDI function on array-like input."""
        if not 1 >= prob > 0:
            raise ValueError("The value of `prob` must be in the (0, 1] interval.")
        ary, axes = process_ary_axes(ary, axes)
        is_discrete = np.issubdtype(ary.dtype, np.integer) or np.issubdtype(ary.dtype, np.bool_)
        is_multimodal = method.startswith("multimodal")
        if is_multimodal and circular and is_discrete:
            raise ValueError("Multimodal hdi not supported for discrete circular data.")
        hdi_func = {
            "nearest": self._hdi_nearest,
            "multimodal": (
                self._hdi_multimodal_discrete if is_discrete else self._hdi_multimodal_continuous
            ),
            "multimodal_sample": (
                self._hdi_multimodal_discrete if is_discrete else self._hdi_multimodal_continuous
            ),
        }[method]
        hdi_array = make_ufunc(
            hdi_func,
            n_output=1,
            n_input=1,
            n_dims=len(axes),
        )
        func_kwargs = {
            "prob": prob,
            "skipna": skipna,
            "out_shape": (max_modes, 2) if is_multimodal else (2,),
            "circular": circular,
        }
        if is_multimodal:
            func_kwargs["max_modes"] = max_modes
            if is_discrete:
                func_kwargs.pop("circular")
                func_kwargs.pop("skipna")
            else:
                func_kwargs["bw"] = "isj" if not circular else "taylor"
            func_kwargs.update(kwargs)

        if method == "multimodal_sample":
            func_kwargs["from_sample"] = True

        result = hdi_array(ary, **func_kwargs)
        if is_multimodal:
            mode_mask = [np.all(np.isnan(result[..., i, :])) for i in range(result.shape[-2])]
            result = result[..., ~np.array(mode_mask), :]
        return result

    def ess(self, ary, chain_axis=-2, draw_axis=-1, method="bulk", relative=False, prob=None):
        """Compute of ess on array-like inputs."""
        method = method.lower()
        # fmt: off
        valid_methods = {
            "bulk", "tail", "mean", "sd", "quantile", "local", "median", "mad",
            "z_scale", "folded", "identity"
        }
        # fmt: on
        if method not in valid_methods:
            raise ValueError(f"Requested method '{method}' but it must be one of {valid_methods}")
        if chain_axis is None:
            ary = np.expand_dims(ary, axis=0)
            chain_axis = 0
        ary, _ = process_ary_axes(ary, [chain_axis, draw_axis])
        ess_func = getattr(self, f"_ess_{method}")
        func_kwargs = {"relative": relative}
        if prob is not None:
            func_kwargs["prob"] = prob
        ess_array = make_ufunc(ess_func, n_output=1, n_input=1, n_dims=2, ravel=False)
        return ess_array(ary, **func_kwargs)

    def rhat(self, ary, chain_axis=-2, draw_axis=-1, method="rank"):
        """Compute of rhat on array-like inputs."""
        method = method.lower()
        valid_methods = {"rank", "folded", "z_scale", "split", "identity"}
        if method not in valid_methods:
            raise ValueError(f"Requested method '{method}' but it must be one of {valid_methods}")
        if chain_axis is None:
            ary = np.expand_dims(ary, axis=0)
            chain_axis = 0
        ary, _ = process_ary_axes(ary, [chain_axis, draw_axis])
        rhat_func = getattr(self, f"_rhat_{method}")
        rhat_array = make_ufunc(rhat_func, n_output=1, n_input=1, n_dims=2, ravel=False)
        return rhat_array(ary)

    def mcse(self, ary, chain_axis=-2, draw_axis=-1, method="mean", prob=None):
        """Compute of mcse on array-like inputs."""
        method = method.lower()
        valid_methods = {"mean", "sd", "median", "quantile"}
        if method not in valid_methods:
            raise ValueError(f"Requested method '{method}' but it must be one of {valid_methods}")
        if chain_axis is None:
            ary = np.expand_dims(ary, axis=0)
            chain_axis = 0
        ary, _ = process_ary_axes(ary, [chain_axis, draw_axis])
        mcse_func = getattr(self, f"_mcse_{method}")
        func_kwargs = {} if prob is None else {"prob": prob}
        mcse_array = make_ufunc(mcse_func, n_output=1, n_input=1, n_dims=2, ravel=False)
        return mcse_array(ary, **func_kwargs)

    def pareto_min_ss(self, ary, chain_axis=-2, draw_axis=-1):
        """Compute minimum effective sample size."""
        if chain_axis is None:
            ary = np.expand_dims(ary, axis=0)
            chain_axis = 0
        ary, _ = process_ary_axes(ary, [chain_axis, draw_axis])
        pms_array = make_ufunc(self._pareto_min_ss, n_output=1, n_input=1, n_dims=2, ravel=False)
        return pms_array(ary)

    def power_scale_lw(self, ary, alpha=0, axes=-1):
        """Compute ranks of MCMC samples."""
        ary, axes = process_ary_axes(ary, axes)
        psl_ufunc = make_ufunc(
            self._power_scale_lw,
            n_output=1,
            n_input=1,
            n_dims=len(axes),
            ravel=False,
        )
        return psl_ufunc(ary, out_shape=(ary.shape[i] for i in axes), alpha=alpha)

    def power_scale_sense(
        self, ary, lower_w, upper_w, lower_alpha, upper_alpha, chain_axis=-2, draw_axis=-1
    ):
        """Compute power-scaling sensitivity."""
        if chain_axis is None:
            ary = np.expand_dims(ary, axis=0)
            lower_w = np.expand_dims(lower_w, axis=0)
            upper_w = np.expand_dims(upper_w, axis=0)
            chain_axis = 0
        ary, _ = process_ary_axes(ary, [chain_axis, draw_axis])
        lower_w, _ = process_ary_axes(lower_w, [chain_axis, draw_axis])
        upper_w, _ = process_ary_axes(upper_w, [chain_axis, draw_axis])
        pss_array = make_ufunc(
            self._power_scale_sense, n_output=1, n_input=3, n_dims=2, ravel=False
        )
        return pss_array(ary, lower_w, upper_w, lower_alpha=lower_alpha, upper_alpha=upper_alpha)

    def compute_ranks(self, ary, axes=-1, relative=False):
        """Compute ranks of MCMC samples."""
        ary, axes = process_ary_axes(ary, axes)
        compute_ranks_ufunc = make_ufunc(
            self._compute_ranks,
            n_output=1,
            n_input=1,
            n_dims=len(axes),
            ravel=False,
        )
        return compute_ranks_ufunc(ary, out_shape=(ary.shape[i] for i in axes), relative=relative)

    def get_bins(self, ary, axes=-1, bins="arviz"):
        """Compute default bins."""
        ary, axes = process_ary_axes(ary, axes)
        get_bininfo_ufunc = make_ufunc(
            self._get_bininfo,
            n_output=3,
            n_input=1,
            n_dims=len(axes),
        )
        # TODO: improve handling of array_like bins
        x_min, x_max, width = get_bininfo_ufunc(ary, bins=bins)
        n_bins = np.ceil((x_max - x_min) / width)
        n_bins = np.ceil(np.mean(n_bins)).astype(int)
        return np.moveaxis(np.linspace(x_min, x_max, n_bins + 1), 0, -1)

    # pylint: disable=redefined-builtin, too-many-return-statements
    # noqa: PLR0911
    def histogram(self, ary, bins=None, range=None, weights=None, axes=-1, density=None):
        """Compute histogram over provided axes."""
        if isinstance(axes, int):
            axes = [axes]
        axes = [ax if ax >= 0 else ary.ndim + ax for ax in axes]
        reordered_axes = [i for i in np.arange(ary.ndim) if i not in axes] + list(axes)
        if weights is not None:
            assert ary.shape == weights.shape
            weights = np.transpose(weights, axes=reordered_axes)
        ary = np.transpose(ary, axes=reordered_axes)
        broadcased_shape = ary.shape[: -len(axes)]
        if bins is None:
            bins = self.get_bins(ary, axes=np.arange(-len(axes), 0, dtype=int))
        if isinstance(bins, int | str):
            # avoid broadcasting over bins -> can't be positional argument
            if (range is None) or (np.size(range) == 2):
                # avoid broadcasting over range
                if weights is not None:
                    # ensure broadcasting over weights
                    histogram_ufunc = make_ufunc(
                        lambda ary, weights: self._histogram(
                            ary, bins=bins, range=range, weights=weights, density=density
                        ),
                        n_output=2,
                        n_input=2,
                        n_dims=len(axes),
                    )
                    return histogram_ufunc(ary, weights, shape_from_1st=True)
                # avoid broadcasting over weights -> no broadcasting anywhere
                histogram_ufunc = make_ufunc(
                    self._histogram, n_output=2, n_input=1, n_dims=len(axes)
                )
                return histogram_ufunc(
                    ary, bins=bins, range=range, density=density, shape_from_1st=True
                )
            # ensure broadcasting over range
            assert range.shape[:-1] == broadcased_shape
            if weights is not None:
                # ensure broadcasting over weights
                histogram_ufunc = make_ufunc(
                    lambda ary, range, weights: self._histogram(
                        ary, bins=bins, range=range, weights=weights, density=density
                    ),
                    n_output=2,
                    n_input=3,
                    n_dims=len(axes),
                )
                return histogram_ufunc(ary, range, weights, shape_from_1st=True)
            # avoid broadcasting over weights while broadcasting over range
            histogram_ufunc = make_ufunc(
                lambda ary, range: self._histogram(ary, bins=bins, range=range, density=density),
                n_output=2,
                n_input=2,
                n_dims=len(axes),
            )
            return histogram_ufunc(ary, range, shape_from_1st=True)
        # ensure broadcasting over bins
        assert bins.shape[:-1] == broadcased_shape
        if (range is None) or (np.size(range) == 2):
            # avoid broadcasting over range
            if weights is not None:
                # ensure broadcasting over weights
                histogram_ufunc = make_ufunc(
                    lambda ary, bins, weights: self._histogram(
                        ary, bins=bins, range=range, weights=weights, density=density
                    ),
                    n_output=2,
                    n_input=3,
                    n_dims=len(axes),
                )
                return histogram_ufunc(ary, bins, weights, shape_from_1st=True)
            # avoid broadcasting over weights
            histogram_ufunc = make_ufunc(
                lambda ary, bins: self._histogram(ary, bins=bins, range=range, density=density),
                n_output=2,
                n_input=2,
                n_dims=len(axes),
            )
            return histogram_ufunc(ary, bins, shape_from_1st=True)
        # ensure broadcasting over range
        assert range.shape[:-1] == broadcased_shape
        if weights is not None:
            # ensure broadcasting over weights
            histogram_ufunc = make_ufunc(
                lambda ary, bins, range, weights: self._histogram(
                    ary, bins=bins, range=range, weights=weights, density=density
                ),
                n_output=2,
                n_input=4,
                n_dims=len(axes),
            )
            return histogram_ufunc(ary, bins, range, weights, shape_from_1st=True)
        # avoid broadcasting over weights
        histogram_ufunc = make_ufunc(
            lambda ary, bins, range: self._histogram(ary, bins=bins, range=range, density=density),
            n_output=2,
            n_input=3,
            n_dims=len(axes),
        )
        return histogram_ufunc(ary, bins, range, shape_from_1st=True)

    def kde(self, ary, axes=-1, circular=False, grid_len=512, **kwargs):
        """Compute of kde on array-like inputs."""
        ary, axes = process_ary_axes(ary, axes)
        kde_ufunc = make_ufunc(
            self._kde,
            n_output=3,
            n_input=1,
            n_dims=len(axes),
        )
        return kde_ufunc(
            ary,
            out_shape=((grid_len,), (grid_len,), ()),
            grid_len=grid_len,
            circular=circular,
            **kwargs,
        )


array_stats = BaseArray()
