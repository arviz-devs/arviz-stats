"""Class with array functions.

"array" functions work on any dimension array,
batching as necessary.
"""

import warnings

import numpy as np

from arviz_stats.base.density import _DensityBase
from arviz_stats.base.diagnostics import _DiagnosticsBase
from arviz_stats.base.stats_utils import make_ufunc


def process_chain_none(ary, chain_axis, draw_axis):
    """Process array with chain and draw axis to cover the case ``chain_axis=None``."""
    if chain_axis is None:
        ary = np.expand_dims(ary, axis=0)
        chain_axis = 0
        draw_axis = draw_axis + 1 if draw_axis > 0 else draw_axis
    return ary, chain_axis, draw_axis


def process_chain_none_multi(*arys, chain_axis, draw_axis):
    """Process multiple arrays with chain and draw axis to cover the case ``chain_axis=None``."""
    if chain_axis is None:
        arys = tuple(np.expand_dims(ary, axis=0) for ary in arys)
        chain_axis = 0
        draw_axis = draw_axis + 1 if draw_axis >= 0 else draw_axis
    return (*arys, chain_axis, draw_axis)


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
        axis=-1,
        method="nearest",
        circular=False,
        max_modes=10,
        skipna=False,
        **kwargs,
    ):
        """Compute highest density interval (HDI) on an array of samples.

        See docstring of :func:`arviz_stats.hdi` for full description of computation
        and arguments.

        Parameters
        ----------
        ary : array-like
        prob : float
        axis : int, sequence of int or None, default -1
        method : str, default "nearest"
            Valid options are "nearest", "multimodal" or "multimodal_sample"
        circular : bool, default False
        max_modes : int, default 10
        skipna : bool, default False
        **kwargs
            Only used for multimodal methods with continuous data.
            Passed to kde computation with a ``bw`` default of "taylor" for
            circular data, "isj" otherwise.
        """
        if not 1 >= prob > 0:
            raise ValueError("The value of `prob` must be in the (0, 1] interval.")
        ary, axes = process_ary_axes(ary, axis)
        is_discrete = np.issubdtype(ary.dtype, np.integer) or np.issubdtype(ary.dtype, np.bool_)
        is_multimodal = method.startswith("multimodal")
        if is_multimodal and circular and is_discrete:
            raise ValueError("Multimodal hdi not supported for discrete circular data.")
        if is_discrete and method == "multimodal_sample":
            raise ValueError("Method multimodal_sample not supported for discrete data.")
        hdi_func = {
            "nearest": self._hdi_nearest,
            "multimodal": (
                self._hdi_multimodal_discrete if is_discrete else self._hdi_multimodal_continuous
            ),
            "multimodal_sample": self._hdi_multimodal_continuous,
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
        else:
            extra_kwargs = list(kwargs.keys())
            if len(extra_kwargs) == 1:
                raise TypeError(f"hdi got an unexpected keyword argument: '{extra_kwargs[0]}'")
            if len(extra_kwargs) > 1:
                raise TypeError(f"hdi got unexpected keyword arguments: {extra_kwargs}")

        if method == "multimodal_sample":
            func_kwargs["from_sample"] = True

        result = hdi_array(ary, **func_kwargs)
        if is_multimodal:
            mode_mask = [np.all(np.isnan(result[..., i, :])) for i in range(result.shape[-2])]
            result = result[..., ~np.array(mode_mask), :]
        return result

    def ess(self, ary, chain_axis=-2, draw_axis=-1, method="bulk", relative=False, prob=None):
        """Compute of ess on array-like inputs.

        See docstring of :func:`arviz_stats.ess` for full description of computation
        and arguments.

        Parameters
        ----------
        ary : array-like
        chain_axis : int, default -2
        draw_axis : int, default -1
        method : str, default "bulk"
        relative : bool, default False
        prob : float or (float, float), default None
            When using the array interface, `prob` is a required argument for
            the "tail", "quantile" (as float) and "local" (as tuple) methods.
        """
        method = method.lower()
        # fmt: off
        valid_methods = {
            "bulk", "tail", "mean", "sd", "quantile", "local", "median", "mad",
            "z_scale", "folded", "identity"
        }
        # fmt: on
        if method not in valid_methods:
            raise ValueError(f"Requested method '{method}' but it must be one of {valid_methods}")
        ary, chain_axis, draw_axis = process_chain_none(ary, chain_axis, draw_axis)
        ary, _ = process_ary_axes(ary, [chain_axis, draw_axis])
        ess_func = getattr(self, f"_ess_{method}")
        func_kwargs = {"relative": relative}
        if prob is not None:
            func_kwargs["prob"] = prob
        ess_array = make_ufunc(ess_func, n_output=1, n_input=1, n_dims=2, ravel=False)
        return ess_array(ary, **func_kwargs)

    def rhat(self, ary, chain_axis=-2, draw_axis=-1, method="rank"):
        """Compute of rhat on array-like inputs.

        See docstring of :func:`arviz_stats.rhat` for full description of computation
        and arguments.

        Parameters
        ----------
        ary : array-like
        chain_axis : int, default -2
        draw_axis : int, default -1
        method : str, default "rank"
        """
        method = method.lower()
        valid_methods = {"rank", "folded", "z_scale", "split", "identity"}
        if method not in valid_methods:
            raise ValueError(f"Requested method '{method}' but it must be one of {valid_methods}")
        ary, chain_axis, draw_axis = process_chain_none(ary, chain_axis, draw_axis)
        ary, _ = process_ary_axes(ary, [chain_axis, draw_axis])
        rhat_func = getattr(self, f"_rhat_{method}")
        rhat_array = make_ufunc(rhat_func, n_output=1, n_input=1, n_dims=2, ravel=False)
        return rhat_array(ary)

    def rhat_nested(self, ary, superchain_ids, method="rank", chain_axis=-2, draw_axis=-1):
        """Compute nested rhat on array-like inputs.

        See docstring of :func:`arviz_stats.rhat_nested` for full description of computation
        and arguments.

        Parameters
        ----------
        ary : array-like
        superchain_ids : array-like
        chain_axis : int, default -2
        draw_axis : int, default -1
        method : str, default "rank"
        """
        method = method.lower()
        valid_methods = {"rank", "folded", "z_scale", "split", "identity"}
        if method not in valid_methods:
            raise ValueError(f"Requested method '{method}' but it must be one of {valid_methods}")
        ary, chain_axis, draw_axis = process_chain_none(ary, chain_axis, draw_axis)
        ary, _ = process_ary_axes(ary, [chain_axis, draw_axis])
        rhat_func = getattr(self, f"_rhat_nested_{method}")
        rhat_ufunc = make_ufunc(rhat_func, n_output=1, n_input=1, n_dims=2, ravel=False)
        return rhat_ufunc(ary, superchain_ids=superchain_ids)

    def mcse(self, ary, chain_axis=-2, draw_axis=-1, method="mean", prob=None):
        """Compute of mcse on array-like inputs.

        See docstring of :func:`arviz_stats.mcse` for full description of computation
        and arguments.

        Parameters
        ----------
        ary : array-like
        chain_axis : int, default -2
        draw_axis : int, default -1
        method : str, default "mean"
        prob : float, default None
            When using the array interface, `prob` is a required argument for
            "quantile" method.
        """
        method = method.lower()
        valid_methods = {"mean", "sd", "median", "quantile"}
        if method not in valid_methods:
            raise ValueError(f"Requested method '{method}' but it must be one of {valid_methods}")
        ary, chain_axis, draw_axis = process_chain_none(ary, chain_axis, draw_axis)
        ary, _ = process_ary_axes(ary, [chain_axis, draw_axis])
        mcse_func = getattr(self, f"_mcse_{method}")
        func_kwargs = {} if prob is None else {"prob": prob}
        mcse_array = make_ufunc(mcse_func, n_output=1, n_input=1, n_dims=2, ravel=False)
        return mcse_array(ary, **func_kwargs)

    def pareto_min_ss(self, ary, chain_axis=-2, draw_axis=-1):
        """Compute minimum effective sample size.

        See docstring of :func:`arviz_stats.pareto_min_ss` for full description of computation
        and arguments.

        Parameters
        ----------
        ary : array-like
        chain_axis : int, default -2
        draw_axis : int, default -1
        """
        ary, chain_axis, draw_axis = process_chain_none(ary, chain_axis, draw_axis)
        ary, _ = process_ary_axes(ary, [chain_axis, draw_axis])
        pms_array = make_ufunc(self._pareto_min_ss, n_output=1, n_input=1, n_dims=2, ravel=False)
        return pms_array(ary)

    def psislw(self, ary, r_eff=1, axis=-1):
        """Compute log weights for Pareto-smoothed importance sampling (PSIS) method.

        Parameters
        ----------
        ary : array-like
        r_eff : float, default 1
        axis : int, sequence of int or None, default -1

        Returns
        -------
        log_weights : array-like
            Same shape as `ary` but `axis` dimensions moved to the end
        khat : array-like
            Shape of `ary` minus dimensions indicated in `axis`
        """
        ary, axes = process_ary_axes(ary, axis)
        psl_ufunc = make_ufunc(
            self._psislw,
            n_output=2,
            n_input=1,
            n_dims=len(axes),
            ravel=False,
        )
        return psl_ufunc(ary, out_shape=[(ary.shape[i] for i in axes), []], r_eff=r_eff)

    def bfmi(self, ary, chain_axis=-2, draw_axis=-1):
        """Calculate the estimated Bayesian fraction of missing information."""
        ary, chain_axis, draw_axis = process_chain_none(ary, chain_axis, draw_axis)
        ary, _ = process_ary_axes(ary, [chain_axis, draw_axis])
        bfmi_ufunc = make_ufunc(
            self._bfmi,
            n_output=1,
            n_input=1,
            n_dims=2,
            ravel=False,
        )
        return bfmi_ufunc(ary, out_shape=(ary.shape[chain_axis],))

    def pareto_khat(
        self, ary, chain_axis=-2, draw_axis=-1, r_eff=None, tail="both", log_weights=False
    ):
        """Compute Pareto k-hat diagnostic.

        Parameters
        ----------
        ary : array-like
        chain_axis : int, default -2
        draw_axis : int, default -1
        r_eff : float
        tail : str
        log_weights : bool
        """
        ary, chain_axis, draw_axis = process_chain_none(ary, chain_axis, draw_axis)
        ary, _ = process_ary_axes(ary, [chain_axis, draw_axis])

        def _khat_wrapper(current_slice):
            return self._pareto_khat(
                current_slice, r_eff=r_eff, tail=tail, log_weights=log_weights
            )[1]

        khat_ufunc = make_ufunc(_khat_wrapper, n_output=1, n_input=1, n_dims=2, ravel=False)
        return khat_ufunc(ary, out_shape=())

    def power_scale_lw(self, ary, alpha=0, axis=-1):
        """Compute log weights for power-scaling component by alpha.

        Parameters
        ----------
        ary : array-like
        alpha : float, default 0
        axis : int, sequence of int or None, default -1
        """
        ary, axes = process_ary_axes(ary, axis)
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
        """Compute power-scaling sensitivity.

        Parameters
        ----------
        ary, lower_w, upper_w : array-like
            All 3 input arrays should have the same shape
        lower_alpha, upper_alpha : float
        chain_axis : int, default -2
        draw_axis : int, default -1
        """
        ary, chain_axis, draw_axis = process_chain_none(ary, chain_axis, draw_axis)
        lower_w, _, _ = process_chain_none(lower_w, chain_axis, draw_axis)
        upper_w, _, _ = process_chain_none(upper_w, chain_axis, draw_axis)
        ary, _ = process_ary_axes(ary, [chain_axis, draw_axis])
        lower_w, _ = process_ary_axes(lower_w, [chain_axis, draw_axis])
        upper_w, _ = process_ary_axes(upper_w, [chain_axis, draw_axis])
        pss_array = make_ufunc(
            self._power_scale_sense, n_output=1, n_input=3, n_dims=2, ravel=False
        )
        return pss_array(ary, lower_w, upper_w, lower_alpha=lower_alpha, upper_alpha=upper_alpha)

    def compute_ranks(self, ary, axis=-1, relative=False):
        """Compute ranks of MCMC samples.

        Parameters
        ----------
        ary : array-like
        axis : int, sequence of int or None, default -1
        relative : bool, default False
        """
        ary, axes = process_ary_axes(ary, axis)
        compute_ranks_ufunc = make_ufunc(
            self._compute_ranks,
            n_output=1,
            n_input=1,
            n_dims=len(axes),
            ravel=False,
        )
        return compute_ranks_ufunc(ary, out_shape=(ary.shape[i] for i in axes), relative=relative)

    def get_bins(self, ary, axis=-1, bins="arviz"):
        """Compute default bins.

        Parameters
        ----------
        ary : array-like
        axis : int, sequence of int or None, default -1
        bins : str, scalar or array-like, default "arviz"
        """
        ary, axes = process_ary_axes(ary, axis)
        get_bininfo_ufunc = make_ufunc(
            self._get_bininfo,
            n_output=3,
            n_input=1,
            n_dims=len(axes),
        )
        # TODO: improve handling of array_like bins
        x_min, x_max, width = get_bininfo_ufunc(ary, bins=bins)
        mask = np.isfinite(width) & (width > 0)

        valid_n_bins = np.ceil((x_max[mask] - x_min[mask]) / width[mask])
        n_bins = np.ceil(np.mean(valid_n_bins)).astype(int)
        return np.moveaxis(np.linspace(x_min, x_max, n_bins + 1), 0, -1)

    # pylint: disable=redefined-builtin, too-many-return-statements
    # noqa: PLR0911
    def histogram(self, ary, bins=None, range=None, weights=None, axis=-1, density=None):
        """Compute histogram over provided axis.

        Parameters
        ----------
        ary : array-like
        bins : str, scalar or array-like, optional
        range : (float, float), optional
        weights : array-like, optional
        axis : int, sequence of int or None, default -1
        density : bool, optional

        Returns
        -------
        hist, bin_edges : array_like
            The shape of `hist` will be that of `ary` minus the dimensions in `axis`
            plus an extra dimension of length ``nbins``, same for `bin_edges` with
            the difference the extra dimension has length ``nbins+1``.
        """
        if isinstance(axis, int):
            axis = [axis]
        axes = [ax if ax >= 0 else ary.ndim + ax for ax in axis]
        reordered_axes = [i for i in np.arange(ary.ndim) if i not in axes] + list(axes)
        if weights is not None:
            assert ary.shape == weights.shape
            weights = np.transpose(weights, axes=reordered_axes)
        ary = np.transpose(ary, axes=reordered_axes)
        broadcased_shape = ary.shape[: -len(axes)]
        if bins is None:
            bins = self.get_bins(ary, axis=np.arange(-len(axes), 0, dtype=int))
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

    def kde(self, ary, axis=-1, circular=False, grid_len=512, **kwargs):
        """Compute of kde on array-like inputs.

        Parameters
        ----------
        ary : array-like
        axis : int, sequence of int or None, default -1
        circular : bool, default False
        grid_len : int, default 512
        **kwargs

        Returns
        -------
        grid, pdf, bw : array-like
            `grid` and `pdf` will have the same shape: the same as `ary` minus the dimensions
            in `axis` plus an extra dimension of lenght `grid_len`. Same for `bw`
            except it will not have the extra dimension.
        """
        ary, axes = process_ary_axes(ary, axis)
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

    def qds(
        self,
        ary,
        nquantiles=100,
        binwidth=None,
        dotsize=1,
        stackratio=1,
        top_only=False,
        axis=-1,
        **kwargs,
    ):
        """Compute quantile dot."""
        ary, axes = process_ary_axes(ary, axis)
        qd_ufunc = make_ufunc(
            self._qds,
            n_output=3,
            n_input=1,
            n_dims=len(axes),
        )

        out_s = min(nquantiles, np.prod([ary.shape[ax] for ax in axes]))

        return qd_ufunc(
            ary,
            out_shape=((out_s,), (out_s,), ()),
            nquantiles=out_s,
            binwidth=binwidth,
            dotsize=dotsize,
            stackratio=stackratio,
            top_only=top_only,
            **kwargs,
        )

    def ecdf(self, ary, npoints=200, pit=False, axis=-1, **kwargs):
        """Compute empirical cumulative distribution function (ECDF).

        Parameters
        ----------
        ary : array-like
        npoints : int, default 200
            Maximun number of evaluation points to use.
        pit : bool
            If True compute the difference between the ecdf and the uniform ecdf
            and the x values will be normalized to the [0, 1] range.
        axis : int, sequence of int or None, default -1
        **kwargs

        Returns
        -------
        x, y : array-like
            Both `x` and `y` will have the same shape: the same as `ary` minus the dimensions
            in `axis` plus an extra dimension of lenght `npoints`.
        """
        ary, axes = process_ary_axes(ary, axis)
        ecdf_ufunc = make_ufunc(
            self._ecdf,
            n_output=2,
            n_input=1,
            n_dims=len(axes),
        )
        return ecdf_ufunc(
            ary,
            out_shape=((npoints,), (npoints,)),
            npoints=npoints,
            pit=pit,
            **kwargs,
        )

    def bayesian_r2(self, mu_pred, scale=None, scale_kind="sd", circular=False):
        """Compute Bayesian R² for regression models."""
        r2_ufunc = make_ufunc(
            self._bayesian_r2,
            n_output=1,
            n_input=1,
            n_dims=2,
            ravel=False,
        )
        return r2_ufunc(mu_pred, scale, scale_kind, circular, out_shape=(mu_pred.shape[0],))

    def residual_r2(self, y_obs, mu_pred, circular=False):
        """Compute residual R² for Bayesian regression models."""
        r2_ufunc = make_ufunc(
            self._residual_r2,
            n_output=1,
            n_input=1,
            n_dims=1,
            ravel=False,
        )

        return r2_ufunc(y_obs, mu_pred, circular, out_shape=(mu_pred.shape[0],))

    def metrics(self, observed, predicted, kind):
        """Compute metrics for Bayesian regression models."""
        func = getattr(self, f"_{kind}", None)

        metrics_ufunc = make_ufunc(
            func,
            n_output=2,
            n_input=2,
            n_dims=1,
            ravel=False,
        )
        return metrics_ufunc(observed, predicted)

    def thin_factor(self, ary, target_ess=None, reduce_func="mean", chain_axis=-2, draw_axis=-1):
        """Get thinning factor over draw dimension to preserve ESS in samples or target a given ESS.

        Parameters
        ----------
        ary : array-like
        target_ess : int, optional
            By default, the ESS target will be preserving the ESS of all available samples.
            If an integer value is passed, it must be lower than the average ESS of the input
            samples.
        reduce_func : {"mean", "min"}, default "mean"
        chain_axis : int, default -2
        draw_axis : int, default -1

        Returns
        -------
        int
            Thinning factor
        """
        ary, chain_axis, draw_axis = process_chain_none(ary, chain_axis, draw_axis)
        n_samples = ary.shape[chain_axis] * ary.shape[draw_axis]

        bulk_ess = self.ess(ary, chain_axis=chain_axis, draw_axis=draw_axis, method="bulk")
        tail_ess = self.ess(
            ary, chain_axis=chain_axis, draw_axis=draw_axis, method="tail", prob=0.95
        )
        ess = np.minimum(bulk_ess, tail_ess)

        if reduce_func == "mean":
            ess_ave = np.mean(ess)
        elif reduce_func == "min":
            ess_ave = np.min(ess)
        else:
            raise ValueError(
                f"`reduce_func` {reduce_func} not recognized. Valid values are 'mean' or 'min'"
            )

        if target_ess is None:
            target_ess = ess_ave
        if target_ess > ess_ave:
            warnings.warn(
                f"ESS not high enough to reach requested {target_ess} ESS. "
                "Returning 1 so no thinning is applied and "
                f"current ESS average at {ess_ave} is preserved."
            )
            return 1
        if reduce_func == "min":
            return int(np.floor(n_samples / target_ess))
        return int(np.ceil(n_samples / target_ess))

    def thin(self, ary, factor="auto", chain_axis=-2, draw_axis=-1):
        """Perform thinning on array input.

        Parameters
        ----------
        ary : array-like
        factor : str or int, default "auto"
            The thinning factor. If "auto", the thinning factor is computed based on bulk and tail
            effective sample size. If an integer, the thinning factor is set to that value.
        chain_axis : int, default -2
        draw_axis : int, default -1

        Returns
        -------
        array-like
            Thinned array
        """
        ary, chain_axis, draw_axis = process_chain_none(ary, chain_axis, draw_axis)

        if factor == "auto":
            factor = self.thin_factor(ary, chain_axis=chain_axis, draw_axis=draw_axis)

        elif isinstance(factor, (float | int)):
            factor = int(factor)
            if factor == 1:
                return ary
            if factor < 1:
                raise ValueError("factor must be greater than 1")

        slices = [slice(None)] * ary.ndim
        slices[draw_axis] = slice(None, None, factor)

        return ary[tuple(slices)]

    def mean(self, ary, round_to=None, skipna=False, axis=None):
        """Compute mean of values along the specified axis.

        Parameters
        ----------
        values : array-like
            Input array.
        round_to : int or str, optional
            If integer, number of decimal places to round the result. If string of the
            form '2g' number of significant digits to round the result. Defaults to '2g'.
            Use None to return raw numbers.
        skipna : bool, default False
            Whether to ignore NaN values when computing the mean.
        axis : int, sequence of int or None, default -1
            Axis or axes along which to compute the mode.

        Returns
        -------
        mean : array-like
            Mean of the input values along the specified axis.
        """
        ary, axes = process_ary_axes(ary, axis)
        mean_ufunc = make_ufunc(
            self._mean,
            n_output=1,
            n_input=1,
            n_dims=len(axes),
            ravel=False,
        )
        return mean_ufunc(ary, round_to=round_to, skipna=skipna)

    def median(self, ary, round_to=None, skipna=False, axis=None):
        """Compute median of values along the specified axis.

        Parameters
        ----------
        values : array-like
            Input array.
        round_to : int or str, optional
            If integer, number of decimal places to round the result. If string of the
            form '2g' number of significant digits to round the result. Defaults to '2g'.
            Use None to return raw numbers.
        skipna : bool, default False
            Whether to ignore NaN values when computing the median.
        axis : int, sequence of int or None, default -1
            Axis or axes along which to compute the mode.

        Returns
        -------
        median : array-like
            Median of the input values along the specified axis.
        """
        ary, axes = process_ary_axes(ary, axis)
        median_ufunc = make_ufunc(
            self._median,
            n_output=1,
            n_input=1,
            n_dims=len(axes),
            ravel=False,
        )
        return median_ufunc(ary, round_to=round_to, skipna=skipna)

    def mode(self, ary, round_to=None, skipna=False, axis=None):
        """Compute mode of values along the specified axis.

        Parameters
        ----------
        values : array-like
            Input array.
        round_to : int or str, optional
            If integer, number of decimal places to round the result. If string of the
            form '2g' number of significant digits to round the result. Defaults to '2g'.
            Use None to return raw numbers.
        skipna : bool, default False
            Whether to ignore NaN values when computing the mode.
        axis : int, sequence of int or None, default -1
            Axis or axes along which to compute the mode.

        Returns
        -------
        mode : array-like
            Mode of the input values along the specified axis.
        """
        ary, axes = process_ary_axes(ary, axis)
        mode_ufunc = make_ufunc(
            self._mode,
            n_output=1,
            n_input=1,
            n_dims=len(axes),
            ravel=False,
        )
        return mode_ufunc(ary, round_to=round_to, skipna=skipna)

    def loo(
        self,
        ary,
        chain_axis=-2,
        draw_axis=-1,
        r_eff=1.0,
        log_weights=None,
        pareto_k=None,
        log_jacobian=None,
    ):
        """Compute Pareto-smoothed importance sampling leave-one-out cross-validation (PSIS-LOO-CV).

        Parameters
        ----------
        ary : array-like
            Log-likelihood values.
        chain_axis : int, default -2
            Axis for chains.
        draw_axis : int, default -1
            Axis for draws.
        r_eff : float, default 1.0
            Relative effective sample size.
        log_weights : array-like, optional
            Pre-computed PSIS log weights.
        pareto_k : array-like, optional
            Pre-computed Pareto k-hat diagnostic values.
        log_jacobian : array-like, optional
            Log-Jacobian adjustment for variable transformations.

        Returns
        -------
        elpd_i : array-like
            Pointwise expected log predictive density.
        pareto_k : array-like
            Pareto k-hat diagnostic values.
        p_loo_i : array-like
            Pointwise effective number of parameters.
        """
        ary, chain_axis, draw_axis = process_chain_none(ary, chain_axis, draw_axis)
        ary, axes = process_ary_axes(ary, [chain_axis, draw_axis])

        if log_weights is not None:
            log_weights = np.asarray(log_weights)
            log_weights, _ = process_ary_axes(log_weights, [chain_axis, draw_axis])
            log_weights = log_weights.ravel()

        loo_ufunc = make_ufunc(self._loo, n_output=3, n_input=1, n_dims=len(axes))
        return loo_ufunc(
            ary, r_eff=r_eff, log_weights=log_weights, pareto_k=pareto_k, log_jacobian=log_jacobian
        )

    def loo_approximate_posterior(
        self,
        ary,
        log_p,
        log_q,
        chain_axis=-2,
        draw_axis=-1,
        log_jacobian=None,
    ):
        """Compute PSIS-LOO-CV with approximate posterior correction.

        Parameters
        ----------
        ary : array-like
            Log-likelihood values.
        log_p : array-like
            Target log-density values.
        log_q : array-like
            Proposal log-density values.
        chain_axis : int, default -2
            Axis for chains.
        draw_axis : int, default -1
            Axis for draws.
        log_jacobian : float, optional
            Log-Jacobian adjustment for variable transformations.

        Returns
        -------
        elpd_i : array-like
            Pointwise expected log predictive density.
        pareto_k : array-like
            Pareto k-hat diagnostic values.
        p_loo_i : array-like
            Pointwise effective number of parameters.
        """
        ary, log_p, log_q, chain_axis, draw_axis = process_chain_none_multi(
            ary, log_p, log_q, chain_axis=chain_axis, draw_axis=draw_axis
        )

        ary, axes = process_ary_axes(ary, [chain_axis, draw_axis])
        log_p, _ = process_ary_axes(log_p, [chain_axis, draw_axis])
        log_q, _ = process_ary_axes(log_q, [chain_axis, draw_axis])

        loo_approx_ufunc = make_ufunc(
            self._loo_approximate_posterior, n_output=3, n_input=3, n_dims=len(axes)
        )
        return loo_approx_ufunc(ary, log_p, log_q, log_jacobian=log_jacobian)

    def loo_mixture(
        self,
        ary,
        obs_axes,
        chain_axis=-2,
        draw_axis=-1,
        log_jacobian=None,
    ):
        """Compute mixture importance sampling LOO (Mix-IS-LOO).

        Parameters
        ----------
        ary : array-like
            Full log-likelihood array.
        obs_axes : tuple of int
            Axes corresponding to observation dimensions.
        chain_axis : int, default -2
            Axis for chains.
        draw_axis : int, default -1
            Axis for draws.
        log_jacobian : array-like, optional
            Log-Jacobian adjustment for variable transformations.

        Returns
        -------
        elpd_i : array-like
            Pointwise expected log predictive density.
        p_loo_i : array-like
            Pointwise effective number of parameters.
        mix_log_weights : array-like
            Mixture log weights.
        """
        ary, chain_axis, draw_axis = process_chain_none(ary, chain_axis, draw_axis)
        ndim = ary.ndim
        chain_axis = chain_axis % ndim
        draw_axis = draw_axis % ndim
        obs_axes = tuple(ax % ndim for ax in obs_axes)
        sample_axes = (chain_axis, draw_axis)

        return self._mixture(
            ary, obs_axes=obs_axes, sample_axes=sample_axes, log_jacobian=log_jacobian
        )

    def loo_score(
        self,
        ary,
        y_obs,
        log_weights,
        kind="crps",
        chain_axis=-2,
        draw_axis=-1,
    ):
        """Compute CRPS or SCRPS with PSIS-LOO-CV weights.

        Parameters
        ----------
        ary : array-like
            Posterior predictive samples.
        y_obs : array-like
            Observed values.
        log_weights : array-like
            Pre-computed PSIS log weights.
        kind : str, default "crps"
            Score type, either "crps" or "scrps".
        chain_axis : int, default -2
            Axis for chains.
        draw_axis : int, default -1
            Axis for draws.

        Returns
        -------
        scores : array-like
            Score values (negative orientation for maximization).
        """
        ary, log_weights, chain_axis, draw_axis = process_chain_none_multi(
            ary, log_weights, chain_axis=chain_axis, draw_axis=draw_axis
        )
        ary, axes = process_ary_axes(ary, [chain_axis, draw_axis])
        log_weights, _ = process_ary_axes(log_weights, [chain_axis, draw_axis])

        loo_score_ufunc = make_ufunc(self._loo_score, n_output=1, n_input=3, n_dims=len(axes))
        return loo_score_ufunc(ary, y_obs, log_weights, kind)

    def loo_pit(
        self,
        ary,
        y_obs,
        log_weights,
        chain_axis=-2,
        draw_axis=-1,
        random_state=None,
    ):
        """Compute LOO-PIT values with PSIS-LOO-CV weights.

        Parameters
        ----------
        ary : array-like
            Posterior predictive samples.
        y_obs : array-like
            Observed values.
        log_weights : array-like
            Pre-computed PSIS log weights.
        chain_axis : int, default -2
            Axis for chains.
        draw_axis : int, default -1
            Axis for draws.
        random_state : int or Generator, optional
            Random seed or Generator for tie-breaking. If None, uses seed 214.

        Returns
        -------
        pit_values : array-like
            LOO-PIT values in [0, 1].
        """
        ary, log_weights, chain_axis, draw_axis = process_chain_none_multi(
            ary, log_weights, chain_axis=chain_axis, draw_axis=draw_axis
        )
        ary, axes = process_ary_axes(ary, [chain_axis, draw_axis])
        log_weights, _ = process_ary_axes(log_weights, [chain_axis, draw_axis])

        if random_state is None:
            rng = np.random.default_rng(214)
        else:
            rng = np.random.default_rng(random_state)

        loo_pit_ufunc = make_ufunc(self._loo_pit, n_output=1, n_input=3, n_dims=len(axes))
        return loo_pit_ufunc(ary, y_obs, log_weights, rng=rng)

    def loo_expectation(
        self,
        ary,
        log_weights,
        kind="mean",
        chain_axis=-2,
        draw_axis=-1,
    ):
        """Compute weighted expectation with PSIS-LOO-CV weights.

        Parameters
        ----------
        ary : array-like
            Posterior predictive samples.
        log_weights : array-like
            Pre-computed PSIS log weights.
        kind : str, default "mean"
            Type of expectation: "mean", "median", "var", "sd",
            "circular_mean", "circular_var", "circular_sd".
        chain_axis : int, default -2
            Axis for chains.
        draw_axis : int, default -1
            Axis for draws.

        Returns
        -------
        expectation : array-like
            Weighted expectation values.
        """
        ary, log_weights, chain_axis, draw_axis = process_chain_none_multi(
            ary, log_weights, chain_axis=chain_axis, draw_axis=draw_axis
        )
        ary, axes = process_ary_axes(ary, [chain_axis, draw_axis])
        log_weights, _ = process_ary_axes(log_weights, [chain_axis, draw_axis])

        loo_expectation_ufunc = make_ufunc(
            self._loo_expectation, n_output=1, n_input=2, n_dims=len(axes)
        )
        return loo_expectation_ufunc(ary, log_weights, kind)

    def loo_quantile(
        self,
        ary,
        log_weights,
        prob,
        chain_axis=-2,
        draw_axis=-1,
    ):
        """Compute weighted quantile with PSIS-LOO-CV weights.

        Parameters
        ----------
        ary : array-like
            Posterior predictive samples.
        log_weights : array-like
            Pre-computed PSIS log weights.
        prob : float
            Quantile probability in [0, 1].
        chain_axis : int, default -2
            Axis for chains.
        draw_axis : int, default -1
            Axis for draws.

        Returns
        -------
        quantile : array-like
            Weighted quantile values.
        """
        ary, log_weights, chain_axis, draw_axis = process_chain_none_multi(
            ary, log_weights, chain_axis=chain_axis, draw_axis=draw_axis
        )
        ary, axes = process_ary_axes(ary, [chain_axis, draw_axis])
        log_weights, _ = process_ary_axes(log_weights, [chain_axis, draw_axis])

        loo_quantile_ufunc = make_ufunc(self._loo_quantile, n_output=1, n_input=2, n_dims=len(axes))
        return loo_quantile_ufunc(ary, log_weights, prob)

    def loo_summary(self, elpd_i, p_loo_i):
        """Aggregate pointwise LOO values.

        Parameters
        ----------
        elpd_i : array-like
            Pointwise expected log predictive density.
        p_loo_i : array-like
            Pointwise effective number of parameters.

        Returns
        -------
        elpd : float
            Total expected log predictive density.
        elpd_se : float
            Standard error of elpd.
        p_loo : float
            Total effective number of parameters.
        lppd : float
            Log pointwise predictive density.
        """
        return self._summary(elpd_i, p_loo_i)

    def loo_r2(self, y_obs, ypred_loo, n_simulations=4000, circular=False, random_state=42):
        """Compute LOO-adjusted R-squared using Dirichlet-weighted bootstrap.

        Parameters
        ----------
        y_obs : array-like
            Observed values.
        ypred_loo : array-like
            LOO predictions (same shape as y_obs).
        n_simulations : int, default 4000
            Number of Dirichlet-weighted bootstrap samples.
        circular : bool, default False
            Whether the variable is circular (angles in radians).
        random_state : int, default 42
            Random seed for reproducibility.

        Returns
        -------
        loo_r_squared : array-like
            R-squared samples with shape (n_simulations,).
        """
        y_obs = np.asarray(y_obs).ravel()
        ypred_loo = np.asarray(ypred_loo).ravel()

        return self._loo_r2(
            y_obs,
            ypred_loo,
            n_simulations=n_simulations,
            circular=circular,
            random_state=random_state,
        )

    def srs_estimator(self, y_sample, n_data_points):
        """Compute simple random sampling estimate for subsampled LOO.

        Parameters
        ----------
        y_sample : array-like
            Values of the statistic for the subsample, shape (m,).
        n_data_points : int
            Total number of data points (N).

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
        return self._srs_estimator(y_sample, n_data_points)

    def diff_srs_estimator(
        self,
        elpd_loo_i,
        lpd_approx_sample,
        lpd_approx_all,
        n_data_points,
    ):
        """Difference estimator for subsampled LOO.

        Parameters
        ----------
        elpd_loo_i : array-like
            Pointwise ELPD values for the subsample, shape (m,).
        lpd_approx_sample : array-like
            LPD approximation values for the subsample, shape (m,).
        lpd_approx_all : array-like
            LPD approximation values for the full dataset, shape (N,).
        n_data_points : int
            Total number of data points (N).

        Returns
        -------
        elpd_loo_hat : float
            The estimated ELPD using the difference estimator.
        subsampling_se : float
            The standard error due to subsampling uncertainty.
        total_se : float
            The total standard error (approximation + sampling uncertainty).
        """
        elpd_loo_i = np.asarray(elpd_loo_i).ravel()
        lpd_approx_sample = np.asarray(lpd_approx_sample).ravel()
        lpd_approx_all = np.asarray(lpd_approx_all).ravel()

        return self._diff_srs_estimator(
            elpd_loo_i, lpd_approx_sample, lpd_approx_all, n_data_points
        )


array_stats = BaseArray()
