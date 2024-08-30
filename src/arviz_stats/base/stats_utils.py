"""Stats-utility functions for ArviZ."""

import logging
from collections.abc import Sequence

import numpy as np
from scipy.interpolate import CubicSpline

__all__ = ["make_ufunc"]

_log = logging.getLogger(__name__)


def make_ufunc(func, n_dims=2, n_output=1, n_input=1, index=Ellipsis, ravel=True, check_shape=None):
    """Make ufunc from a function taking 1D array input.

    Parameters
    ----------
    func : callable
    n_dims : int, optional
        Number of core dimensions not broadcasted. Dimensions are skipped from the end.
        At minimum n_dims > 0.
    n_output : int, optional
        Select number of results returned by `func`.
        If n_output > 1, ufunc returns a tuple of objects else returns an object.
    n_input : int, optional
        Number of **array** inputs to func, i.e. ``n_input=2`` means that func is called
        with ``func(ary1, ary2, *args, **kwargs)``
    index : int, optional
        Slice ndarray with `index`. Defaults to `Ellipsis`.
    ravel : bool, optional
        If true, ravel the ndarray before calling `func`.
    check_shape: bool, optional
        If false, do not check if the shape of the output is compatible with n_dims and
        n_output. By default, True only for n_input=1. If n_input is larger than 1, the last
        input array is used to check the shape, however, shape checking with multiple inputs
        may not be correct.

    Returns
    -------
    callable
        ufunc wrapper for `func`.
    """
    if n_dims is not None and n_dims < 1:
        raise TypeError("n_dims must be one or higher.")

    if n_input == 1 and check_shape is None:
        check_shape = True
    elif check_shape is None:
        check_shape = False

    def _ufunc(*args, out=None, out_shape=None, shape_from_1st=False, **kwargs):
        """General ufunc for single-output function."""
        arys = args[:n_input]
        if n_dims is None:
            element_shape = arys[0].shape
        else:
            element_shape = arys[0].shape[:-n_dims]
        if shape_from_1st and out_shape is not None:
            raise ValueError("out_shape and shape_from_1st are incompatible")
        if out is None and not shape_from_1st:
            if out_shape is None:
                out = np.empty(element_shape)
            else:
                out = np.empty((*element_shape, *out_shape))
        elif check_shape and not shape_from_1st:
            if out.shape != arys[0].shape[:-n_dims]:
                msg = f"Shape incorrect for `out`: {out.shape}."
                msg += f" Correct shape is {arys[-1].shape[:-n_dims]}"
                raise TypeError(msg)
        for idx in np.ndindex(element_shape):
            arys_idx = [ary[idx].ravel() if ravel else ary[idx] for ary in arys]
            aux = np.asarray(func(*arys_idx, *args[n_input:], **kwargs))[index]
            if idx == ():
                aux = np.squeeze(aux)
            if shape_from_1st:
                out = np.empty((*element_shape, *aux.shape))
                shape_from_1st = False
            out[idx] = aux
        return out

    def _multi_ufunc(*args, out=None, out_shape=None, shape_from_1st=False, **kwargs):
        """General ufunc for multi-output function."""
        arys = args[:n_input]
        element_shape = arys[0].shape[:-n_dims]
        if out is None:
            if out_shape is None:
                out = tuple(np.empty(element_shape) for _ in range(n_output))
            else:
                out = tuple(np.empty((*element_shape, *out_shape[i])) for i in range(n_output))

        elif check_shape:
            raise_error = False
            correct_shape = tuple(element_shape for _ in range(n_output))
            if isinstance(out, tuple):
                out_shape = tuple(item.shape for item in out)
                if out_shape != correct_shape:
                    raise_error = True
            else:
                raise_error = True
                out_shape = "not tuple, type={type(out)}"
            if raise_error:
                msg = f"Shapes incorrect for `out`: {out_shape}."
                msg += f" Correct shapes are {correct_shape}"
                raise TypeError(msg)
        for idx in np.ndindex(element_shape):
            arys_idx = [ary[idx].ravel() if ravel else ary[idx] for ary in arys]
            results = func(*arys_idx, *args[n_input:], **kwargs)
            if shape_from_1st:
                out = tuple(np.empty((*element_shape, *res.shape)) for res in results)
                shape_from_1st = False
            for i, res in enumerate(results):
                if idx == ():
                    res = np.squeeze(res)
                out[i][idx] = np.asarray(res)[index]
        return out

    if n_output > 1:
        ufunc = _multi_ufunc
    else:
        ufunc = _ufunc

    update_docstring(ufunc, func, n_output)
    return ufunc


def update_docstring(ufunc, func, n_output=1):
    """Update ArviZ generated ufunc docstring."""
    module = ""
    name = ""
    docstring = ""
    if hasattr(func, "__module__") and isinstance(func.__module__, str):
        module += func.__module__
    if hasattr(func, "__name__"):
        name += func.__name__
    if hasattr(func, "__doc__") and isinstance(func.__doc__, str):
        docstring += func.__doc__
    ufunc.__doc__ += "\n\n"
    if module or name:
        ufunc.__doc__ += "This function is a ufunc wrapper for "
        ufunc.__doc__ += module + "." + name
        ufunc.__doc__ += "\n"
    ufunc.__doc__ += 'Call ufunc with n_args from xarray against "chain" and "draw" dimensions:'
    ufunc.__doc__ += "\n\n"
    input_core_dims = 'tuple(("chain", "draw") for _ in range(n_args))'
    if n_output > 1:
        output_core_dims = f" tuple([] for _ in range({n_output}))"
        msg = f"xr.apply_ufunc(ufunc, dataset, input_core_dims={input_core_dims}, "
        msg += f"output_core_dims={ output_core_dims})"
    else:
        output_core_dims = ""
        msg = f"xr.apply_ufunc(ufunc, dataset, input_core_dims={input_core_dims})"
    ufunc.__doc__ += msg
    ufunc.__doc__ += "\n\n"
    ufunc.__doc__ += "For example: np.std(data, ddof=1) --> n_args=2"
    if docstring:
        ufunc.__doc__ += "\n\n"
        ufunc.__doc__ += module
        ufunc.__doc__ += name
        ufunc.__doc__ += " docstring:"
        ufunc.__doc__ += "\n\n"
        ufunc.__doc__ += docstring


def logsumexp(ary, *, b=None, b_inv=None, axis=None, keepdims=False, out=None, copy=True):
    """Stable logsumexp when b >= 0 and b is scalar.

    b_inv overwrites b unless b_inv is None.
    """
    # check dimensions for result arrays
    ary = np.asarray(ary)
    if ary.dtype.kind == "i":
        ary = ary.astype(np.float64)
    dtype = ary.dtype.type
    shape = ary.shape
    shape_len = len(shape)
    if isinstance(axis, Sequence):
        axis = tuple(axis_i if axis_i >= 0 else shape_len + axis_i for axis_i in axis)
        agroup = axis
    else:
        axis = axis if (axis is None) or (axis >= 0) else shape_len + axis
        agroup = (axis,)
    shape_max = (
        tuple(1 for _ in shape)
        if axis is None
        else tuple(1 if i in agroup else d for i, d in enumerate(shape))
    )
    # create result arrays
    if out is None:
        if not keepdims:
            out_shape = (
                tuple()
                if axis is None
                else tuple(d for i, d in enumerate(shape) if i not in agroup)
            )
        else:
            out_shape = shape_max
        out = np.empty(out_shape, dtype=dtype)
    if b_inv == 0:
        return np.full_like(out, np.inf, dtype=dtype) if out.shape else np.inf
    if b_inv is None and b == 0:
        return np.full_like(out, -np.inf) if out.shape else -np.inf
    ary_max = np.empty(shape_max, dtype=dtype)
    # calculations
    ary.max(axis=axis, keepdims=True, out=ary_max)
    if copy:
        ary = ary.copy()
    ary -= ary_max
    np.exp(ary, out=ary)
    ary.sum(axis=axis, keepdims=keepdims, out=out)
    np.log(out, out=out)
    if b_inv is not None:
        ary_max -= np.log(b_inv)
    elif b:
        ary_max += np.log(b)
    out += ary_max if keepdims else ary_max.squeeze()
    # transform to scalar if possible
    return out if out.shape else dtype(out)


def not_valid(ary, check_nan=True, check_shape=True, nan_kwargs=None, shape_kwargs=None):
    """Validate ndarray.

    Parameters
    ----------
    ary : numpy.ndarray
    check_nan : bool
        Check if any value contains NaN.
    check_shape : bool
        Check if array has correct shape. Assumes dimensions in order (chain, draw, *shape).
        For 1D arrays (shape = (n,)) assumes chain equals 1.
    nan_kwargs : dict
        Valid kwargs are:
            axis : int,
                Defaults to None.
            how : str, {"all", "any"}
                Default to "any".
    shape_kwargs : dict
        Valid kwargs are:
            min_chains : int
                Defaults to 1.
            min_draws : int
                Defaults to 4.

    Returns
    -------
    bool
    """
    ary = np.asarray(ary)

    nan_error = False
    draw_error = False
    chain_error = False

    # for arviz-plots alignment, if all elements are nan return nan without indicating
    # any error
    isnan = np.isnan(ary)
    if isnan.all():
        return True

    if check_nan:
        if nan_kwargs is None:
            nan_kwargs = {}

        axis = nan_kwargs.get("axis", None)
        if nan_kwargs.get("how", "any").lower() == "all":
            nan_error = isnan.all(axis)
        else:
            nan_error = isnan.any(axis)

        if (isinstance(nan_error, bool) and nan_error) or nan_error.any():
            _log.warning("Array contains NaN-value.")

    if check_shape:
        shape = ary.shape

        if shape_kwargs is None:
            shape_kwargs = {}

        min_chains = shape_kwargs.get("min_chains", 2)
        min_draws = shape_kwargs.get("min_draws", 4)
        error_msg = f"Shape validation failed: input_shape: {shape}, "
        error_msg += f"minimum_shape: (chains={min_chains}, draws={min_draws})"

        chain_error = ((min_chains > 1) and (len(shape) < 2)) or (shape[0] < min_chains)
        draw_error = ((len(shape) < 2) and (shape[0] < min_draws)) or (
            (len(shape) > 1) and (shape[1] < min_draws)
        )

        if chain_error or draw_error:
            _log.warning(error_msg)

    return nan_error | chain_error | draw_error


def smooth_data(obs_vals, pp_vals):
    """Smooth data, helper function for discrete data in plot_pbv, loo_pit and plot_loo_pit."""
    x = np.linspace(0, 1, len(obs_vals))
    csi = CubicSpline(x, obs_vals)
    obs_vals = csi(np.linspace(0.01, 0.99, len(obs_vals)))

    x = np.linspace(0, 1, pp_vals.shape[1])
    csi = CubicSpline(x, pp_vals, axis=1)
    pp_vals = csi(np.linspace(0.01, 0.99, pp_vals.shape[1]))

    return obs_vals, pp_vals
