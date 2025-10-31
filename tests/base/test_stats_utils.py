"""Tests for stats_utils."""

# ruff: noqa: E731
#  pylint: disable=unnecessary-lambda-assignment
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from scipy.special import logsumexp

from arviz_stats.base.stats_utils import logsumexp as _logsumexp
from arviz_stats.base.stats_utils import make_ufunc, not_valid


@pytest.mark.parametrize("ary_dtype", [np.float64, np.float32, np.int32, np.int64])
@pytest.mark.parametrize("axis", [None, 0, 1, (-2, -1)])
@pytest.mark.parametrize("b", [None, 0, 1 / 100, 1 / 101])
@pytest.mark.parametrize("keepdims", [True, False])
def test_logsumexp_b(rng, ary_dtype, axis, b, keepdims):  # pylint: disable=invalid-name
    """Test ArviZ implementation of logsumexp.

    Test also compares against Scipy implementation.
    Case where b=None, they are equal. (N=len(ary))
    Second case where b=x, and x is 1/(number of elements), they are almost equal.

    Test tests against b parameter.
    """
    ary = rng.normal(size=(100, 101)).astype(ary_dtype)
    assert _logsumexp(ary=ary, axis=axis, b=b, keepdims=keepdims, copy=True) is not None
    ary = ary.copy()
    assert _logsumexp(ary=ary, axis=axis, b=b, keepdims=keepdims, copy=False) is not None
    out = np.empty(5)
    assert _logsumexp(ary=rng.normal(size=(10, 5)), axis=0, out=out) is not None

    # Scipy implementation
    scipy_results = logsumexp(ary, b=b, axis=axis, keepdims=keepdims)
    arviz_results = _logsumexp(ary, b=b, axis=axis, keepdims=keepdims)

    assert_array_almost_equal(scipy_results, arviz_results)


@pytest.mark.parametrize("ary_dtype", [np.float64, np.float32, np.int32, np.int64])
@pytest.mark.parametrize("axis", [None, 0, 1, (-2, -1)])
@pytest.mark.parametrize("b_inv", [None, 0, 100, 101])
@pytest.mark.parametrize("keepdims", [True, False])
def test_logsumexp_b_inv(rng, ary_dtype, axis, b_inv, keepdims):
    """Test ArviZ implementation of logsumexp.

    Test also compares against Scipy implementation.
    Case where b=None, they are equal. (N=len(ary))
    Second case where b=x, and x is 1/(number of elements), they are almost equal.

    Test tests against b_inv parameter.
    """
    ary = rng.normal(size=(100, 101)).astype(ary_dtype)
    assert _logsumexp(ary=ary, axis=axis, b_inv=b_inv, keepdims=keepdims, copy=True) is not None
    ary = ary.copy()
    assert _logsumexp(ary=ary, axis=axis, b_inv=b_inv, keepdims=keepdims, copy=False) is not None
    out = np.empty(5)
    assert _logsumexp(ary=rng.normal(size=(10, 5)), axis=0, out=out) is not None

    if b_inv != 0:
        # Scipy implementation when b_inv != 0
        b_scipy = 1 / b_inv if b_inv is not None else None
        scipy_results = logsumexp(ary, b=b_scipy, axis=axis, keepdims=keepdims)
        arviz_results = _logsumexp(ary, b_inv=b_inv, axis=axis, keepdims=keepdims)

        assert_array_almost_equal(scipy_results, arviz_results)


@pytest.mark.parametrize("quantile", ((0.5,), (0.5, 0.1)))
@pytest.mark.parametrize("arg", (True, False))
def test_make_ufunc_output(rng, quantile, arg):
    ary = rng.normal(size=(4, 100))
    n_output = len(quantile)
    if arg:
        res = make_ufunc(np.quantile, n_output=n_output)(ary, quantile)
    elif n_output == 1:
        res = make_ufunc(np.quantile)(ary, q=quantile)
    else:
        res = make_ufunc(np.quantile, n_output=n_output)(ary, q=quantile)
    if n_output == 1:
        assert not isinstance(res, tuple)
    else:
        assert isinstance(res, tuple)
        assert len(res) == n_output


@pytest.mark.parametrize("out_shape", ((1, 2), (1, 2, 3), (2, 3, 4, 5)))
@pytest.mark.parametrize("input_dim", ((4, 100), (4, 100, 3), (4, 100, 4, 5)))
def test_make_ufunc_out_shape(out_shape, input_dim):
    func = lambda x: np.random.rand(*out_shape)
    ary = np.ones(input_dim)
    res = make_ufunc(func, n_dims=1)(ary, out_shape=out_shape)
    assert res.shape == (*ary.shape[:-1], *out_shape)


def test_make_ufunc_out_shape_multi_input():
    out_shape = (2, 4)
    func = lambda x, y: np.random.rand(*out_shape)
    ary1 = np.ones((4, 100))
    ary2 = np.ones((4, 5))
    res = make_ufunc(func, n_dims=1)(ary1, ary2, out_shape=out_shape)
    assert res.shape == (*ary1.shape[:-1], *out_shape)


def test_make_ufunc_out_shape_multi_output_same():
    func = lambda x: (np.random.rand(1, 2), np.random.rand(1, 2))
    ary = np.ones((4, 100))
    res1, res2 = make_ufunc(func, n_dims=1, n_output=2)(ary, out_shape=((1, 2), (1, 2)))
    assert res1.shape == (*ary.shape[:-1], 1, 2)
    assert res2.shape == (*ary.shape[:-1], 1, 2)


def test_make_ufunc_out_shape_multi_output_diff():
    func = lambda x: (np.random.rand(5, 3), np.random.rand(10, 4))
    ary = np.ones((4, 100))
    res1, res2 = make_ufunc(func, n_dims=1, n_output=2)(ary, out_shape=((5, 3), (10, 4)))
    assert res1.shape == (*ary.shape[:-1], 5, 3)
    assert res2.shape == (*ary.shape[:-1], 10, 4)


@pytest.mark.parametrize("n_output", (1, 2, 3))
def test_make_ufunc(n_output):
    if n_output == 3:
        func = lambda x: (np.mean(x), np.mean(x), np.mean(x))
    elif n_output == 2:
        func = lambda x: (np.mean(x), np.mean(x))
    else:
        func = np.mean
    ufunc = make_ufunc(func, n_dims=1, n_output=n_output)
    ary = np.ones((4, 100))
    res = ufunc(ary)
    if n_output > 1:
        assert all(len(res_i) == 4 for res_i in res)
        assert all((res_i == 1).all() for res_i in res)
    else:
        assert len(res) == 4
        assert (res == 1).all()


@pytest.mark.parametrize("n_output", (1, 2, 3))
def test_make_ufunc_out(n_output):
    if n_output == 3:
        func = lambda x: (np.mean(x), np.mean(x), np.mean(x))
        res = (np.empty((4,)), np.empty((4,)), np.empty((4,)))
    elif n_output == 2:
        func = lambda x: (np.mean(x), np.mean(x))
        res = (np.empty((4,)), np.empty((4,)))
    else:
        func = np.mean
        res = np.empty((4,))
    ufunc = make_ufunc(func, n_dims=1, n_output=n_output)
    ary = np.ones((4, 100))
    ufunc(ary, out=res)
    if n_output > 1:
        assert all(len(res_i) == 4 for res_i in res)
        assert all((res_i == 1).all() for res_i in res)
    else:
        assert len(res) == 4
        assert (res == 1).all()


def test_make_ufunc_bad_ndim():
    with pytest.raises(TypeError):
        make_ufunc(np.mean, n_dims=0)


@pytest.mark.parametrize("n_output", (1, 2, 3))
def test_make_ufunc_out_bad(n_output):
    if n_output == 3:
        func = lambda x: (np.mean(x), np.mean(x), np.mean(x))
        res = (np.empty((100,)), np.empty((100,)))
    elif n_output == 2:
        func = lambda x: (np.mean(x), np.mean(x))
        res = np.empty((100,))
    else:
        func = np.mean
        res = np.empty((100,))
    ufunc = make_ufunc(func, n_dims=1, n_output=n_output)
    ary = np.ones((4, 100))
    with pytest.raises(TypeError):
        ufunc(ary, out=res)


@pytest.mark.parametrize("how", ("all", "any"))
def test_nan(rng, how):
    assert not not_valid(np.ones(10), check_shape=False, nan_kwargs={"how": how})
    if how == "any":
        assert not_valid(
            np.concatenate((rng.normal(size=100), np.full(2, np.nan))),
            check_shape=False,
            nan_kwargs={"how": how},
        )
    else:
        assert not not_valid(
            np.concatenate((rng.normal(size=100), np.full(2, np.nan))),
            check_shape=False,
            nan_kwargs={"how": how},
        )
        assert not_valid(np.full(10, np.nan), check_shape=False, nan_kwargs={"how": how})


@pytest.mark.parametrize("axis", (-1, 0, 1))
def test_nan_axis(rng, axis):
    data = rng.normal(size=(4, 100))
    data[0, 0] = np.nan  #  pylint: disable=unsupported-assignment-operation
    axis_ = (len(data.shape) + axis) if axis < 0 else axis
    assert not_valid(data, check_shape=False, nan_kwargs={"how": "any"})
    assert not_valid(data, check_shape=False, nan_kwargs={"how": "any", "axis": axis}).any()
    assert not not_valid(data, check_shape=False, nan_kwargs={"how": "any", "axis": axis}).all()
    assert not_valid(
        data, check_shape=False, nan_kwargs={"how": "any", "axis": axis}
    ).shape == tuple(dim for ax, dim in enumerate(data.shape) if ax != axis_)


def test_valid_shape():
    assert not not_valid(
        np.ones((2, 200)), check_nan=False, shape_kwargs={"min_chains": 2, "min_draws": 100}
    )
    assert not not_valid(
        np.ones((200, 2)), check_nan=False, shape_kwargs={"min_chains": 100, "min_draws": 2}
    )
    assert not_valid(
        np.ones((10, 10)), check_nan=False, shape_kwargs={"min_chains": 2, "min_draws": 100}
    )
    assert not_valid(
        np.ones((10, 10)), check_nan=False, shape_kwargs={"min_chains": 100, "min_draws": 2}
    )


def test_logsumexp_loo(rng):
    log_lik = rng.normal(loc=-10, scale=3, size=(4, 100, 50))
    n_samples = 4 * 100
    b = 1 / n_samples

    scipy_result = logsumexp(log_lik, axis=(0, 1), b=b)
    arviz_result = _logsumexp(log_lik, axis=(0, 1), b=b)

    assert scipy_result.shape == (50,)
    assert_array_almost_equal(scipy_result, arviz_result, decimal=10)


def test_logsumexp_stability(rng):
    large_vals = rng.normal(loc=700, scale=10, size=(4, 100))
    small_vals = rng.normal(loc=-700, scale=10, size=(4, 100))

    for ary in [large_vals, small_vals]:
        scipy_result = logsumexp(ary)
        arviz_result = _logsumexp(ary)
        assert np.isfinite(scipy_result)
        assert np.isfinite(arviz_result)
        assert_array_almost_equal(scipy_result, arviz_result, decimal=10)


def test_logsumexp_b_inv_match(rng):
    ary = rng.normal(size=(10, 20))
    b_inv = 50

    arviz_b_inv = _logsumexp(ary, b_inv=b_inv, axis=0)
    scipy_b = logsumexp(ary, b=1 / b_inv, axis=0)

    assert_array_almost_equal(arviz_b_inv, scipy_b, decimal=10)


@pytest.mark.parametrize("axis", [0, 1, -1, (0, 1), (-2, -1)])
def test_logsumexp_axes(rng, axis):
    ary = rng.normal(loc=-5, scale=2, size=(4, 100, 30))

    scipy_result = logsumexp(ary, axis=axis)
    arviz_result = _logsumexp(ary, axis=axis)

    assert_array_almost_equal(scipy_result, arviz_result, decimal=10)


def test_logsumexp_inf_handling():
    assert _logsumexp(np.array([1.0, np.inf, 2.0])) == np.inf
    assert _logsumexp(np.array([-np.inf, -np.inf])) == -np.inf


def test_logsumexp_edge_b():
    ary = np.array([1.0, 2.0, 3.0])
    assert _logsumexp(ary, b=0) == -np.inf
    assert _logsumexp(ary, b_inv=0) == np.inf


@pytest.mark.parametrize(
    "shape,axis,expected_shape",
    [
        ((4, 500, 20), (0, 1), (20,)),
        ((4, 500, 1), (0, 1), (1,)),
        ((4, 500, 8, 10), (0, 1), (8, 10)),
        ((4, 500, 20), (-3, -2), (20,)),
    ],
)
def test_logsumexp_loo_shapes(rng, shape, axis, expected_shape):
    log_lik = rng.normal(loc=-5, scale=2, size=shape)
    n_samples = shape[0] * shape[1]

    scipy_result = logsumexp(log_lik, axis=axis, b=1 / n_samples)
    arviz_result = _logsumexp(log_lik, axis=axis, b=1 / n_samples)

    assert scipy_result.shape == expected_shape
    assert arviz_result.shape == expected_shape
    assert_array_almost_equal(scipy_result, arviz_result, decimal=10)

    max_log_lik = np.max(log_lik, axis=axis)
    assert np.all(scipy_result <= max_log_lik + 1)
    assert np.all(scipy_result >= -10)
    assert np.all(arviz_result <= max_log_lik + 1)
    assert np.all(arviz_result >= -10)


def test_logsumexp_loo_weights(rng):
    n_chains, n_draws, n_obs = 4, 500, 20
    log_lik = rng.normal(loc=-5, scale=2, size=(n_chains, n_draws, n_obs))

    log_weights = rng.normal(loc=-5, scale=1, size=(n_chains, n_draws, n_obs))
    log_weights = log_weights - logsumexp(log_weights, axis=(0, 1), keepdims=True)

    log_weighted = log_weights + log_lik

    scipy_elpd_i = logsumexp(log_weighted, axis=(0, 1))
    arviz_elpd_i = _logsumexp(log_weighted, axis=(0, 1))

    assert scipy_elpd_i.shape == (n_obs,)
    assert_array_almost_equal(scipy_elpd_i, arviz_elpd_i, decimal=10)

    assert np.all(np.isfinite(scipy_elpd_i))
    assert np.all(np.isfinite(arviz_elpd_i))
    weights_sum = logsumexp(log_weights, axis=(0, 1))
    assert_array_almost_equal(weights_sum, np.zeros(n_obs), decimal=10)


@pytest.mark.parametrize(
    "loc,scale",
    [
        (-0.9, 0.1),
        (-3.5, 1.5),
        (-20, 5),
        (-500, 50),
        (-0.1, 0.01),
    ],
)
def test_logsumexp_loo_values(rng, loc, scale):
    log_lik = rng.normal(loc=loc, scale=scale, size=(4, 500, 10))
    n_samples = log_lik.shape[0] * log_lik.shape[1]

    scipy_result = logsumexp(log_lik, axis=(0, 1), b=1 / n_samples)
    arviz_result = _logsumexp(log_lik, axis=(0, 1), b=1 / n_samples)

    assert np.all(np.isfinite(scipy_result))
    assert np.all(np.isfinite(arviz_result))
    assert_array_almost_equal(scipy_result, arviz_result, decimal=8)

    assert np.all(np.abs(scipy_result - loc) < scale * 5)
    assert np.all(np.abs(arviz_result - loc) < scale * 5)


def test_logsumexp_loo_b_inv(rng):
    n_chains, n_draws, n_obs = 4, 500, 20
    log_lik = rng.normal(loc=-5, scale=2, size=(n_chains, n_draws, n_obs))
    n_samples = n_chains * n_draws

    scipy_result = logsumexp(log_lik, axis=(0, 1), b=1 / n_samples)
    arviz_result = _logsumexp(log_lik, axis=(0, 1), b_inv=n_samples)

    assert_array_almost_equal(scipy_result, arviz_result, decimal=10)


@pytest.mark.parametrize(
    "n_chains,n_draws,n_obs",
    [
        (1, 1000, 10),
        (2, 500, 10),
        (8, 250, 10),
        (4, 2000, 5),
    ],
)
def test_logsumexp_loo_varying_dims(rng, n_chains, n_draws, n_obs):
    log_lik = rng.normal(loc=-5, scale=2, size=(n_chains, n_draws, n_obs))
    n_samples = n_chains * n_draws

    scipy_result = logsumexp(log_lik, axis=(0, 1), b=1 / n_samples)
    arviz_result = _logsumexp(log_lik, axis=(0, 1), b=1 / n_samples)

    assert_array_almost_equal(scipy_result, arviz_result, decimal=10)


@pytest.mark.parametrize("keepdims", [True, False])
def test_logsumexp_loo_keepdims(rng, keepdims):
    n_chains, n_draws, n_obs = 4, 500, 20
    log_lik = rng.normal(loc=-5, scale=2, size=(n_chains, n_draws, n_obs))
    n_samples = n_chains * n_draws

    scipy_result = logsumexp(log_lik, axis=(0, 1), b=1 / n_samples, keepdims=keepdims)
    arviz_result = _logsumexp(log_lik, axis=(0, 1), b=1 / n_samples, keepdims=keepdims)

    expected_shape = (1, 1, n_obs) if keepdims else (n_obs,)
    assert scipy_result.shape == expected_shape
    assert arviz_result.shape == expected_shape
    assert_array_almost_equal(scipy_result, arviz_result, decimal=10)


def test_logsumexp_loo_copy(rng):
    n_chains, n_draws, n_obs = 4, 500, 20
    log_lik = rng.normal(loc=-5, scale=2, size=(n_chains, n_draws, n_obs))
    n_samples = n_chains * n_draws

    result_copy = _logsumexp(log_lik.copy(), axis=(0, 1), b=1 / n_samples, copy=True)

    log_lik_no_copy = log_lik.copy()
    result_no_copy = _logsumexp(log_lik_no_copy, axis=(0, 1), b=1 / n_samples, copy=False)

    assert_array_almost_equal(result_copy, result_no_copy, decimal=10)

    log_lik_test = log_lik.copy()
    _ = _logsumexp(log_lik_test, axis=(0, 1), b=1 / n_samples, copy=True)
    assert_array_almost_equal(log_lik, log_lik_test)


def test_logsumexp_loo_out(rng):
    n_chains, n_draws, n_obs = 4, 500, 20
    log_lik = rng.normal(loc=-5, scale=2, size=(n_chains, n_draws, n_obs))
    n_samples = n_chains * n_draws

    out = np.empty(n_obs)
    result = _logsumexp(log_lik, axis=(0, 1), b=1 / n_samples, out=out)

    assert result is out

    scipy_result = logsumexp(log_lik, axis=(0, 1), b=1 / n_samples)
    assert_array_almost_equal(out, scipy_result, decimal=10)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_logsumexp_loo_dtype(rng, dtype):
    n_chains, n_draws, n_obs = 4, 500, 10
    log_lik = rng.normal(loc=-5, scale=2, size=(n_chains, n_draws, n_obs)).astype(dtype)
    n_samples = n_chains * n_draws

    result = _logsumexp(log_lik, axis=(0, 1), b=1 / n_samples)

    assert result.dtype == dtype


def test_logsumexp_loo_constant():
    n_chains, n_draws, n_obs = 4, 500, 10

    constant_value = -5.0
    log_lik = np.full((n_chains, n_draws, n_obs), constant_value)
    n_samples = n_chains * n_draws

    scipy_result = logsumexp(log_lik, axis=(0, 1), b=1 / n_samples)
    arviz_result = _logsumexp(log_lik, axis=(0, 1), b=1 / n_samples)

    expected = np.full(n_obs, constant_value)

    assert_array_almost_equal(scipy_result, expected, decimal=10)
    assert_array_almost_equal(arviz_result, expected, decimal=10)
    assert_array_almost_equal(scipy_result, arviz_result, decimal=10)


def test_logsumexp_loo_known_values():
    log_lik = np.array([[[0.0, -1.0], [-2.0, -3.0]]])
    n_samples = 2

    scipy_result = logsumexp(log_lik, axis=(0, 1), b=1 / n_samples)
    arviz_result = _logsumexp(log_lik, axis=(0, 1), b=1 / n_samples)

    expected_0 = np.log((np.exp(0.0) + np.exp(-2.0)) / 2)
    expected_1 = np.log((np.exp(-1.0) + np.exp(-3.0)) / 2)
    expected = np.array([expected_0, expected_1])

    assert_array_almost_equal(scipy_result, expected, decimal=10)
    assert_array_almost_equal(arviz_result, expected, decimal=10)
    assert_array_almost_equal(scipy_result, arviz_result, decimal=10)


def test_logsumexp_loo_bounds(rng):
    n_chains, n_draws, n_obs = 4, 500, 20
    log_lik = rng.normal(loc=-5, scale=2, size=(n_chains, n_draws, n_obs))
    n_samples = n_chains * n_draws

    result = _logsumexp(log_lik, axis=(0, 1), b=1 / n_samples)

    max_vals = np.max(log_lik, axis=(0, 1))
    mean_vals = np.mean(log_lik, axis=(0, 1))

    assert np.all(result <= max_vals)
    assert np.all(result >= mean_vals - 5)
