# Bugs found in `src/arviz_stats/base/diagnostics.py`

## 1. `_loo_pit` doesn't normalize log weights for the `pareto_pit` path (line 546)

The standard PIT path normalizes weights internally (lines 548–549), but the
`pareto_pit=True` path passes raw `log_weights_2d` straight to `_pareto_pit_vec`,
which *assumes normalized* log weights (`raw_pit = np.exp(logsumexp(lw[sel_below]))`).

Verified numerically with uniform (unnormalized) weights (`log_weights=0`):

```
pareto-PIT, UNNORMALIZED weights: mean=0.8690, KS p=0.0000   <- wrong
pareto-PIT, NORMALIZED weights:   mean=0.4930, KS p=0.5471   <- correct
standard PIT, UNNORMALIZED:       mean=0.4928, KS p=0.5471   <- fine
```

Currently masked because the public API passes PSIS-normalized weights, but the
two branches of the same function have inconsistent contracts.

**Fix:** normalize `log_weights_2d` before line 546 (or inside `_pareto_pit_vec`).

## 2. `_rhat_nested` rejects valid non-contiguous superchain ids (lines 1474–1478)

```python
superchain_counts = np.bincount(superchain_ids)
nchains_per_superchain = np.max(superchain_counts)
if nchains_per_superchain != np.min(superchain_counts):  # min includes 0s for missing labels
    raise ValueError(...)
```

`np.bincount` produces zero counts for unoccupied labels, so equal-sized
superchains like `[1,1,3,3]` or `[5,5,9,9]` falsely raise `ValueError`. Verified.
The public docstring doesn't require contiguous 0-based ids.

**Fix:** use `_, counts = np.unique(superchain_ids, return_counts=True)` instead
of `np.bincount`.

## 3. `_pareto_pit_single` crashes with `rng=None` when ties exist (line 1012)

`rng.uniform(raw_pit, pit_upper)` raises `AttributeError: 'NoneType' object has
no attribute 'uniform'` when `rng is None`. The `_pareto_pit_vec` docstring
promises "If None, midpoint is used", and `_loo_pit`'s standard path implements
that midpoint fallback (lines 554–558) — the pareto path doesn't.

Verified: discrete/rounded data + default `rng=None` -> crash (also reproducible
via `_loo_pit(..., pareto_pit=True, rng=None)`).

**Fix:** mirror lines 554–558, e.g. use the midpoint `0.5 * (raw_pit + pit_upper)`
when `rng is None`.

## 4. `_bfmi` mixes `ary` and `ary_mat` (line 381)

```python
ary_mat = np.atleast_2d(ary)
num = np.square(np.diff(ary_mat, axis=1)).mean(axis=1)
den = np.var(ary, axis=1, ddof=1)   # <- should be ary_mat
```

1D input -> `AxisError: axis 1 is out of bounds`. Verified. Latent today because
the `make_ufunc` wrapper always passes 2D slices, but the `atleast_2d` call shows
1D was meant to be supported.

**Fix:** `den = np.var(ary_mat, axis=1, ddof=1)`.

## 5. `_ess_tail` fails when `prob` is a numpy array (line 202)

```python
if not isinstance(prob, Sequence):
    prob = sorted((prob, 1 - prob))
```

`isinstance(prob, Sequence)` is `False` for `np.ndarray`, so a 2-element numpy
`prob` falls into `sorted((prob, 1 - prob))` which raises an ambiguous
truth-value error. Tuples and lists work fine.

**Fix:** accept `np.ndarray` too, e.g. `isinstance(prob, (Sequence, np.ndarray))`.
