# API reference

## Accessors
Currently, using accessors is the recommended way to call functions from `arviz_stats`.

### Dataset accessors

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_attribute.rst

   xarray.Dataset.azstats.ds

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

   xarray.Dataset.azstats.filter_vars
   xarray.Dataset.azstats.eti
   xarray.Dataset.azstats.hdi
   xarray.Dataset.azstats.ess
   xarray.Dataset.azstats.rhat
   xarray.Dataset.azstats.mcse
   xarray.Dataset.azstats.kde
   xarray.Dataset.azstats.histogram
   xarray.Dataset.azstats.ecdf

```



## DataArray facing functions

### Base submodule

```{eval-rst}
.. autosummary::
   :toctree: generated/

   arviz_stats.base.dataarray_stats.eti
   arviz_stats.base.dataarray_stats.hdi
   arviz_stats.base.dataarray_stats.ess
   arviz_stats.base.dataarray_stats.rhat
   arviz_stats.base.dataarray_stats.mcse
   arviz_stats.base.dataarray_stats.histogram
   arviz_stats.base.dataarray_stats.kde
```

### Numba submodule
The numba accelerated computations are available as the same methods
but of the `arviz_stats.numba.dataarray_stats` class.
Both their API and implementation is the same as for the base module,
the only difference being that one calls `arviz_stats.base.array_stats`
for array facing functions whereas the other one calls `arviz_stats.numba.array_stats`.

Implementation differences are thus documented below, at the array facing classes.


## Array facing functions

### Base submodule

```{eval-rst}
.. autosummary::
   :toctree: generated/

   arviz_stats.base.array_stats.eti
   arviz_stats.base.array_stats.hdi
   arviz_stats.base.array_stats.ess
   arviz_stats.base.array_stats.rhat
   arviz_stats.base.array_stats.mcse
   arviz_stats.base.array_stats.get_bins
   arviz_stats.base.array_stats.histogram
   arviz_stats.base.array_stats.kde
```

### Numba submodule
Some functions are accelerated internally without changes to the public API,
others are purely inherited from the base backend, and a last group is partially
or completely reimplemented. This last group is documented here:

```{eval-rst}
.. autosummary::
   :toctree: generated/

   arviz_stats.numba.array_stats.quantile
   arviz_stats.numba.array_stats.histogram
   arviz_stats.numba.array_stats.kde
```
