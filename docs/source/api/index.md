# API reference

:::{note}
Similarly to xarray, ArviZ aims for idempotent functions. However,
there are two important caveats. First and foremost, several functions
require combining data from multiple groups or variables, in which case the
output won't keep the type of the input data.

Moreover, ArviZ works on
data following the InferenceData schema, there are functions that can accept
PPL outputs directly, but when this happens, the first step is converting
those outputs to InferenceData. Consequently, the output won't be of the same type.
:::

As indicated in {ref}`installation`, the recommended way to install `arviz-stats`
is with `xarray` and `arviz-base` as optional dependencies.
This activates most of the features of the libraries and the bulk of this reference
guide documents them. If you only installed the minimal version of `arviz-stats`
you should jump to {ref}`array_stats_api` and read forward.

## Top level functions

### Diagnostics

```{eval-rst}
.. autosummary::
   :toctree: generated/

   arviz_stats.ess
   arviz_stats.loo_pit
   arviz_stats.mcse
   arviz_stats.psense
   arviz_stats.psense_summary
   arviz_stats.rhat
   arviz_stats.rhat_nested
```

### Statistical summaries

```{eval-rst}
.. autosummary::
   :toctree: generated/

   arviz_stats.bayes_factor
   arviz_stats.loo_expectations
   arviz_stats.loo_metrics
   arviz_stats.summary
```

### Model comparison

```{eval-rst}
.. autosummary::
   :toctree: generated/

   arviz_stats.compare
   arviz_stats.loo
   arviz_stats.loo_approximate_posterior
   arviz_stats.loo_subsample
   arviz_stats.update_subsample
```

### Other

```{eval-rst}
.. autosummary::
   :toctree: generated/

   arviz_stats.thin
```

## Accessors
In addition, many functions are also available via accessors:

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
   xarray.Dataset.azstats.compute_ranks
   xarray.Dataset.azstats.ess
   xarray.Dataset.azstats.rhat
   xarray.Dataset.azstats.rhat_nested
   xarray.Dataset.azstats.mcse
   xarray.Dataset.azstats.thin
   xarray.Dataset.azstats.kde
   xarray.Dataset.azstats.histogram
   xarray.Dataset.azstats.ecdf
   xarray.Dataset.azstats.autocorr

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
   arviz_stats.base.dataarray_stats.rhat_nested
   arviz_stats.base.dataarray_stats.mcse
   arviz_stats.base.dataarray_stats.histogram
   arviz_stats.base.dataarray_stats.kde
   arviz_stats.base.dataarray_stats.autocorr
```

### Numba submodule
The numba accelerated computations are available as the same methods
but of the `arviz_stats.numba.dataarray_stats` class.
Both their API and implementation is the same as for the base module,
the only difference being that one calls `arviz_stats.base.array_stats`
for array facing functions whereas the other one calls `arviz_stats.numba.array_stats`.

Implementation differences are thus documented below, at the array facing classes.


(array_stats_api)=
## Array facing functions
All functions and methods described after this point work when installing
`arviz-stats` without optional dependencies.

:::{warning}
Keep in mind this is not the recommended install and the main target of
such functions are developers of other libraries who want to use ArviZ
but keep the dependency list small.

The documentation is more bare bones than other functions and will often
refer you to other pages for the full argument or algorithm descriptions.
:::

### Base submodule

#### Sampling diagnostics
```{eval-rst}
.. autosummary::
   :toctree: generated/

   arviz_stats.base.array_stats.ess
   arviz_stats.base.array_stats.pareto_min_ss
   arviz_stats.base.array_stats.rhat
   arviz_stats.base.array_stats.rhat_nested
   arviz_stats.base.array_stats.mcse
```

#### Statistical summaries
```{eval-rst}
.. autosummary::
   :toctree: generated/

   arviz_stats.base.array_stats.eti
   arviz_stats.base.array_stats.hdi
   arviz_stats.base.array_stats.histogram
   arviz_stats.base.array_stats.kde
   arviz_stats.base.array_stats.quantile
```

#### Other
```{eval-rst}
.. autosummary::
   :toctree: generated/

   arviz_stats.base.array_stats.autocorr
   arviz_stats.base.array_stats.autocov
   arviz_stats.base.array_stats.compute_ranks
   arviz_stats.base.array_stats.get_bins
   arviz_stats.base.array_stats.psislw
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
