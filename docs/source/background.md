(architecture)=
# Architecture
Currently `arviz_stats` has some top level general functionality, and then submodules
that take care of the actual computations. Submodules can be completely independent or
build on top of one another.

![Architecture diagram of ArviZ stats](_static/architecture.svg)

## Top level functionality
This includes the top level functions that accept multiple types as input,
the accessors for dataarray, Dataset and Datatree objects,
the dispatcher mechanism and some general dataclasses and utilities like `ELPDData`.

## Computation submodules
Computation submodules are structured into two main classes: an array facing class and a dataarray facing class.
All submodules should have both available, however, if two modules are somewhat similar,
one can limit itself to inherit a class from the other without any dedicated implementation.

### The `array_stats` class instance
The array facing class takes array_like inputs, and aims to have an API similar to NumPy/SciPy.
It can be used independently of the dataarray class (to the point of not needing to have
`arviz_base` nor `xarray` installed) but it is (consequently) lower level interface.
There are more required arguments, no `rcParams` integration...

Within this class, functions are generally defined as "atomic" methods which take the lowest
dimension array possible and the proper array functions, also called ufuncs, which take
arbitrary dimensionality arrays and perform batched computation as needed.
This conversion from atomic functions to array functions happens through
utility functions implemented within the same module.

To make integration with the dataarray facing class easier, all array functions should take
`axes` (or equivalent arguments such as `chain_axis` and `draw_axis`) which should allow
either integers or sequences of them for the functions to work over,
batching over the rest of the axes.
It is also imperative that whenever new axes are added to an array,
these are added as ending dimensions; otherwise interfacing with xarray via
{func}`xarray.apply_ufunc` won't behave correctly.

### The `dataarray_stats` class instance

The dataarray facing class, builds on top of the array facing one, and takes
{class}`~xarray.DataArray` inputs aiming to have a more xarray-like and arvizian API.
Among other things, this means that the order of the dimensions shouldn't matter,
only their names, it should use defaults defined in {class}`arviz_base.rcParams`.
As the array facing class API is defined and should be common between submodules,
that means that this class can very often be limited to an instance of the base array facing class.

#### On `dim`, `dims` and `sample_dims`
At one point in time, `dim` and `dims` coexisted as equivalent arguments,
some functions had one, other the other but their behaviour was exactly the same.
Then the xarray developers decided to enforce consistency and use `dim` everywhere.
We decided to have `arviz-stats` follow that convention.

`sample_dims` is not another name for the same argument but a different concept altogether.
Both in xarray and in ArviZ `sample_dims` are expected to be present in all variables
of the input. Let's see an example.

Suppose we have a {class}`xarray.Dataset` `ds` with multiple variables:

* `mu (chain: 4, draw: 100)`
* `theta (chain: 4, draw: 100, school: 8)`

It is perfectly valid to do `ds.mean(dim=["chain", "school"])` even though "school" is not a dimension
in `mu`. In the output both `mu` and `theta` will have only the "draw" dimension.
That is, the operation applied to each variable was different, we reduced a single dimension,
"chain", in `mu` and two dimensions, "chain" and "school" in `theta`. A similar thing
would happen with `ds.azstats.kde(dim=["chain", "school"])`, we would compute the KDE over
the "chain" dimension only for `mu` and over the stacked "chain"+"school" dimension for `theta`.

On the other hand, `ds.to_stacked_array(sample_dims=["chain", "school"])` or
`ds.azstats.ess(sample_dims=["chain", "school"])` is **not** valid. `sample_dims` must be present
in all variables of the input, but `mu` doesn't have a the "school" dimension.
With `dim` we can make many combinations of "chain", "draw" and "school", but
with `sample_dims` there are only 3 valid options: `"chain"`, `"draw"`, or `["chain", "draw"]`.

As a final note, keep in mind that while there are more valid combinations of `dim`
than there are for `sample_dims` that doesn't mean that any combination of "chain", "draw"
and "school" will be valid as `dim`. This will depend on the function.
There are functions that need at least 1 dimension to operate over.
The function to compute the KDE for example needs >=1 dimensions to reduce
whereas the mean also works on 0d arrays.

Consequently, `dim="school"` would be valid
for `.mean` but not for `.kde` as that second case would imply computing the KDE
over nothing for `mu` which is not supported. There are still many more valid cases
than `sample_dims` though: `["chain", "draw", "school"]` or `["draw", "school"]`
would not be valid as `sample_dims` but are valid `dim` values for `.kde`.

### Specific implementations
#### Base (aka numpy+scipy)
This is the core backend which should have most functionality available and that defines
the general API for both array and dataarray facing classes.

#### Numba
The numba submodule builds on top of the base submodule, using numba to both
accelerate computations and generate better behaved ufuncs, ensuring compatibility
with Dask for example.

:::{note}
In a large percentage of cases, functions from `arviz-stats` are used to compute plot elements.
Therefore, while reimplementing the most expensive operations can speed things up,
be sure to profile both stats and plotting functions to make sure it will actually provide
a noticeable speed-up before dedicating too much time to it.
:::
