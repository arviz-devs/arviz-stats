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
