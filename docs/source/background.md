# Architecture
Currently `arviz_stats` has some top level general functionality, and then submodules
that take care of the actual computations. Submodules can be completely independent or
build on top of one another.

## Top level functionality
This includes the accessors for dataarray, Dataset and Datatree objects,
the dispatcher mechanism and some general dataclasses and utilities like `ELPDData`.

## Computation submodules
Computation submodules are structured into two main classes: an array facing class and a dataarray facing class. All submodules should in principle have both available.

The array facing class takes array_like inputs, and aims to have an API similar to NumPy/SciPy.
It can be used independently of the dataarray class (to the point of not needing to have
`arviz_base` nor `xarray` installed) but it is (consequently) lower level interface.
There are more required arguments, no `rcParams` integration... 

To make integration with the dataarray facing class easier, all functions should take
`axes` (or equivalent arguments) which should allow either integers or sequences of axes
for the functions to work over.
It is also imperative that whenever new axes are added to an array,
these are added as ending dimensions.

The dataarray facing class, builds on top of the array facing one, and takes
{class}`~xarray.DataArray` inputs aiming to have a more xarray-like API.
As the array facing class API is defined and should be common between submodules,
that means that this class can very often be limited to an instance of the base array facing class.


### Base (aka numpy+scipy)
This is the core backend which should have most functionality available and that defines
the general API for both array and dataarray facing classes.

### Numba
The numba submodule builds on top of the base submodule, using numba to both
accelerate computations and generate better behaved ufuncs, ensuring compatibility
with Dask for example.
