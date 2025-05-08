# Testing arviz-stats

## How to run the test suite
The recommended way to run the test suite is to do it via `tox`.
Tox manages the environment, its env variables and the command to run
to allow testing the library with different combinations of optional dependencies
from the same development env.

To run the full test suite run:

```bash
tox -e py312  # or py313, should match your python version
```

To run only the parts of the test suite that use modules with NumPy and SciPy
as dependencies run:

```bash
tox -e minimal  # should work for any python version
```

## How to write tests

Use `importorskip` helper function from `tests/helpers` for any import outside of
the Python standard library plus NumPy and SciPy. For example:

```python
import copy
import re

import numpy as np

from .helpers import importorskip

xr = importorskip("xarray")

#... in the code use xr.Dataset, xr.DataArray as usual
```

As `importorskip` will skip all tests in that file, tests should be divided into
files with tests of the `array_stats` interface always being on their own file
with no optional dependencies import and the rest in a second (or more) files.

## About arviz-stats testing

The recommended install is via `arviz-stats[xarray]` which provides functionality similar to
the functions that used to be within `arviz/stats` module. However, it is also possible to install
arviz-stats without the xarray or arviz-base dependencies in which case a minimal version of the
library that depends only on NumPy and SciPy is installed.

To take this into account, tests use the `importorskip` helper function from `tests/helpers.py`
when importing optional dependencies like xarray, arviz-base or pandas so that tests are skipped
if a dependency is not available.
In addition, the env variable `ARVIZ_REQUIRE_ALL_DEPS` can be set to disable this behaviour
and ensure uninstalled dependencies raise an error.

When using `tox -e pyXXX` all optional dependencies are installed,
and `ARVIZ_REQUIRE_ALL_DEPS` is set to ensure all tests in the test suite run.
However, `tox -e minimal` only installs `arviz-stats` and doesn't set the env variable
which ensures that the minimal install is viable and works as expected.

Consequently, when writing tests we need to take this into account and separate the
testing of the array functions and their results from the xarray interface/api built
on top of that.

At the time of writing, on GitHub Actions we run the full test suite for all python versions
and the minimal test suite for one python version.
Which tests are executed is defined by the combination of `tox.ini` and `.github/workflows/test.yml`.
