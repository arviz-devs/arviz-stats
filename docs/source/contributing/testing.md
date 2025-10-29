# Testing arviz-stats

## How to run the test suite
The recommended way to run the test suite is to do it via `tox`.
Tox manages the environment, its env variables and the command to run
to allow testing the library with different combinations of optional dependencies
from the same development env.

To run the full test suite run:

```bash
tox -e full
```

:::{tip}
Whenever you need dependencies to be updated for tests to pass add the `r` flag to indicate
you want the virtual environment the job runs from to be recreated:

```bash
tox -e full -r
```
:::

At the time of writing, this will run the test suite after installing `arviz-stats[xarray,numba]`
and with env vars to ensure no test is skipped due to missing dependencies.
If a dependency is missing the respective test will fail instead.

To run the parts of the test suite that can be run with only some of the optional
dependencies installed there are the following two commands:

```bash
tox -e minimal  # after installing arviz-stats
tox -e xarray   # after installing arviz-stats[xarray]
```

In addition, there is also a `nightlies` job that will install available packages
from [the scientific python nightly index](https://anaconda.org/scientific-python-nightly-wheels):

```bash
tox -e nightlies
```

:::{note}
Installing this env will generally take noticeably longer than the envs for the other jobs.
This is because the cache pip builds when you install the same package and version will
generally not work and some nightlies might also require installing from source instead
of using a wheel. Moreover, this env is also much more prone to needing the `-r` flag.
:::

## How to write tests

Use `importorskip` helper function from
[`tests/helpers.py`](https://github.com/arviz-devs/arviz-stats/blob/main/tests/helpers.py)
for any import outside of the Python standard library plus NumPy and SciPy. For example:

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

To take this into account, tests use the `importorskip` helper function from
[`tests/helpers.py`](https://github.com/arviz-devs/arviz-stats/blob/main/tests/helpers.py)
when importing optional dependencies like xarray, arviz-base or pandas so that tests are skipped
if a dependency is not available.
In addition, the env variable `ARVIZ_REQUIRE_ALL_DEPS` can be set to disable this behaviour
and ensure uninstalled dependencies raise an error.

When using `tox -e full` all optional dependencies are installed,
and `ARVIZ_REQUIRE_ALL_DEPS` is set to ensure all tests in the test suite run.
However, the other jobs only install some optional dependencies and don't set the env variable
which ensures that the smaller installs are also viable and work as expected.

Consequently, when writing tests we need to take this into account and separate the
testing of the array functions and their results from the xarray interface/api built
on top of that.

At the time of writing, we run the `full`, `xarray` and `minimal` jobs for all python versions
and the `nightlies` job for one python version.
Which tests are executed is defined by the combination of
[`tox.ini`](https://github.com/arviz-devs/arviz-stats/blob/main/tox.ini) and
[`.github/workflows/test.yml`](https://github.com/arviz-devs/arviz-stats/blob/main/.github/workflows/test.yml).
