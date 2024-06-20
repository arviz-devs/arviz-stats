# arviz-stats
ArviZ statistics and diagnostics functions.

## Installation

It currenly can only be installed with pip and from GitHub:

```bash
pip install "arviz-stats[xarray] @ git+https://github.com/arviz-devs/arviz-stats"
```

Note that it is also possible to install `arviz-stats` without the `[xarray]`.
Doing that will install a minimal package, which only depends on numpy and scipy.
Consequently, the functions that take arrays as inputs will be available,
but many features won't be available. This is only recommended for libraries
to depend on so they can use diagnostics and statistical summaries but don't want
to depend on xarray.

```{toctree}
:hidden:

api/index
```

```{toctree}
:caption: About
:hidden:

Twitter <https://twitter.com/arviz_devs>
Mastodon <https://bayes.club/@ArviZ>
GitHub repository <https://github.com/arviz-devs/arviz-stats>
```
