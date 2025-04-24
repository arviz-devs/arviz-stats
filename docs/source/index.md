# arviz-stats
ArviZ statistics and diagnostics functions.

## Installation

It currently can only be installed with pip:

::::{tab-set}
:::{tab-item} PyPI
:sync: stable

```bash
pip install "arviz-stats[xarray]"
```
:::
:::{tab-item} GitHub
:sync: dev

```bash
pip install "arviz-stats[xarray] @ git+https://github.com/arviz-devs/arviz-stats"
```
:::
::::


Note that it is also possible to install `arviz-stats` without the `[xarray]`.
Doing that will install a minimal package, which only depends on numpy and scipy.
Consequently, the functions that take arrays as inputs will be available,
but many features won't be available. This is only recommended for libraries
that want to compute diagnostics and statistical summaries but don't want
to depend on xarray.

```{toctree}
:caption: Reference
:hidden:

api/index
```

```{toctree}
:hidden:
:caption: Tutorials

ArviZ in Context <https://arviz-devs.github.io/EABM/>
```

```{toctree}
:caption: Background
:hidden:

background
```

```{toctree}
:caption: About
:hidden:

BlueSky <https://bsky.app/profile/arviz.bsky.social>
Mastodon <https://bayes.club/@ArviZ>
GitHub repository <https://github.com/arviz-devs/arviz-stats>
```
