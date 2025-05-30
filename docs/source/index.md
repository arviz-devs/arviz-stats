# ArviZ-stats


Welcome to the ArviZ-stats documentation! This library focuses on statistical functions and diagnostics for exploratory analysis of Bayesian models. It is one of the 3 components of the ArviZ library, the other two being:

* [arviz-base](https://arviz-base.readthedocs.io/en/latest/) data related functionality, including converters from different PPLs.
* [arviz-plots](https://arviz-plots.readthedocs.io/en/latest/) for visual checks built on top of arviz-stats and arviz-base.

We recommend most users install and use all three ArviZ components together through the main ArviZ package, as they're designed to work seamlessly as one toolkit. Advanced users may choose to install components individually for finer control over dependencies.

Note: All statistical functions - whether accessed through the full ArviZ package or directly via ArviZ-stats - are documented here.


## Overview

In Modern Bayesian statistics models are usually build and solve using probabilistic programming languages (PPLs) such as PyMC, Stan, NumPyro, etc. These languages allow users to specify models in a high-level language and perform inference using state-of-the-art algorithms like Markov Chain Monte Carlo (MCMC) or Variational Inference (VI). As a result we usually get a posterior distribution, in the form of samples. The posterior distribution has a central role in Bayesian statistics, but other distributions like the posterior and prior predictive distribution are also of interest. And other quantities may be relevant too.

The correct visualization, analysis, and interpretation of these computed data is key to properly answer the questions that motivate our analysis.

When working with Bayesian models there are a series of related tasks that need to be addressed besides inference itself:

* Diagnoses of the quality of the inference

* Model criticism, including evaluations of both model assumptions and model predictions

* Comparison of models, including model selection or model averaging

* Preparation of the results for a particular audience.

We call these tasks exploratory analysis of Bayesian models (EABM). Successfully performing such tasks are central to the iterative and interactive modelling process (See Bayesian Workflow). In the words of Persi Diaconis.

> Exploratory data analysis seeks to reveal structure, or simple descriptions in data. We look at numbers or graphs and try to find patterns. We pursue leads suggested by background information, imagination, patterns perceived, and experience with other data analyses.

The goal of ArviZ is to provide a unified interface for performing exploratory analysis of Bayesian models in Python, regardless of the PPL used to perform inference. This allows users to focus on the analysis and interpretation of the results, rather than on the details of the implementation.


(installation)=
## Installation

For instructions on how to install the full ArviZ package (including `arviz-base`, `arviz-stats` and `arviz-plots`), please refer to the [installation guide](https://arviz-devs.github.io/arviz/stable/getting_started/Installation.html).


However, if you are only interested in the statistical summaries, diagnostics, and metrics provided by ArviZ-stats, please follow the instructions below:


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
but many features won't be available.

This is mostly aimed at advanced users or libraries that want to use ArviZ
to compute diagnostics and statistical summaries but don't want to depend on xarray.


```{toctree}
:hidden:
:caption: User guide

overview
array_stats_only
ArviZ in Context <https://arviz-devs.github.io/EABM/>
```

```{toctree}
:caption: Reference
:hidden:

api/index
background
```

```{toctree}
:hidden:
:caption: Tutorials

ArviZ in Context <https://arviz-devs.github.io/EABM/>
```


```{toctree}
:caption: Contributing
:hidden:

contributing/testing.md
```

```{toctree}
:caption: About
:hidden:

BlueSky <https://bsky.app/profile/arviz.bsky.social>
Mastodon <https://bayes.club/@ArviZ>
GitHub repository <https://github.com/arviz-devs/arviz-stats>
```
