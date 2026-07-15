# File generated with docstub

from collections.abc import Generator

from _typeshed import Incomplete
from numpy.typing import ArrayLike
from xarray import Dataset

def difference_ecdf_pit(
    predictive_dist: Dataset,
    observed_dist: Dataset,
    ci_prob: float,
    coverage: Incomplete,
    n_simulations: int,
) -> None: ...
def ecdf_pit(
    vals: ArrayLike,
    ci_prob: float,
    n_simulations: int,
    n_chains: int = ...,
    rng: Generator | None = ...,
) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]: ...
def compute_ecdf(sample: Incomplete, eval_points: Incomplete) -> None: ...
def get_pointwise_confidence_band(
    prob: Incomplete, ndraws: Incomplete, eval_points: Incomplete
) -> None: ...
def simulate_confidence_bands(
    n_draws: int,
    n_chains: int,
    eval_points: ArrayLike,
    prob: float,
    n_simulations: int,
    rng: Generator,
) -> None: ...
def _build_hypergeom_lookup_table(
    population: Incomplete, draws: Incomplete, successes: Incomplete
) -> None: ...
def _scalar_lookup(
    x_val: Incomplete, population: Incomplete, draws: Incomplete, successes: Incomplete
) -> None: ...
def _hypergeometric_cdf_lookup(
    x_val: Incomplete, population: Incomplete, draws: Incomplete, successes: Incomplete
) -> None: ...
def hypergeom_cdf(
    x_values: Incomplete,
    draws: Incomplete,
    successes: Incomplete,
    population: Incomplete,
) -> None: ...
def _pit_for_hist(
    value: Incomplete,
    left_edges: Incomplete,
    right_edges: Incomplete,
    histogram: Incomplete,
    *,
    eps: Incomplete,
    rng: Incomplete,
) -> None: ...
def compute_pit_for_histogram(
    dt_group: Incomplete, hist_dt: Incomplete, sample_dims: Incomplete
) -> None: ...
def _interp_cdf(
    value: Incomplete, grid: Incomplete, cdf: Incomplete, *, eps: Incomplete, rng: Incomplete
) -> None: ...
def compute_pit_for_kde(
    dt_group: Incomplete, kde_dt: Incomplete, sample_dims: Incomplete
) -> None: ...
def _pit_f_for_qds(
    values: Incomplete,
    quantile_positions: Incomplete,
    radius: Incomplete,
    nqds: Incomplete,
    *,
    rng: Incomplete,
) -> None: ...
def compute_pit_for_qds(
    dt_group: Incomplete, qds_dt: Incomplete, sample_dims: Incomplete
) -> None: ...
