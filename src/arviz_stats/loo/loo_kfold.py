"""K-fold cross-validation for model assessment."""

import numpy as np
import xarray as xr
from arviz_base import rcParams

from arviz_stats.loo.helper_loo_kfold import (
    _combine_fold_elpds,
    _compute_kfold_results,
    _prepare_kfold_inputs,
)
from arviz_stats.utils import ELPDData


def loo_kfold(
    data,
    pointwise=None,
    var_name=None,
    wrapper=None,
    k=10,
    folds=None,
    stratify_by=None,
    group_by=None,
    save_fits=False,
):
    """Perform exact K-fold cross-validation.

    K-fold cross-validation evaluates model predictive accuracy by partitioning the data
    into K complementary subsets (folds), then iteratively refitting the model K times,
    each time holding out one fold as a test set and training on the remaining K-1 folds.

    This method provides an unbiased estimate of model performance by ensuring each
    observation is used exactly once for testing. Unlike PSIS-LOO-CV (Pareto-smoothed
    importance sampling leave-one-out cross-validation), which approximates cross-validation
    efficiently, K-fold requires actual model refitting but yields exact results.

    Parameters
    ----------
    data : DataTree or InferenceData
        Input data containing the posterior and log_likelihood groups from the full model fit.
    pointwise : bool, optional
        If True, return pointwise estimates. Defaults to ``rcParams["stats.ic_pointwise"]``.
    var_name : str, optional
        The name of the variable in log_likelihood group storing the pointwise log
        likelihood data to use for computation.
    wrapper : SamplingWrapper
        An instance of SamplingWrapper class handling model refitting. The wrapper must
        implement the following methods: sel_observations, sample, get_inference_data,
        and log_likelihood__i.
    k : int, default=10
        The number of folds for cross-validation. The data will be partitioned into k subsets
        of equal (or approximately equal) size.
    folds : array-like, optional
        An optional integer array with one element per observation in the data. Each element
        should be an integer in 1:k indicating which fold the observation belongs to.
        If not provided, data will be randomly partitioned into k folds.
    stratify_by : array-like, optional
        A categorical variable to use for stratified K-fold splitting. When provided,
        the folds will be created to preserve the relative frequencies of the categories.
        Cannot be used together with `folds` or `group_by`.
    group_by : array-like, optional
        A grouping variable to use for grouped K-fold splitting. When provided,
        all observations from the same group will be kept together in the same fold.
        Cannot be used together with `folds` or `stratify_by`.
    save_fits : bool, default=False
        If True, store the fitted models and fold indices in the returned object.

    Returns
    -------
    ELPDData
        Object with the following attributes:

        - **elpd**: expected log pointwise predictive density
        - **se**: standard error of the elpd
        - **p**: effective number of parameters
        - **n_samples**: number of samples per fold
        - **n_data_points**: number of data points
        - **warning**: True if any issues occurred during fitting
        - **elpd_i**: pointwise predictive accuracy (if ``pointwise=True``)
        - **pareto_k**: None (not applicable for k-fold)
        - **scale**: "log"

        Additional attributes when ``save_fits=True``:

        - **fold_fits**: Dictionary containing fitted models for each fold
        - **fold_indices**: Dictionary containing test indices for each fold

    Notes
    -----
    When K equals the number of observations, this becomes exact leave-one-out
    cross-validation. Note that :func:`arviz_stats.loo` provides a much more efficient
    approximation for that case and is recommended for large datasets.

    See Also
    --------
    loo : Pareto-smoothed importance sampling LOO-CV
    SamplingWrapper : Base class for implementing sampling wrappers

    References
    ----------

    .. [1] Vehtari et al. *Practical Bayesian model evaluation using leave-one-out cross-validation
        and WAIC*. Statistics and Computing. 27(5) (2017) https://doi.org/10.1007/s11222-016-9696-4
        arXiv preprint https://arxiv.org/abs/1507.04544.

    .. [2] Vehtari et al. *Pareto Smoothed Importance Sampling*.
        Journal of Machine Learning Research, 25(72) (2024) https://jmlr.org/papers/v25/19-556.html
        arXiv preprint https://arxiv.org/abs/1507.02646
    """
    pointwise = rcParams["stats.ic_pointwise"] if pointwise is None else pointwise

    kfold_inputs = _prepare_kfold_inputs(data, var_name, wrapper, k, folds, stratify_by, group_by)
    kfold_results = _compute_kfold_results(kfold_inputs, wrapper, save_fits)

    if pointwise:
        pointwise_data = np.column_stack([kfold_results.elpds, kfold_results.ps])
        obs_dim = kfold_inputs.obs_dims[-1] if kfold_inputs.obs_dims else "observation"

        pointwise_df = xr.DataArray(
            pointwise_data,
            dims=[obs_dim, "metric"],
            coords={
                obs_dim: kfold_inputs.log_likelihood.coords[obs_dim],
                "metric": ["elpd_loo_kfold", "p_loo_kfold"],
            },
        )
    else:
        pointwise_df = None

    combined_results = _combine_fold_elpds([kfold_results.elpds], kfold_inputs.n_data_points)
    elpd_sum = combined_results["elpd_kfold"]
    se_elpd = combined_results["se_elpd_kfold"]
    p_sum = np.sum(kfold_results.ps)

    elpd_data = ELPDData(
        kind="loo_kfold",
        elpd=elpd_sum,
        se=se_elpd,
        p=p_sum,
        n_samples=kfold_inputs.n_samples,
        n_data_points=kfold_inputs.n_data_points,
        scale="log",
        warning=False,
        good_k=None,
        elpd_i=pointwise_df if pointwise else None,
        pareto_k=None,
    )

    if save_fits:
        elpd_data.fold_fits = kfold_results.fold_fits
    return elpd_data
