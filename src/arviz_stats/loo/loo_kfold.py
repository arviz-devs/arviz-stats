"""K-fold cross-validation for model assessment."""

import numpy as np
from arviz_base import rcParams

from arviz_stats.loo.helper_loo_kfold import (
    _combine_fold_elpds,
    _compute_kfold_results,
    _prepare_kfold_inputs,
)
from arviz_stats.utils import ELPDData


def loo_kfold(
    data,
    wrapper,
    pointwise=None,
    var_name=None,
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
    wrapper : SamplingWrapper
        An instance of SamplingWrapper class handling model refitting. The wrapper must
        implement the following methods: sel_observations, sample, get_inference_data,
        and log_likelihood__i.
    pointwise : bool, optional
        If True, return pointwise estimates. Defaults to ``rcParams["stats.ic_pointwise"]``.
    var_name : str, optional
        The name of the variable in log_likelihood group storing the pointwise log
        likelihood data to use for computation.
    k : int, default=10
        The number of folds for cross-validation. The data will be partitioned into k subsets
        of equal (or approximately equal) size.
    folds : array or DataArray, optional
        An optional integer array or DataArray with one element per observation in the data.
        Each element should be an integer from 1 to k indicating which fold the observation
        belongs to. For example, with k=4 and 8 observations, one possible assignment is
        [1,1,2,2,3,3,4,4] to put the first two observations in fold 1, next two in fold 2, etc.
        If not provided, data will be randomly partitioned into k folds of approximately
        equal size. DataArray inputs will be automatically flattened to 1D.
    stratify_by : array or DataArray, optional
        A categorical variable to use for stratified K-fold splitting. For example, with
        8 observations where [0,0,1,1,0,0,1,1] indicates two categories (0 and 1), the
        algorithm ensures each fold contains approximately the same 50/50 split of 0s and 1s
        as the full dataset. Cannot be used together with `folds` or `group_by`. DataArray
        inputs will be automatically flattened to 1D.
    group_by : array or DataArray, optional
        A grouping variable to use for grouped K-fold splitting. For example, with
        [1,1,2,2,3,3,4,4] representing 4 subjects with 2 observations each, all observations
        from subject 1 will be placed in the same fold, all from subject 2 in the same fold,
        etc. This ensures related observations stay together. Cannot be used together with
        `folds` or `stratify_by`. DataArray inputs will be automatically flattened to 1D.
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
        - **p_kfold_i**: pointwise effective number of parameters (if ``pointwise=True``)
        - **pareto_k**: None (not applicable for k-fold)
        - **scale**: "log"

        Additional attributes when ``save_fits=True``:

        - **fold_fits**: Dictionary containing fitted models for each fold
        - **fold_indices**: Dictionary containing test indices for each fold

    Notes
    -----
    When K equals the number of observations, this becomes exact leave-one-out
    cross-validation. Note that :func:`arviz_stats.loo` provides a much more efficient
    approximation for that case and is recommended.

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
        elpd_i=kfold_results.elpd_i if pointwise else None,
        pareto_k=None,
    )

    if save_fits:
        elpd_data.fold_fits = kfold_results.fold_fits

    if pointwise:
        elpd_data.p_kfold_i = kfold_results.p_kfold_i

    return elpd_data
