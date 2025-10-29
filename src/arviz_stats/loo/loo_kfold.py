"""K-fold cross-validation."""

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
        Manual fold assignments (1 to k) for each observation. For example, [1,1,2,2,3,3,4,4]
        assigns first two obs to fold 1, next two to fold 2, etc. If not provided, creates k
        random folds of equal size. Cannot be used together with `stratify_by` and `group_by`.
    stratify_by : array or DataArray, optional
        Maintains class proportions across folds. For example, [0,0,1,1,0,0,1,1] ensures each
        fold has 50% class 0 and 50% class 1. Cannot be used together with `folds` and `group_by`.
    group_by : array or DataArray, optional
        Grouping variable to keep related observations together in the same fold. For example,
        [1,1,2,2,3,3] keeps all obs from group 1 in one fold, group 2 in another, etc. Useful
        for repeated measures or clustered data. Cannot be used together with `folds` and
        `stratify_by`.
    save_fits : bool, default=False
        If True, store the fitted models and fold indices in the returned object.

    Returns
    -------
    ELPDData
        Object with the following attributes:

        - **kind**: "loo_kfold"
        - **elpd**: expected log pointwise predictive density
        - **se**: standard error of the elpd
        - **p**: effective number of parameters
        - **n_samples**: number of samples per fold
        - **n_data_points**: number of data points
        - **scale**: "log"
        - **warning**: False (not applicable for :math:`k`-fold)
        - **good_k**: None (not applicable for :math:`k`-fold)
        - **elpd_i**: :class:`~xarray.DataArray` with pointwise predictive accuracy,
          only if ``pointwise=True``
        - **pareto_k**: None (not applicable for :math:`k`-fold)
        - **n_folds**: number of folds (:math:`k`)
        - **p_kfold_i**: :class:`~xarray.DataArray` with pointwise effective number of parameters,
          only if ``pointwise=True``
        - **fold_fits**: Dictionary containing fitted models for each fold,
          only if ``save_fits=True``

    Examples
    --------
    Unlike PSIS-LOO (which approximates LOO-CV), k-fold cross-validation refits
    the model k times. So we need to tell ``loo_kfold`` how to refit the model.

    This is done by creating an instance of the ``SamplingWrapper`` class that
    implements four key methods: ``sel_observations``, ``sample``, ``get_inference_data``,
    and ``log_likelihood__i``.

    .. ipython::

        In [1]: import numpy as np
           ...: import xarray as xr
           ...: from scipy import stats
           ...: from arviz_base import load_arviz_data, from_dict
           ...: from arviz_stats import loo_kfold
           ...: from arviz_stats.loo import SamplingWrapper
           ...:
           ...: class CenteredEightWrapper(SamplingWrapper):
           ...:     def __init__(self, idata):
           ...:         super().__init__(model=None, idata_orig=idata)
           ...:         self.y_obs = idata.observed_data["obs"].values
           ...:         self.sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18])
           ...:
           ...:     def sel_observations(self, idx):
           ...:         all_idx = np.arange(len(self.y_obs))
           ...:         train_idx = np.setdiff1d(all_idx, idx)
           ...:
           ...:         train_data = {
           ...:             "y": self.y_obs[train_idx],
           ...:             "sigma": self.sigma[train_idx],
           ...:             "indices": train_idx
           ...:         }
           ...:         test_data = {
           ...:             "y": self.y_obs[idx],
           ...:             "sigma": self.sigma[idx],
           ...:             "indices": idx
           ...:         }
           ...:         return train_data, test_data
           ...:
           ...:     def sample(self, modified_observed_data):
           ...:         # (Simplified version where we normally would use the actual sampler)
           ...:         train_y = modified_observed_data["y"]
           ...:         n = 1000
           ...:         mu = np.random.normal(train_y.mean(), 5, n)
           ...:         tau = np.abs(np.random.normal(10, 2, n))
           ...:         return {"mu": mu, "tau": tau}
           ...:
           ...:     def get_inference_data(self, fitted_model):
           ...:         posterior = {
           ...:             "mu": fitted_model["mu"].reshape(1, -1),
           ...:             "tau": fitted_model["tau"].reshape(1, -1)
           ...:         }
           ...:         return from_dict({"posterior": posterior})
           ...:
           ...:     def log_likelihood__i(self, excluded_obs, idata__i):
           ...:         test_y = excluded_obs["y"]
           ...:         test_sigma = excluded_obs["sigma"]
           ...:         mu = idata__i.posterior["mu"].values.flatten()
           ...:         tau = idata__i.posterior["tau"].values.flatten()
           ...:
           ...:         var_total = tau[:, np.newaxis] ** 2 + test_sigma**2
           ...:         log_lik = stats.norm.logpdf(
           ...:             test_y, loc=mu[:, np.newaxis], scale=np.sqrt(var_total)
           ...:         )
           ...:
           ...:         dims = ["chain", "school", "draw"]
           ...:         coords = {"school": excluded_obs["indices"]}
           ...:         return xr.DataArray(
           ...:             log_lik.T[np.newaxis, :, :], dims=dims, coords=coords
           ...:         )

    Now let's run k-fold cross-validation. With k=4, we'll refit the model 4 times,
    each time leaving out 2 schools for testing:

    .. ipython::

        In [2]: data = load_arviz_data("centered_eight")
           ...: wrapper = CenteredEightWrapper(data)
           ...: kfold_results = loo_kfold(data, wrapper, k=4, pointwise=True)
           ...: kfold_results

    Sometimes we want more control over how the data is split. For instance,
    if you have imbalanced groups, stratified k-fold ensures each fold has
    a similar distribution:

    .. ipython::

        In [3]: strata = (data.observed_data["obs"] > 5).astype(int)
           ...: kfold_strat = loo_kfold(data, wrapper, k=4, stratify_by=strata)
           ...: kfold_strat

    Moreover, sometimes we want to group observations together. For instance,
    if we have repeated measurements from the same subject, we can group by subject:

    .. ipython::

        In [4]: groups = xr.DataArray([1, 1, 2, 2, 3, 3, 4, 4], dims="school")
           ...: kfold_group = loo_kfold(data, wrapper, k=4, group_by=groups)
           ...: kfold_group

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
        n_folds=k,
    )

    if save_fits:
        elpd_data.fold_fits = kfold_results.fold_fits

    if pointwise:
        elpd_data.p_kfold_i = kfold_results.p_kfold_i

    return elpd_data
