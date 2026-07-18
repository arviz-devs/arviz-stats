"""Helper functions for PSIS-LOO-CV sub-sampling."""

import numpy as np
from xarray import DataArray

from arviz_stats.loo.loo_moment_match_helper import _get_batched_func, _get_observed_rv

__all__ = ["ll_from_pymc"]


def ll_from_pymc(
    idata,
    *,
    model=None,
    var_name=None,
):
    """Build a log-likelihood function for :func:`arviz_stats.loo_subsample` from a PyMC model.

    :func:`arviz_stats.loo_subsample` and :func:`arviz_stats.update_subsample` accept a
    ``log_lik_fn`` computing the pointwise log likelihood of the observed variable, which
    is required for ``method="plpd"`` and optional for ``method="lpd"``. This helper
    compiles that function directly from the PyMC model, so it does not have to be
    constructed manually.

    Parameters
    ----------
    idata : DataTree
        InferenceData with ``posterior`` and ``observed_data`` groups. The posterior
        group must contain the free random variables of the model.
    model : Model
        The PyMC model that produced the ``idata``.
    var_name : str, optional
        Name of the observed variable whose log-likelihood is used for LOO. Can be omitted
        if the model has exactly one observed random variable.

    Returns
    -------
    log_lik_fn : callable
        ``log_lik_fn(observed, data) -> DataArray``. When ``data.posterior`` contains the
        ``chain`` and ``draw`` dimensions, the result has dims ``(chain, draw, *obs_dims)``
        holding the per-draw log likelihood of every observation, as required by
        ``method="lpd"``. When ``data.posterior`` holds a point estimate without sample
        dimensions, the result has the observation dimensions only, as required by
        ``method="plpd"``.

    See Also
    --------
    loo_subsample : PSIS-LOO-CV with sub-sampling.
    update_subsample : Update a previously computed sub-sampled PSIS-LOO-CV.
    """
    from pymc.model import modelcontext
    from pymc.model.transform.conditioning import remove_value_transforms

    try:
        model = modelcontext(model)
    except TypeError as err:
        raise ValueError("A PyMC model is required to build the log likelihood function.") from err

    if model.discrete_value_vars:
        raise NotImplementedError(
            "Building a log likelihood function is not supported for models with discrete "
            "random variables. "
            f"Found discrete value variables: {[v.name for v in model.discrete_value_vars]}."
        )

    _, var_name = _get_observed_rv(model, idata, var_name)

    # The posterior group stores draws in the constrained space, so compile the
    # likelihood on the transform-removed model whose value variables live there too.
    umodel = remove_value_transforms(model)
    obs_rv = umodel[var_name]
    free_rv_names = [rv.name for rv in umodel.free_RVs]
    value_vars = [umodel.rvs_to_values[rv] for rv in umodel.free_RVs]
    initial_point = dict(umodel.initial_point())
    var_shapes = [initial_point[v.name].shape for v in value_vars]
    named_dims = dict(umodel.named_vars_to_dims)

    lik_func = _get_batched_func(
        inputs=value_vars,
        outputs=list(umodel.logp(vars=[obs_rv], sum=False)),
        initial_point=initial_point,
    )

    if hasattr(idata, "observed_data") and var_name in idata.observed_data:
        obs_ref = idata.observed_data[var_name]
    else:
        obs_ref = None

    def log_lik_fn(observed, data):
        posterior = getattr(data.posterior, "dataset", data.posterior)

        missing = [name for name in free_rv_names if name not in posterior.data_vars]
        if missing:
            raise KeyError(
                f"Posterior draws for free random variables {missing} were not found in "
                f"`data.posterior`. Available variables: {list(posterior.data_vars)}."
            )

        has_sample_dims = all(dim in posterior.dims for dim in ("chain", "draw"))

        sample_dims = ("chain", "draw") if has_sample_dims else ()
        blocks = []
        for name, shape in zip(free_rv_names, var_shapes):
            alen = int(np.prod(shape, dtype=int)) if shape else 1
            values = posterior[name]
            rv_dims = named_dims.get(name)
            if rv_dims is not None and all(
                isinstance(dim, str) and dim in values.dims for dim in rv_dims
            ):
                values = values.transpose(*sample_dims, *rv_dims)
            elif has_sample_dims:
                values = values.transpose("chain", "draw", ...)
            blocks.append(np.asarray(values.values, dtype="float64").reshape(-1, alen))
        out = np.asarray(lik_func(np.concatenate(blocks, axis=1))[0])

        target = obs_ref if obs_ref is not None else observed
        if out.shape[1:] != target.shape:
            raise ValueError(
                f"The pointwise log likelihood of `{var_name}` has shape {out.shape[1:]}, "
                f"which does not match the observed data shape {target.shape}. Models whose "
                "likelihood reduces over observation dimensions are not supported."
            )

        coords = {dim: target[dim] for dim in target.dims if dim in target.coords}
        if has_sample_dims:
            vals = out.reshape(posterior.sizes["chain"], posterior.sizes["draw"], *target.shape)
            if "chain" in posterior.coords:
                coords["chain"] = posterior["chain"]
            if "draw" in posterior.coords:
                coords["draw"] = posterior["draw"]
            return DataArray(vals, dims=("chain", "draw", *target.dims), coords=coords)
        return DataArray(out.reshape(target.shape), dims=target.dims, coords=coords)

    return log_lik_fn
