"""Helper for PSIS-LOO-CV moment matching.

function arviz_stats.loo_moment_match requires the user to supply an
array of posterior draws in the unconstrained parameter space and two
functions:

* ``log_prob_upars_func(upars)`` - log posterior density evaluated at
  unconstrained parameter draws.
* ``log_lik_i_upars_func(upars, i)`` - log likelihood of a single observation
  ``i`` evaluated at unconstrained parameter draws.
* ``upars`` - posterior draws transformed to the unconstrained space
"""

import numpy as np
from xarray import DataArray

__all__ = ["mm_from_pymc"]


def mm_from_pymc(
    idata,
    *,
    model=None,
    var_name=None,
):
    """Build the functions and posterior array needed by loo_moment_match from a PyMC model.

    Parameters
    ----------
    idata : DataTree
        InferenceData with `posterior` and `log_likelihood` groups. The posterior
        group must contain the free random variables of the model.
    model : Model
        The PyMC model that produced the `idata`.
    var_name : str, optional
        Name of the observed variable whose log-likelihood is used for LOO. Can be omitted
        if the model has exactly one observed random variable.

    Returns
    -------
    log_prob_upars_func : callable
        ``log_prob_upars_func(upars) -> DataArray`` with dims ``(chain, draw)``.
        Computes the (Jacobian-adjusted) log posterior density in the
        unconstrained parameter space.
    log_lik_i_upars_func : callable
        ``log_lik_i_upars_func(upars, i) -> DataArray`` with dims ``(chain, draw)``.
        Computes the log likelihood of a single observation ``i`` (a flat
        integer index in C-order over the observation dimensions of the
        selected observed variable).
    upars : xarray.DataArray
        Posterior draws transformed to the unconstrained parameter space, with
        dims ``(chain, draw, unconstrained_parameter)`` and coordinate values
        given by the (possibly transformed) value-variable names (with
        ``[i,j,...]`` suffixes for multi-element value variables).
    """
    from pymc.model import modelcontext

    model = modelcontext(model)
    if model is None:
        raise ValueError("A PyMC model is required to build moment matching functions.")

    _validate_model(model)
    obs_rv, var_name = _get_observed_rv(model, idata, var_name)
    initial_point = dict(model.initial_point())
    value_vars = list(model.value_vars)

    upars, n_params = _get_upars_da(idata, model, initial_point, value_vars)

    logp_func = _get_batched_func(
        inputs=value_vars,
        outputs=[model.logp(jacobian=True, sum=True)],
        initial_point=initial_point,
    )

    def log_prob_upars(upars_da):
        chain = upars_da.sizes["chain"]
        draw = upars_da.sizes["draw"]
        flat = np.asarray(upars_da.values).reshape(-1, n_params)
        vals = np.asarray(logp_func(flat)[0]).reshape(chain, draw)
        return DataArray(
            vals,
            dims=("chain", "draw"),
            coords={"chain": upars_da["chain"], "draw": upars_da["draw"]},
        )

    # log_lik_i_upars_func: elementwise log likelihood for the selected observed variable.
    elemwise = list(model.logp(vars=[obs_rv], sum=False))
    lik_func = _get_batched_func(
        inputs=value_vars,
        outputs=elemwise,
        initial_point=initial_point,
    )

    def log_lik_i_upars(upars_da, obs_i):
        chain = upars_da.sizes["chain"]
        draw = upars_da.sizes["draw"]
        flat = np.asarray(upars_da.values).reshape(-1, n_params)
        out = np.asarray(lik_func(flat)[0])
        if out.ndim == 1:
            vals = out
        else:
            idx = (slice(None), *np.unravel_index(int(obs_i), out.shape[1:]))
            vals = out[idx]
        vals = vals.reshape(chain, draw)
        return DataArray(
            vals,
            dims=("chain", "draw"),
            coords={"chain": upars_da["chain"], "draw": upars_da["draw"]},
        )

    return log_prob_upars, log_lik_i_upars, upars


def _validate_model(model):
    """Check that the model is compatible with moment matching.

    Parameters
    ----------
    model : Model
        The PyMC model to check.

    Raises
    ------
    NotImplementedError
        If the model has discrete random variables or potentials, which are not
        supported by moment matching.
    """
    if model.discrete_value_vars:
        raise NotImplementedError(
            "Moment matching is not supported for models with discrete random"
            f"Found discrete value variables: {[v.name for v in model.discrete_value_vars]}."
        )
    if model.potentials:
        raise NotImplementedError(
            "Moment matching for models with potentials is not supported"
            "Remove the potentials or supply the moment matching functions manually."
        )


def _get_observed_rv(model, idata, var_name):
    """Identify the observed random variable to use for moment matching.

    Parameters
    ----------
    model : Model
        The PyMC model to check.
    idata : DataTree
        The DataTree object with an observed_data group.
    var_name : str or None
        The name of the observed variable to use. If None, the function will
        attempt to infer the observed variable.

    Returns
    -------
    observed_rv : RandomVariable
        The observed random variable to use for moment matching.
    var_name : str
        The name of the observed variable.
    """
    observed = list(model.observed_RVs)

    if var_name is not None and var_name not in idata.observed_data:
        raise ValueError(
            f"`{var_name!r}` is not present in `idata.observed_data`. Make sure "
            "the InferenceData contains the observed variable whose log-likelihood "
            "is used for LOO."
        )

    if var_name is None:
        if len(observed) != 1:
            raise ValueError(
                "The model has multiple observed random variables "
                f"({[rv.name for rv in observed]}); please pass `var_name`."
            )
        return observed[0], observed[0].name

    matches = [rv for rv in observed if rv.name == var_name]
    return matches[0], var_name


def _get_upars_da(idata, model, initial_point, value_vars):
    """Compute unconstrained posterior draws.

    Parameters
    ----------
    idata : DataTree
        The DataTree object containing the posterior samples.
    model : Model
        The PyMC model.
    initial_point : dict
        The initial point of the model.
    value_vars : list
        The list of value variables.

    Returns
    -------
    upars : DataArray
        Shape ``(chain, draw, n_params)``.
    n_params : int
        Total number of unconstrained parameters.
    """
    import pytensor
    import pytensor.tensor as pt

    chain = idata.posterior.sizes["chain"]
    draw = idata.posterior.sizes["draw"]
    posterior_data_vars = set(idata.posterior.data_vars)

    # Build the PyTensor transform graph: for each value var, either apply the
    # forward transform or use the identity.
    inputs_for_func = []  # PyTensor input variables
    outputs_for_func = []  # PyTensor output expressions (post-transform or identity)
    posterior_names = []  # variable names to look up in idata.posterior
    unc_shapes = []  # unconstrained shapes (from initial_point)

    for v in value_vars:
        rv = model.values_to_rvs[v]
        transform = model.rvs_to_transforms[rv]
        unc_shape = initial_point[v.name].shape

        if transform is not None and rv.name in posterior_data_vars:
            cvar = rv.type()
            cvar.name = rv.name
            inputs_for_func.append(cvar)
            outputs_for_func.append(transform.forward(cvar, *rv.owner.inputs))
            posterior_names.append(rv.name)
        else:
            post_name = rv.name if rv.name in posterior_data_vars else v.name
            if post_name not in posterior_data_vars:
                raise KeyError(
                    f"Could not find {v.name!r} or {rv.name!r} in idata.posterior. "
                    f"Available: {list(posterior_data_vars)}"
                )
            inputs_for_func.append(v)
            outputs_for_func.append(v)
            posterior_names.append(post_name)

        unc_shapes.append(unc_shape)

    # Compile a single batched function
    joined = pt.matrix("upars_input", dtype="float64")
    replace = {}
    last = 0
    for cvar, unc_shape in zip(inputs_for_func, unc_shapes):
        alen = int(np.prod(unc_shape, dtype=int)) if unc_shape else 1
        replace[cvar] = joined[:, last : last + alen].reshape((joined.shape[0], *unc_shape))
        last += alen
    new_outputs = pytensor.graph.vectorize_graph(outputs_for_func, replace)
    joined_unc = pt.concatenate([pt.reshape(b, (pt.shape(b)[0], -1)) for b in new_outputs], axis=1)
    transform_func = pytensor.function([joined], joined_unc, on_unused_input="ignore")

    # Assemble the input matrix from posterior draws and apply the transform
    flat_blocks = []
    for post_name, unc_shape in zip(posterior_names, unc_shapes):
        alen = int(np.prod(unc_shape, dtype=int)) if unc_shape else 1
        flat_blocks.append(
            np.asarray(idata.posterior[post_name].values).reshape(chain * draw, alen)
        )
    upars_flat = transform_func(np.concatenate(flat_blocks, axis=1)).reshape(chain, draw, -1)
    n_params = upars_flat.shape[2]

    # Parameter labels: value_var name for scalars, name[i,j,...] for multi-element vars
    labels = []
    for v, unc_shape in zip(value_vars, unc_shapes):
        alen = int(np.prod(unc_shape, dtype=int)) if unc_shape else 1
        if alen == 1:
            labels.append(v.name)
        else:
            for idx in np.ndindex(*unc_shape):
                labels.append(f"{v.name}[{','.join(str(j) for j in idx)}]")

    return (
        DataArray(
            upars_flat,
            dims=("chain", "draw", "unconstrained_parameter"),
            coords={
                "chain": idata.posterior["chain"],
                "draw": idata.posterior["draw"],
                "unconstrained_parameter": labels,
            },
            name="upars",
        ),
        n_params,
    )


def _get_batched_func(inputs, outputs, initial_point, *, dtype="float64", on_unused_input="ignore"):
    """Compile a batched function mapping ``(N, n_params)`` unconstrained draws to ``outputs``.

    Parameters
    ----------
    inputs : list of PyTensor variables
        Model value variables (unconstrained parameters), in column order.
    outputs : list of PyTensor expressions
        Expressions to evaluate (e.g. log posterior, elementwise log likelihood).
    initial_point : dict
        Model initial point; used to determine the shape of each value variable.

    Returns
    -------
    func : callable
        Compiled PyTensor function ``func(q) -> list`` where ``q`` has shape ``(N, n_params)``.
    """
    import pytensor
    import pytensor.tensor as pt

    joined = pt.matrix("upars_joined", dtype=dtype)
    batch_size_var = joined.shape[0]
    replace = {}
    last = 0
    for v in inputs:
        shape = initial_point[v.name].shape
        v_length = int(np.prod(shape, dtype=int)) if shape else 1
        replace[v] = joined[:, last : last + v_length].reshape((batch_size_var, *tuple(shape)))
        last += v_length
    new_outputs = pytensor.graph.vectorize_graph(outputs, replace)
    return pytensor.function([joined], new_outputs, on_unused_input=on_unused_input)
