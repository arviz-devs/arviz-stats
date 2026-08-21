"""Shared PyMC helper functions for PSIS-LOO-CV."""

import numpy as np


def _get_observed_rv(model, idata, var_name):
    """Identify the observed random variable to use for PSIS-LOO-CV.

    Parameters
    ----------
    model : Model
        The PyMC model to check.
    idata : DataTree
        The DataTree object. If it contains an ``observed_data`` group,
        ``var_name`` is validated against it. Otherwise, the group is optional.
    var_name : str or None
        The name of the observed variable to use. If None, the function will
        attempt to infer the observed variable.

    Returns
    -------
    observed_rv : RandomVariable
        The observed random variable to use for PSIS-LOO-CV.
    var_name : str
        The name of the observed variable.
    """
    observed = list(model.observed_RVs)

    if (
        var_name is not None
        and hasattr(idata, "observed_data")
        and var_name not in idata.observed_data
    ):
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
    if not matches:
        raise ValueError(
            f"`{var_name!r}` does not match any observed random variable in the model. "
            f"Available: {[rv.name for rv in observed]}."
        )
    return matches[0], var_name


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
        Function ``func(q) -> list`` where ``q`` has shape ``(N, n_params)``.
    """
    import pytensor
    import pytensor.tensor as pt

    block_inputs = []
    block_slices = []
    replace = {}
    last = 0
    for v in inputs:
        shape = initial_point[v.name].shape
        v_length = int(np.prod(shape, dtype=int)) if shape else 1
        block = pt.matrix(f"upars_{v.name}", dtype=dtype)
        block_inputs.append(block)
        block_slices.append(slice(last, last + v_length))
        replace[v] = block.reshape((block.shape[0], *tuple(shape)))
        last += v_length
    new_outputs = pytensor.graph.vectorize_graph(outputs, replace)
    compiled = pytensor.function(block_inputs, new_outputs, on_unused_input=on_unused_input)

    def func(q):
        return compiled(*(np.ascontiguousarray(q[:, s]) for s in block_slices))

    return func
