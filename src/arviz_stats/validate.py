"""Validator functions for common arguments."""

from arviz_base import rcParams


def validate_dims(dims):
    """Validate `dims` argument.

    Uses the default in rcParams and ensures the returned object is a list.

    Parameters
    ----------
    dims : str, sequence of hashable, or None

    Returns
    -------
    list
    """
    if dims is None:
        dims = rcParams["data.sample_dims"]
    if isinstance(dims, str):
        dims = [dims]
    return list(dims)


def validate_dims_chain_draw_axis(dims):
    """Validate `dims` argument for functions that use chain_axis and draw_axis.

    In such cases, dims can have length 1 or 2 depending on there being a chain dimension.

    Returns
    -------
    list
        List of dimensions
    int or None
        Positional index for chain dimension
    int
        Positional index for draw dimension
    """
    dims = validate_dims(dims)
    draw_axis = -1
    if len(dims) == 1:
        chain_axis = None
    elif len(dims) == 2:
        chain_axis = -2
    else:
        raise ValueError("dims can only have 1 or 2 elements")
    return dims, chain_axis, draw_axis


def validate_ci_prob(prob):
    """Validate `prob`/`ci_prob` argument.

    Returns
    -------
    float
    """
    if prob is None:
        prob = rcParams["stats.ci_prob"]
    elif not 1 >= prob > 0:
        raise ValueError("The value of prob should be in the interval (0, 1]")
    return prob
