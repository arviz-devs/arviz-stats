# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-public-methods
"""Base class for sampling wrappers."""

from arviz_base import convert_to_datatree
from xarray import apply_ufunc


class SamplingWrapper:
    """Class wrapping sampling routines for its usage via ArviZ.

    Using a common class, all inference backends can be supported in ArviZ. Hence, statistical
    functions requiring refitting like Leave Future Out or Simulation Based Calibration can be
    performed from ArviZ.

    Parameters
    ----------
    model
        The model object used for sampling.
    idata_orig : optional
        Original inference data object (InferenceData or datatree).
    log_lik_fun : callable, optional
        For simple cases where the pointwise log likelihood is a Python function, this
        function will be used to calculate the log likelihood. Otherwise,
        ``point_log_likelihood`` method must be implemented. It's callback must be
        ``log_lik_fun(*args, **log_lik_kwargs)`` and will be called using
        :func:`wrap_xarray_ufunc` or :func:`xarray:xarray.apply_ufunc` depending
        on the value of `is_ufunc`.

        For more details on ``args`` or ``log_lik_kwargs`` see the notes and
        parameters ``posterior_vars`` and ``log_lik_kwargs``.
    is_ufunc : bool, default True
        If True, call ``log_lik_fun`` using :func:`xarray:xarray.apply_ufunc` otherwise
        use :func:`wrap_xarray_ufunc`.
    posterior_vars : list of str, optional
        List of variable names to unpack as ``args`` for ``log_lik_fun``. Each string in
        the list will be used to retrieve a DataArray from the Dataset in the posterior
        group and passed to ``log_lik_fun``.
    sample_kwargs : dict, optional
        Sampling kwargs are stored as class attributes for their usage in the ``sample``
        method.
    idata_kwargs : dict, optional
        kwargs are stored as class attributes to be used in the ``get_inference_data`` method.
    log_lik_kwargs : dict, optional
        Keyword arguments passed to ``log_lik_fun``.
    apply_ufunc_kwargs : dict, optional
        Passed to :func:`xarray:xarray.apply_ufunc` or :func:`wrap_xarray_ufunc`.
    """

    def __init__(
        self,
        model,
        idata_orig=None,
        log_lik_fun=None,
        is_ufunc=True,
        posterior_vars=None,
        sample_kwargs=None,
        idata_kwargs=None,
        log_lik_kwargs=None,
        apply_ufunc_kwargs=None,
    ):
        self.model = model

        if idata_orig is not None:
            self.idata_orig = convert_to_datatree(idata_orig)
        else:
            self.idata_orig = idata_orig

        if log_lik_fun is None or callable(log_lik_fun):
            self.log_lik_fun = log_lik_fun
            self.is_ufunc = is_ufunc
            self.posterior_vars = posterior_vars
        else:
            raise TypeError("log_like_fun must be a callable object or None")

        self.sample_kwargs = {} if sample_kwargs is None else sample_kwargs
        self.idata_kwargs = {} if idata_kwargs is None else idata_kwargs
        self.log_lik_kwargs = {} if log_lik_kwargs is None else log_lik_kwargs
        self.apply_ufunc_kwargs = {} if apply_ufunc_kwargs is None else apply_ufunc_kwargs

    def sel_observations(self, idx):
        """Select a subset of the observations in idata_orig.

        **Not implemented**: This method must be implemented by the SamplingWrapper subclasses.
        It is documented here to show its format and call signature.

        Parameters
        ----------
        idx
            Indexes to separate from the rest of the observed data.

        Returns
        -------
        modified_observed_data
            Observed data whose index is *not* ``idx``
        excluded_observed_data
            Observed data whose index is ``idx``
        """
        raise NotImplementedError("sel_observations method must be implemented for each subclass")

    def sample(self, modified_observed_data):
        """Sample ``self.model`` on the ``modified_observed_data`` subset.

        **Not implemented**: This method must be implemented by the SamplingWrapper subclasses.
        It is documented here to show its format and call signature.

        Parameters
        ----------
        modified_observed_data
            Data to fit the model on.

        Returns
        -------
        fitted_model
            Result of the fit.
        """
        raise NotImplementedError("sample method must be implemented for each subclass")

    def get_inference_data(self, fitted_model):
        """Convert the ``fitted_model`` to an InferenceData object.

        **Not implemented**: This method must be implemented by the SamplingWrapper subclasses.
        It is documented here to show its format and call signature.

        Parameters
        ----------
        fitted_model
            Result of the current fit.

        Returns
        -------
        idata_current: datatree or InferenceData
            Inference data object containing the samples in ``fitted_model``
        """
        raise NotImplementedError("get_inference_data method must be implemented for each subclass")

    def log_likelihood__i(self, excluded_obs, idata__i):
        r"""Get the log likelilhood samples :math:`\log p_{post(-i)}(y_i)`.

        Calculate the log likelihood of the data contained in excluded_obs using the
        model fitted with this data excluded, the results of which are stored in ``idata__i``.

        Parameters
        ----------
        excluded_obs
            Observations for which to calculate their log likelihood. The second item from
            the tuple returned by `sel_observations` is passed as this argument.
        idata__i: datatree or InferenceData
            Inference results of refitting the data excluding some observations. The
            result of `get_inference_data` is used as this argument.

        Returns
        -------
        log_likelihood: xr.Dataarray
            Log likelihood of ``excluded_obs`` evaluated at each of the posterior samples
            stored in ``idata__i``.
        """
        if self.log_lik_fun is None:
            raise NotImplementedError(
                "When `log_like_fun` is not set during class initialization "
                "log_likelihood__i method must be overwritten"
            )
        posterior = idata__i.posterior

        if not isinstance(excluded_obs, list | tuple):
            excluded_obs = (excluded_obs,)

        arys = (*excluded_obs, *[posterior[var_name] for var_name in self.posterior_vars])
        log_lik_idx = apply_ufunc(
            self.log_lik_fun,
            *arys,
            kwargs=self.log_lik_kwargs,
            **self.apply_ufunc_kwargs,
        )
        return log_lik_idx

    def check_implemented_methods(self, methods):
        """Check that all methods listed are implemented.

        Not all functions that require refitting need to have all the methods implemented in
        order to work properly. This function should be used before using the SamplingWrapper and
        its subclasses to get informative error messages.

        Parameters
        ----------
        methods: list
            Check all elements in methods are implemented.

        Returns
        -------
            List with all non implemented methods
        """
        supported_methods = (
            "sel_observations",
            "sample",
            "get_inference_data",
            "log_likelihood__i",
        )
        bad_methods = [method for method in methods if method not in supported_methods]
        if bad_methods:
            raise ValueError(
                f"Not all method(s) in {bad_methods} supported. "
                f"Supported methods in SamplingWrapper subclasses are:{supported_methods}"
            )

        not_implemented = []
        for method_name in methods:
            instance_method = getattr(self.__class__, method_name)
            base_class_method = getattr(SamplingWrapper, method_name)
            if instance_method == base_class_method:
                not_implemented.append(method_name)
        return not_implemented
