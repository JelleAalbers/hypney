import hypney

export, __all__ = hypney.exporter()


@export
class TransformedModel(hypney.Model):
    """A model which transforms data and/or parameters, then feeds them to another model
    """

    # ... maybe we should integrate _transform_params and _transform_data in the base model instead?
    # Or would that lead to another layer wrapping pdf -> _pdf -> __pdf?

    _orig_model: hypney.Model

    def _transform_params(self, params: dict):
        return params

    def _transform_data(self, data):
        return data

    ##
    # Initialization
    ##

    def __init__(
        self,
        *args,
        orig_model=hypney.NotChanged,
        transform_data=hypney.NotChanged,
        transform_params=hypney.NotChanged,
        **kwargs
    ):
        if transform_params is not hypney.NotChanged:
            self._transform_params = transform_params
        if transform_data is not hypney.NotChanged:
            self._transform_data = transform_data
        if orig_model is not hypney.NotChanged:
            self._orig_model = orig_model
        kwargs.setdefault("observables", self._orig_model.observables)
        kwargs.setdefault("param_specs", self._orig_model.param_specs)
        super().__init__(*args, **kwargs)

    def _init_data(self, data):
        self._orig_model = self._orig_model(data=self._transform_data(data))

    def _init_cut(self):
        if self._has_redefined("_transform_data") and self.cut != hypney.NoCut:
            raise NotImplementedError(
                "Rectangular cuts may not be rectangular after transformation"
            )
        self._orig_model = self._orig_model(cut=self.cut)

    ##
    # Simulation
    ##

    def _simulate(self, params):
        if self._has_redefined('_transform_data'):
            raise NotImplementedError("Simulating transformed data requires inverse transformation")
        return self._orig_model._simulate(self._transform_params(params))

    def _rvs(self, params, size):
        if self._has_redefined('_transform_data'):
            raise NotImplementedError("Simulating transformed data requires inverse transformation")
        return self._orig_model._rvs(self._transform_params(params), size)

    ##
    # Methods using data
    ##

    def _apply_cut(self):
        return self._orig_model._apply_cut()

    def _diff_rate(self, params: dict):
        if self._has_redefined('_transform_data'):
            raise NotImplementedError("diff rate in transformed data requires Jacobian")
        return self._orig_model._diff_rate(self._transform_params(params=params))

    def _pdf(self, params):
        if self._has_redefined('_transform_data'):
            raise NotImplementedError("PDF in transformed data requires Jacobian")
        return self._orig_model._pdf(self._transform_params(params))

    def _cdf(self, params):
        # For non-increasing transformations this will be invalid...
        return self._orig_model._cdf(self._transform_params(params))

    ##
    # Methods not using data
    ##

    def _rate(self, params):
        return self._orig_model._rate(self._transform_params(params))

    def _cut_efficiency(self, params: dict):
        return self._orig_model._cut_efficiency(self._transform_params(params))


@export
def transform_params(
    orig_model: hypney.Model, transform_f: callable, param_specs: tuple, **kwargs
):
    """Return a new model which transforms params, then feeds them to orig_model

    Args:
     - orig_model: original model taking transformed parameters
     - transform_f: function mapping dict of params the new model takes
        to dict of params the old model takes
     - param_specs: Parameter specification of new model

    """
    return TransformedModel(
        wrapped_model=orig_model,
        transform_params=transform_f,
        param_specs=param_specs,
        **kwargs,
    )


@export
def transform_data(
    orig_model: hypney.Model, transform_data: callable, observables=None, **kwargs
):
    """Return a new model which transforms data, then feeds it to orig_model

    Args:
     - orig_model: original model taking transformed data
     - transform_data: function mapping data the new model expects
        to data the old model expects
     - observables: Observables of the new model
        If omitted, assume they do not change. This is likely incorrect for
        bounded parameters.
    """
    if observables:
        kwargs["observables"] = observables
    return TransformedModel(
        wrapped_model=orig_model, transform_data=transform_data, **kwargs,
    )
