import hypney

export, __all__ = hypney.exporter()


@export
class TransformedModel(hypney.Model):
    """A model which transforms data and/or parameters, then feeds them to another model

    Args (beyond those of Model):
     - orig_model: original model taking transformed parameters
     - transform_params: function mapping dict of params the new model takes
        to dict of params the old model takes
     - transform_data: function mapping data the new model takes
        to data the old model takes
    """

    # ... maybe we should integrate _transform_params and _transform_data in the base model instead?
    # Or would that lead to another layer wrapping pdf -> _pdf -> __pdf?

    _orig_model: hypney.Model

    def _transform_params(self, params: dict):
        return params

    def _transform_data(self):
        return self.data

    def _reverse_transform_data(self, orig_data):
        return orig_data

    def _transform_data_jac_det(self, params, orig_params):
        return 1.0

    ##
    # Initialization
    ##

    def __init__(
        self,
        orig_model=hypney.NotChanged,
        *args,
        transform_data=hypney.NotChanged,
        transform_params=hypney.NotChanged,
        **kwargs
    ):
        if transform_params is not hypney.NotChanged:
            self._transform_params = transform_params
        if transform_data is not hypney.NotChanged:
            self._transform_data = transform_data
        if orig_model is not hypney.NotChanged:
            # No need to make a copy now; any attempted state change
            # (set data, cut, change defaults...) will trigger that
            self._orig_model = orig_model
        kwargs.setdefault("observables", self._orig_model.observables)
        kwargs.setdefault("param_specs", self._orig_model.param_specs)
        super().__init__(*args, **kwargs)

    def _init_data(self):
        self._orig_model = self._orig_model(data=self._transform_data())

    def _init_cut(self):
        if self._has_redefined("_transform_data") and self.cut != hypney.NoCut:
            raise NotImplementedError(
                "Rectangular cuts may not be rectangular after transformation"
            )
        self._orig_model = self._orig_model(cut=self.cut)

    ##
    # Simulation
    ##

    def _check_reverse_transform(self):
        if self._has_redefined(
            "_transform_data", from_base=TransformedModel
        ) and not self._has_redefined(
            "_reverse_transform_data", from_base=TransformedModel
        ):
            raise NotImplementedError("Missing reverse transformation")

    def _simulate(self, params):
        self._check_reverse_transform()
        return self._reverse_transform_data(
            self._orig_model._simulate(self._transform_params(params))
        )

    def _rvs(self, size: int, params: dict):
        self._check_reverse_transform()
        return self._reverse_transform_data(
            self._orig_model._rvs(size=size, params=self._transform_params(params))
        )

    ##
    # Methods using data
    ##

    def _apply_cut(self):
        # See init_cut: cut is NoCut, or we don't transform the data
        if self.cut == hypney.NoCut:
            return self.data
        # If we get here, we're not transforming data -- see init_cut.
        assert not self._has_redefined("_transform_data")
        return super()._apply_cut()

    def _check_transform_data_jac_det(self):
        if self._has_redefined(
            "_transform_data", from_base=TransformedModel
        ) and not self._has_redefined(
            "_transform_data_jac_det", from_base=TransformedModel
        ):
            raise NotImplementedError("Jacobian determinant missing")

    def _pdf(self, params):
        self._check_transform_data_jac_det()
        orig_params = self._transform_params(params)
        return self._orig_model._pdf(orig_params) * self._transform_data_jac_det(
            params, orig_params
        )

    def _cdf(self, params):
        if self._has_redefined("_transform_data", from_base=TransformedModel):
            raise NotImplementedError(
                "Don't know how to define CDF for generic transformation"
            )
        return self._orig_model._cdf(self._transform_params(params))

    ##
    # Methods not using data
    ##

    def _rate(self, params):
        return self._orig_model._rate(self._transform_params(params))

    def _cut_efficiency(self, params: dict):
        return self._orig_model._cut_efficiency(self._transform_params(params))


@export
class NegativeData(TransformedModel):
    def _transform_data(self):
        return -self.data

    def _reverse_transform_data(self):
        return -self.data

    def _transform_data_jac_det(self, params, orig_params):
        return 1

    def _cdf(self, params):
        assert len(self.observables) == 1
        return 1 - self._orig_model._cdf(params)
