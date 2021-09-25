import hypney

export, __all__ = hypney.exporter()


@export
class WrappedModel(hypney.Model):
    _orig_model: hypney.Model

    def __init__(self, orig_model=hypney.NotChanged, *args, **kwargs):
        if orig_model is not hypney.NotChanged:
            # No need to make a copy now; any attempted state change
            # (set data, cut, change defaults...) will trigger that
            self._orig_model = orig_model
        kwargs.setdefault("observables", self._orig_model.observables)
        kwargs.setdefault("param_specs", self._orig_model.param_specs)
        super().__init__(*args, **kwargs)

    def _init_data(self):
        self._orig_model = self._orig_model(data=self.data)

    def _apply_cut(self):
        return self._orig_model.apply_cut()


@export
class TransformParams(WrappedModel):
    """A model which transforms parameters, then feeds them to another model

    Args (beyond those of Model):
     - orig_model: original model taking transformed parameters
     - transform_params: function mapping dict of params the new model takes
        to dict of params the old model takes
    """

    # ... maybe we should integrate _transform_params and _transform_data in the base model instead?
    # Or would that lead to another layer wrapping pdf -> _pdf -> __pdf?

    def _transform_params(self, params: dict):
        return params

    # Initialization

    def __init__(self, *args, transform_params=hypney.NotChanged, **kwargs):
        if transform_params is not hypney.NotChanged:
            self._transform_params = transform_params
        super().__init__(*args, **kwargs)

    # Simulation

    def _simulate(self, params):
        return self._orig_model._simulate(self._transform_params(params))

    def _rvs(self, size: int, params: dict):
        return self._orig_model._rvs(size=size, params=self._transform_params(params))

    def _pdf(self, params):
        return self._orig_model._pdf(self._transform_params(params))

    def _cdf(self, params):
        return self._orig_model._cdf(self._transform_params(params))

    def _ppf(self, params):
        return self._orig_model._ppf(self._transform_params(params))

    # Methods not using data

    def _rate(self, params):
        return self._orig_model._rate(self._transform_params(params))

    def _mean(self, params):
        return self._orig_model._mean(self._transform_params(params))

    def _std(self, params: dict):
        return self._orig_model._std(self._transform_params(params))

    def _cut_efficiency(self, params: dict):
        return self._orig_model._cut_efficiency(self._transform_params(params))


@export
class TransformData(WrappedModel):
    """A model which transforms data, then feeds it to another model.

    Only constant scales and offsets are supported for now.

    Args (beyond those of Model):
     - orig_model: original model taking transformed parameters
     - data_shift: constant to add to scaled data
     - data_scale: constant to multiply data by
    """

    _shift = 0.0
    _scale = 1.0

    def _transform_data(self):
        result = self.data * self._scale + self._shift
        return result

    def _reverse_transform_data(self, orig_data):
        return (orig_data - self._shift) / self._scale

    def _transform_data_jac_det(self):
        return abs(self._scale)

    ##
    # Initialization
    ##

    def __init__(
        self, *args, shift=hypney.NotChanged, scale=hypney.NotChanged, **kwargs
    ):
        if shift is not hypney.NotChanged:
            self._shift = shift
        if scale is not hypney.NotChanged:
            self._scale = scale
        super().__init__(*args, **kwargs)

    def _init_data(self):
        self._orig_model = self._orig_model(data=self._transform_data())

    def _init_cut(self):
        if self._has_redefined("_transform_data") and self.cut != hypney.NoCut:
            raise NotImplementedError(
                "Rectangular cuts may not be rectangular after transformation"
            )
        self._orig_model = self._orig_model(cut=self.cut)

    # Simulation

    def _simulate(self, params):
        return self._reverse_transform_data(self._orig_model._simulate(params))

    def _rvs(self, size: int, params: dict):
        return self._reverse_transform_data(
            self._orig_model._rvs(size=size, params=params)
        )

    # Methods using data / quantiles

    def _apply_cut(self):
        # See init_cut: cut is NoCut, or we don't transform the data
        if self.cut == hypney.NoCut:
            return self.data
        # If we get here, we're not transforming data -- see init_cut.
        assert not self._has_redefined("_transform_data")
        return super()._apply_cut()

    def _pdf(self, params):
        return self._orig_model._pdf(params) * self._transform_data_jac_det()

    def _cdf(self, params):
        result = self._orig_model._cdf(params)
        if self._scale < 0:
            result = 1 - result
        return result

    def _ppf(self, params):
        result = self._orig_model._ppf(params)
        return self._scale * result + self._shift

    # Methods not using data

    def _rate(self, params):
        return self._orig_model._rate(params)

    def _mean(self, params):
        orig_mean = self._orig_model._mean(params)
        return self._reverse_transform_data(orig_mean)

    def _std(self, params: dict):
        orig_std = self._orig_model._std(params)
        return orig_std / self._scale

    def _cut_efficiency(self, params: dict):
        return self._orig_model._cut_efficiency(params)


@export
class NegativeData(TransformData):
    _scale = -1


@export
class NormalizedData(TransformData):
    def __init__(self, orig_model=hypney.NotChanged, *args, **kwargs):
        kwargs.setdefault("scale", orig_model.std())
        kwargs.setdefault("shift", orig_model.mean())
        return super().__init__(orig_model=orig_model, *args, **kwargs)
