import hypney

export, __all__ = hypney.exporter()


@export
class ScaleShiftData(hypney.WrappedModel):
    """A model which scales and shifts data, then feeds it to another model.

    Args (beyond those of Model):
     - orig_model: original model taking transformed parameters
     - shift: constant to add to scaled data
     - scale: constant to multiply data by
    """

    shift = 0.0
    scale = 1.0

    def _transform_data(self):
        result = self.data * self.scale + self.shift
        return result

    def _reverse_transform_data(self, orig_data):
        return (orig_data - self.shift) / self.scale

    def _transform_data_jac_det(self):
        return abs(self.scale)

    ##
    # Initialization
    ##

    def __init__(
        self, *args, shift=hypney.NotChanged, scale=hypney.NotChanged, **kwargs
    ):
        if shift is not hypney.NotChanged:
            self.shift = shift
        if scale is not hypney.NotChanged:
            self.scale = scale
        super().__init__(*args, **kwargs)

    def _init_data(self):
        self._orig_model = self._orig_model(data=self._transform_data())

    def _init_quantiles(self):
        raise NotImplementedError

    # Simulation

    def _simulate(self, params):
        return self._reverse_transform_data(super()._simulate(params))

    def _rvs(self, size: int, params: dict):
        return self._reverse_transform_data(super()._rvs(size=size, params=params))

    # Methods using data / quantiles

    def _pdf(self, params):
        return super()._pdf(params) * self._transform_data_jac_det()

    def _cdf(self, params):
        result = super()._cdf(params)
        if self.scale < 0:
            result = 1 - result
        return result

    def _ppf(self, params):
        result = super()._ppf(params)
        return self.scale * result + self.shift

    # Methods not using data

    def _rate(self, params):
        return super()._rate(params)

    def _mean(self, params):
        return self._reverse_transform_data(super()._mean(params))

    def _std(self, params: dict):
        # TODO: this is part of reverse_transform_data..
        orig_std = super()._std(params)
        return orig_std / self.scale


@export
class NegativeData(ScaleShiftData):
    scale = -1


@export
class NormalizedData(ScaleShiftData):
    def __init__(self, orig_model=hypney.NotChanged, *args, **kwargs):
        kwargs.setdefault("scale", orig_model.std())
        kwargs.setdefault("shift", orig_model.mean())
        return super().__init__(orig_model=orig_model, *args, **kwargs)
