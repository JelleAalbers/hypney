import hypney

export, __all__ = hypney.exporter()


@export
class TransformedDataModel(hypney.WrappedModel):
    """Model for data that has been shifted, then scaled.

    Args (beyond those of Model):
     - orig_model: original model
     - shift: constant to add to data
     - scale: constant to multiply shifted data
    """

    shift = 0.0
    scale = 1.0

    def _data_from_orig(self, orig_data):
        """Apply to data generated from model"""
        return self.scale * (orig_data + self.shift)

    def _data_to_orig(self):
        """Return self.data, with reverse of _data_from_orig applied
        so it can be fed to original model.
        """
        return (self.data / self.scale) - self.shift

    def _transform_jac_det(self):
        return abs(1 / self.scale)

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
        self._orig_model = self._orig_model(data=self._data_to_orig())

    def _init_quantiles(self):
        # ppf not implemented yet
        pass

    # Simulation

    def _simulate(self, params):
        return self._data_from_orig(self._orig_model._simulate(params))

    def _rvs(self, size: int, params: dict):
        return self._data_from_orig(self._orig_model._rvs(size=size, params=params))

    # Methods using data / quantiles

    def _pdf(self, params):
        return self._orig_model._pdf(params) * self._transform_jac_det()

    def _cdf(self, params):
        result = self._orig_model._cdf(params)
        if self.scale < 0:
            result = 1 - result
        return result

    def _ppf(self, params):
        raise NotImplementedError

    # Methods not using data

    def _rate(self, params):
        return self._orig_model._rate(params)

    def _mean(self, params):
        return self._data_from_orig(self._orig_model._mean(params))

    def _std(self, params: dict):
        return self._orig_model._std(params) * self.scale
