import hypney

export, __all__ = hypney.exporter()


@export
class Reparametrized(hypney.WrappedModel):
    """A model which transforms parameters, then feeds them to another model

    Args (beyond those of Model):
     - orig_model: original model taking transformed parameters
     - transform_params: function mapping dict of params the new model takes
        to dict of params the old model takes
    """

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
