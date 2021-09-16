import numpy as np

import hypney

export, __all__ = hypney.exporter()


@export
class DiracDelta(hypney.Model):
    param_specs = hypney.RATE_LOC_PARAMS

    def _cdf(self, params):
        return np.where(self.data[:, 0] > params["loc"], 1, 0)

    def _pdf(self, params):
        return np.where(self.data[:, 0] == params["loc"], float("inf"), 0)

    def _rvs(self, params, size):
        return np.ones((size, 1)) * params["loc"]
