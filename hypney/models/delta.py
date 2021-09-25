import numpy as np
import eagerpy as ep

import hypney

export, __all__ = hypney.exporter()


@export
class DiracDelta(hypney.Model):
    param_specs = hypney.RATE_LOC_PARAMS

    def _pdf(self, params):
        return ep.where(self.data[:, 0] == params["loc"], float("inf"), 0)

    def _cdf(self, params):
        return ep.where(self.data[:, 0] > params["loc"], 1, 0)

    def _ppf(self, params: dict) -> np.ndarray:
        return 0 * self.quantiles + params["loc"]

    def _rvs(self, size: int, params: dict) -> np.ndarray:
        return np.ones((size, 1)) * params["loc"]

    def _mean(self, params):
        return params["loc"]

    def _std(self, params):
        return 0.0
