from copy import copy

import numpy as np

import hypney
from hypney import NotChanged

export, __all__ = hypney.exporter()


class Statistic(hypney.DataContainer):
    pass


@export
class Statistic(hypney.DataContainer):
    # Is data necessary to compute the statistic on different parameters?
    # e.g. if _init_data computes sufficient summary statistics, it won't be
    keep_data = False

    # Model of the data.
    model: hypney.Model

    # Model of the statistic's results; takes the same parameters
    dist: hypney.Model

    def __init__(self, model: hypney.Model, data=None, dist=None):
        self.model = model
        if not self.param_dependent:
            self.keep_data = False

        if dist is None and hasattr(self, "_build_dist"):
            self.dist = hypney.models.TransformedModel(
                self._build_dist(),
                transform_params=self._dist_params,
                param_specs=self.model.param_spec,
            )
        else:
            self.dist = dist

        super().__init__(data=data)

    def _dist_params(self, params):
        """Return distribution params given model params"""
        return dict()

    def validate_data(self, data):
        return self.model.validate_data(data)

    def _init_data(self):
        if not self.keep_data:
            # Statistic relies only on stuff computed in init_data,
            # so we can throw away our reference to the data
            self.data = None
        super()._init_data()

    def freeze(self, data=NotChanged) -> Statistic:
        """Return a model with possibly changed data"""
        if data is NotChanged:
            return self
        new_self = copy(self)
        new_self._set_data(data)
        return new_self

    def __call__(self, params: dict = None, data=NotChanged) -> float:
        self = self.freeze(data)
        if self.data is None:
            raise ValueError("Must provide data")

        params = self.model.validate_params(params)
        return self._compute(params)

    def _compute(self, params):
        raise NotImplementedError

    def rvs(self, size=1, params=None):
        """Return statistic evaluated on simulated data,
        generated from model with params"""
        results = np.zeros(size)
        for i in range(len(size)):
            sim_data = self.model.simulate(params=params)
            results[i] = self(data=sim_data, params=params)
        return results


@export
class IndependentStatistic(Statistic):
    """Statistic depending only on data, not on any parameters.

    The distribution may still depend on parameters.

    For speed, the result will be precomputed as soon as data is known.
    """

    keep_data = False

    def _init_data(self):
        # Precompute result on the data.
        self._result = self._compute()
        super()._init_data()

    def __call__(self, params: dict = None, data=NotChanged):
        self = self.freeze(data)
        if self.data is None:
            raise ValueError("Must provide data")
        return self._result

    def _compute(self):
        raise NotImplementedError
