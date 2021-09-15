from copy import copy

import numpy as np

import hypney
from hypney import NotChanged

export, __all__ = hypney.exporter()


@export
class Statistic(hypney.DataContainer):
    # Does statistic depends on parameters? If not, it depends only on the data
    # (like the count of events).
    # In the latter case,
    #   * compute takes no arguments (data is in self.data)
    #   * compute will be run on initialization (if data is given)
    #     and the result stored in _result, which __call__ will return.
    #   * The distribution may still depend on parameters.
    param_dependent = True

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
            self.dist = hypney.transform_parameters(
                self._build_dist(), self._dist_params
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
        if not self.param_dependent:
            # Precompute result on the data.
            self._result = self._compute()

        if not self.keep_data:
            # Statistic relies only on stuff computed in init_data,
            # so we can throw away our reference to the data
            self.data = None
        super()._init_data()

    def __call__(self, params: dict = None, data=NotChanged):
        if data is NotChanged:
            if not self.param_dependent:
                return self._result
            if self.data is None:
                raise ValueError("Must provide data")
        else:
            # Data was passed, work on a copy of self using the new data
            self = copy(self)
            self._set_data(data)

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
