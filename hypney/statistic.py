from copy import copy

import eagerpy as ep
import numpy as np

import hypney
from hypney import NotChanged

export, __all__ = hypney.exporter()


@export
class Statistic:
    model: hypney.Model  # Model of the data
    dist: hypney.Model  # Model of the statistic; takes same parameters

    def __init__(self, model: hypney.Model, data=hypney.NotChanged, dist=None):
        self.model = model

        if dist is None and hasattr(self, "_build_dist"):
            self.dist = hypney.models.TransformedModel(
                self._build_dist(),
                transform_params=self._dist_params,
                param_specs=self.model.param_specs,
            )
        else:
            self.dist = dist

        self._set_data(data)

    def _set_data(self, data):
        if data is not hypney.NotChanged:
            self.model = self.model(data=data)
        # self.data is just self.model.data, see below
        if self.data is not None:
            self._init_data()

    @property
    def data(self) -> ep.types.NativeTensor:
        return self.model.data

    def _dist_params(self, params):
        """Return distribution params given model params"""
        _ = params  # Prevent static analyzer warning
        return dict()

    def validate_data(self, data):
        return self.model.validate_data(data)

    def _init_data(self):
        """Initialize self.data (either from construction or data change)"""
        pass

    def set(self, data=NotChanged):
        """Return a statistic with possibly changed data"""
        if data is NotChanged:
            return self
        new_self = copy(self)
        new_self._set_data(data)
        return new_self

    def __call__(self, data=NotChanged, params: dict = None, **kwargs):
        return self.compute(data=data, params=params, **kwargs)

    def compute(self, data=NotChanged, params: dict = None, **kwargs):
        return self.compute_(data=data, params=params, **kwargs).raw

    def compute_(self, data=NotChanged, params: dict = None, **kwargs) -> ep.TensorType:
        self = self.set(data)
        if self.data is None:
            raise ValueError("Must provide data")

        params = self.model.validate_params(params, **kwargs)
        return self._compute(params)

    def _compute(self, params):
        raise NotImplementedError

    def rvs(self, size=1, params=None, **kwargs) -> np.ndarray:
        """Return statistic evaluated on simulated data,
        generated from model with params"""
        params = self.model.validate_params(params, **kwargs)
        results = np.zeros(size)
        for i in range(size):
            sim_data = self.model._simulate(params=params)
            results[i] = self(data=sim_data, params=params).numpy().item()
        return results


@export
class IndependentStatistic(Statistic):
    """Statistic depending only on data, not on any parameters.

    The distribution may still depend on parameters.

    For speed, the result will be precomputed as soon as data is known.
    """

    def _init_data(self):
        # Precompute result on the data.
        self._result = self._compute()
        super()._init_data()

    def __call__(self, params: dict = None, data=NotChanged):
        self = self.set(data)
        if self.data is None:
            raise ValueError("Must provide data")
        return self._result

    def _compute(self):
        raise NotImplementedError
