import numpy as np

import hypney


export, __all__ = hypney.exporter()


@export
class Count(hypney.Statistic):
    def _compute(self, params):
        return self.data.shape[-2] * self.model.backend.ones(
            self.model._batch_shape(params)
        )

    def _build_dist(self):
        return hypney.models.poisson()

    def _dist_params(self, params):
        return dict(mu=self.model._rate(params))


@export
class AugmentedCount(hypney.Statistic):
    """Number of events in dataset + a random [0,1] number
    (fixed when initializing data)
    """

    def _init_data(self):
        self.augment = np.random.rand()
        return super()._init_data()

    def _compute(self, params):
        result = self.data.shape[-2] * self.model.backend.ones(
            self.model._batch_shape(params)
        )
        return result + self.augment

    def _dist_params(self, params):
        return dict(mu=self.model._rate(params))
