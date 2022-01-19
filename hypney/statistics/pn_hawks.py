"""Deficit hawks using P(observing fewer events) as the statistic
"""
from scipy import stats

import hypney
from .deficit_hawks import (
    AllRegionSimpleHawk,
    AllRegionFullHawk,
    FixedRegionSimpleHawk,
    FixedRegionFullHawk,
)

export, __all__ = hypney.exporter()


@export
class PNHawk:
    def _dist_params(self, params):
        # Distribution depends only on # expected events
        return dict(mu=self.model._rate(params))

    def _compute_scores(self, n, mu, frac):
        return self.model._to_tensor(
            stats.poisson.cdf(n, mu=hypney.utils.eagerpy.ensure_numpy(mu))
        )


@export
class PNAllRegionHawk(PNHawk, AllRegionSimpleHawk):
    pass


@export
class PNFixedRegionHawk(PNHawk, FixedRegionSimpleHawk):
    pass


# Alternate implementation via Full hawk. Should be much slower!


@export
class PNOneCut(hypney.Statistic):
    """Return Poisson P of observing <= the observed N events

    Yellin's pmax method uses P(observing more events), i.e. 1 - this.
    (We want something that yields LOW value for deficits instead,
     since we take the minimum over cuts/regions.)
    """

    def _compute(self, params):
        # ({batch_shape})
        mu = self.model._rate(params)
        n = len(self.data)
        if self.model._backend_name == "numpy":
            # Shortcut, avoids some overhead... :-(
            return stats.poisson.cdf(n, mu=mu.raw)
        poisson_model = hypney.models.poisson(mu=mu)
        return poisson_model.cdf(n)


@export
class PNFixedRegionHawkSlow(FixedRegionFullHawk):

    statistic_class = PNOneCut

    def _dist_params(self, params):
        # Distribution depends only on # expected events in the region
        return dict(mu=self.model._rate(params))


@export
class PNAllRegionHawkSlow(AllRegionFullHawk):

    statistic_class = PNOneCut

    def _dist_params(self, params):
        # Distribution depends only on # expected events in the region
        return dict(mu=self.model._rate(params))
