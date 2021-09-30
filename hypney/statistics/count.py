import hypney

export, __all__ = hypney.exporter()


@export
class Count(hypney.IndependentStatistic):
    def _compute(self, params):
        return len(self.data)

    def _build_dist(self):
        return hypney.models.poisson()

    def _dist_params(self, params):
        return dict(mu=self.model._rate(params))
