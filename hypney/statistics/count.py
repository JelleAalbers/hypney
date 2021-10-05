import hypney

export, __all__ = hypney.exporter()


@export
class Count(hypney.Statistic):
    def _compute(self, params):
        return self.data.shape[-2]

    def _build_dist(self):
        return hypney.models.poisson()

    def _dist_params(self, params):
        return dict(mu=self.model._rate(params))
