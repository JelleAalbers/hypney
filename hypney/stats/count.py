import hypney

export, __all__ = hypney.exporter()


@export
class Count(hypney.Statistic):
    param_dependent = False

    def _compute(self):
        return len(self.data)

    def _build_dist(self):
        return hypney.models.Poisson()

    def _dist_params(self, params):
        return dict(mu=self.model.rate(params))
