import hypney

from scipy import optimize

export, __all__ = hypney.exporter()


class UpperLimit(hypney.Estimator):
    def __init__(self, poi: str, *args, cl=0.9, poi_max=None, **kwargs):
        self.poi = poi
        self.cl = 0.9
        self.poi_max = poi_max
        super().__init__(*args, **kwargs)

    def __call__(self, data, bestfit):
        stat = self.stat.freeze(data)

        def objective(params):
            return self.cl - stat.dist.cdf(data=stat(params), params=params)

        if self.poi_max is not None:
            poi_max = self.poi_max
        else:
            # Get max from the bound
            poi_max = self.stat.model.param_spec_for(self.poi).max
        if poi_max == float("inf"):
            raise ValueError("POI {self.poi} has no upper bound, give poi_max")

        return optimize.brentq(objective, bestfit, max)
