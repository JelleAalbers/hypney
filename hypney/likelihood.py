import hypney as hp

export, __all__ = hp.exporter()


@export
class LikelihoodRatioExact(hp.NeymanConstruction):
    def __init__(self, *, log_likelihood, target, nuisance, neyman_grid):
        self.log_likelihood = log_likelihood
        super().__init__(target=target, nuisance=nuisance, neyman_grid=neyman_grid)
