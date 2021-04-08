import hypney as hp

export, __all__ = hp.exporter()


@export
class Model:
    def simulate(self, **parameters):
        raise NotImplementedError

    def log_likelihood(self, **kwargs):
        raise NotImplementedError


@export
class CountingExperiment(Model):
    def __init__(self, signal=None, background=0.0, efficiency=1.0):
        raise NotImplementedError


@export
class ScipyStatsModel(Model):
    pass
