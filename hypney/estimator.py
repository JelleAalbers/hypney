import hypney

export, __all__ = hypney.exporter()


@export
class Estimator:
    stat: hypney.Statistic

    def __init__(self, stat, fix=None, keep=None):
        self.stat = stat
        self.fix = self.stat.model._process_fix_keep(fix, keep)

    def __call__(self, data=hypney.NotChanged):
        if data is not hypney.NotChanged:
            stat = self.stat.freeze(data=data)
        return self._compute(stat)

    def _compute(self, stat):
        raise NotImplementedError
