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
        else:
            if self.stat.data is None:
                raise ValueError("Provide data")
            stat = self.stat
        return self._compute(stat)

    def _compute(self, stat):
        raise NotImplementedError

    # Generic routines, may be useful

    def _free_params(self):
        return [p for p in self.stat.model.param_specs if p.name not in self.fix]

    def _param_sequence_to_dict(self, x):
        params = {
            p.name: hypney.utils.eagerpy.ensure_raw(x[i])
            for i, p in enumerate(self._free_params())
        }
        return {**params, **self.fix}
