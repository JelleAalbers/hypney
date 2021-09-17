import hypney

export, __all__ = hypney.exporter()


@export
class Estimator:
    stat: hypney.Statistic

    def __init__(self, stat, fix: dict = None, free: tuple = None):
        self.stat = stat

        if free is not None:
            if fix is not None:
                raise ValueError("Specify either free or fix, not both")
            fix = [pname for pname in self.stat.model.param_names if pname not in free]

        if fix is None:
            fix = dict()
        if isinstance(fix, (tuple, list)):
            fix = {pname: self.stat.model.defaults[pname] for pname in fix}
        fix = self.stat.model.validate_params(fix, set_defaults=False)
        self.fix = fix

    def __call__(self, data):
        raise NotImplementedError
