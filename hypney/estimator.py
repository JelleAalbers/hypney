import hypney

export, __all__ = hypney.exporter()


@export
class Estimator:
    stat: hypney.Statistic

    def __init__(self, stat, fix: dict = None):
        self.stat = stat

        if fix is None:
            fix = dict()
        else:
            fix = self.stat.validate_params(fix, set_defaults=False)
        self.fix = fix

    def __call__(self, data):
        raise NotImplementedError
