import hypney

export, __all__ = hypney.exporter()


@export
class FunctionLike(type):
    """An function-like type implemented as a class

    FunctionLike's return the result of some computation when called,
    just like a function, but are implemented as a class, and can thus use
    attributes, methods, inheritance, etc.

    Roughly, a call to a FunctionLike results in:
      * Making a new instance (`self`) as usual,
        including a call to `self.__init__(*args, **kwargs)`
      * `return self()`, rather than `return self`
    """

    def __call__(cls, *args, **kwargs):
        return super().__call__(*args, **kwargs).__call__()


@export
class Estimator(metaclass=FunctionLike):
    stat: hypney.Statistic

    def __init__(self, stat: hypney.Statistic, fix: dict = None, **kwargs):
        self.stat = stat
        self.fix = self.stat.model.validate_params(fix, set_defaults=False)

    def __call__(self):
        return self._compute()

    # Generic routines, may be useful

    def _free_params(self):
        return [p for p in self.stat.model.param_specs if p.name not in self.fix]

    def _param_sequence_to_dict(self, x):
        params = {
            p.name: hypney.utils.eagerpy.ensure_raw(x[..., i])
            for i, p in enumerate(self._free_params())
        }
        return {**params, **self.fix}
