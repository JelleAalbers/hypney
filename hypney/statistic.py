import hypney as hp

export, __all__ = hp.exporter(also_export=["_DEFAULTS"])

_DEFAULTS = dict(
    interval_kind="central",
    confidence_level=0.68,
)


class Statistic:
    @classmethod
    def from_function(cls, function):
        statistic = cls()
        statistic.compute = function
        return statistic

    def compute(self, data, **parameters):
        raise NotImplementedError

    def confidence_interval(
        self,
        data,
        kind=_DEFAULTS["interval_kind"],
        confidence_level=_DEFAULTS["confidence_level"],
    ):
        raise NotImplementedError


@export
class NeymanConstruction(Statistic):
    def __init__(self, *, target, nuisance, neyman_grid):
        super().__init__()

    def compute(self, data, **parameters):
        raise NotImplementedError

    def confidence_interval(
        self,
        data,
        kind=_DEFAULTS["interval_kind"],
        confidence_level=_DEFAULTS["confidence_level"],
    ):
        print("TODO")
