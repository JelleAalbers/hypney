import typing as ty

import eagerpy as ep
import numpy as np
from scipy import stats

import hypney

export, __all__ = hypney.exporter()


class ScipyUnivariate(hypney.Model):
    dist: ty.Union[stats.rv_continuous, stats.rv_discrete]
    param_specs = hypney.RATE_LOC_SCALE_PARAMS

    def _dist_params(self, params):
        return {k: v for k, v in params.items() if k != hypney.DEFAULT_RATE_PARAM.name}

    def _rvs(self, params: dict, size: int = 1) -> ep.TensorType:
        return self.dist.rvs(size=size, **self._dist_params(params))[:, None]

    def _pdf(self, params: dict) -> ep.TensorType:
        pdf = self.dist.pdf if hasattr(self.dist, "pdf") else self.dist.pmf
        return ep.astensor(pdf(self.data[:, 0].numpy(), **self._dist_params(params)))

    def _cdf(self, params: dict) -> np.ndarray:
        return ep.astensor(
            self.dist.cdf(self.data[:, 0].numpy(), **self._dist_params(params))
        )


@export
class From1DHistogram(ScipyUnivariate):
    def __init__(self, histogram, bin_edges=None, *args, **kwargs):
        if bin_edges is None:
            # We probably got some kind of histogram container
            if isinstance(histogram, tuple) and len(histogram) == 2:
                histogram, bin_edges = histogram
            elif hasattr(histogram, "to_numpy"):
                # boost_histogram / hist
                histogram, bin_edges = histogram.to_numpy()
            elif hasattr(histogram, "bin_edges"):
                # multihist
                histogram, bin_edges = histogram.histogram, histogram.bin_edges
            else:
                raise ValueError("Pass histogram and bin edges arrays")

        self.dist = stats.rv_histogram(
            (histogram, bin_edges),
        )
        super().__init__(*args, **kwargs)


# Create classes for all continuous distributions
for dname in dir(stats):
    dist = getattr(stats, dname)
    if not isinstance(dist, (stats.rv_continuous, stats.rv_discrete)):
        continue
    is_discrete = isinstance(dist, stats.rv_discrete)

    # Construct appropriate param spec for this distribution.
    # Discrete distributions don't have a scale parameter.
    # We'll assume shape parameters are positive and have default 0...
    # TODO: this can't always be true!
    spec = list(hypney.RATE_LOC_PARAMS if is_discrete else hypney.RATE_LOC_SCALE_PARAMS)
    if dist.shapes:
        for pname in dist.shapes.split(", "):
            spec.append(
                hypney.ParameterSpec(name=pname, min=0, max=float("inf"), default=0)
            )

    # Create the new class
    dname = dname.capitalize()
    locals()[dname] = dist_class = type(dname, (ScipyUnivariate,), dict())
    dist_class.dist = dist
    dist_class.param_specs = tuple(spec)
    if is_discrete:
        dist_class.observables = (
            hypney.Observable(
                name=hypney.DEFAULT_OBSERVABLE.name,
                min=-float("inf"),
                max=float("inf"),
                integer=True,
            ),
        )
    export(dist_class)
