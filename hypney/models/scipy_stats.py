import typing as ty

import numpy as np
from scipy import stats

import hypney

export, __all__ = hypney.exporter()


standard_param_specs = (
    hypney.DEFAULT_RATE_PARAM,
    hypney.ParameterSpec(name="loc", min=-float("inf"), max=float("inf"), default=0),
    hypney.ParameterSpec(name="scale", min=0, max=float("inf"), default=1),
)


class ScipyUnivariate(hypney.Model):
    dist: ty.Union[stats.rv_continuous, stats.rv_discrete]
    param_specs = standard_param_specs

    def dist_params(self, params):
        return {k: v for k, v in params.items() if k != hypney.DEFAULT_RATE_PARAM.name}

    def rvs(self, n: int, params: dict = None) -> np.ndarray:
        params = self.validate_params(params)
        return self.dist.rvs(size=n, **self.dist_params(params))[:, None]

    def _pdf(self, params: dict = None) -> np.ndarray:
        pdf = self.dist.pdf if hasattr(self.dist, "pdf") else self.dist.pmf
        return pdf(self.data[:, 0], **self.dist_params(params))

    def _cdf(self, params: dict = None) -> np.ndarray:
        return self.dist.cdf(self.data[:, 0], **self.dist_params(params))


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

        self.dist = stats.rv_histogram((histogram, bin_edges),)
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
    spec = list(standard_param_specs[:2] if is_discrete else standard_param_specs)
    if dist.shapes:
        for pname in dist.shapes.split(", "):
            spec.append(
                hypney.ParameterSpec(name=pname, min=0, max=float("inf"), default=0)
            )

            standard_param_specs[:2]

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
