import hypney

import numpy as np
from scipy import stats

export, __all__ = hypney.exporter()


class ScipyUnivariate(hypney.Model):
    def dist_params(self, params):
        return {k: v for k, v in params.items() if k != hypney.DEFAULT_RATE_PARAM.name}

    def simulate_n(self, n: int, params: dict = None) -> np.ndarray:
        params = self.validate_params(params)
        return self.dist.rvs(size=n, **self.dist_params(params))[:, None]

    def pdf(self, data: np.ndarray, params: dict = None) -> np.ndarray:
        params = self.validate_params(params)
        data = self.validate_data(data)
        return self.dist.pdf(data[:, 0], **self.dist_params(params))

    def cdf(self, data: np.ndarray, params: dict = None) -> np.ndarray:
        params = self.validate_params(params)
        data = self.validate_data(data)
        return self.dist.cdf(data[:, 0], **self.dist_params(params))


# Create classes for all distributions
for dname in dir(stats):
    dist = getattr(stats, dname)
    if not isinstance(dist, stats.rv_continuous):
        continue

    # Construct appropriate param spec for this distribution.
    # Assume shape parameters are positive and have default 0...
    spec = [
        hypney.DEFAULT_RATE_PARAM,
        hypney.ParameterSpec(
            name="loc", min=-float("inf"), max=float("inf"), default=0
        ),
        hypney.ParameterSpec(name="scale", min=0, max=float("inf"), default=1),
    ]
    if dist.shapes:
        for pname in dist.shapes.split(", "):
            spec.append(
                hypney.ParameterSpec(name=pname, min=0, max=float("inf"), default=0)
            )

    dname = dname.capitalize()
    locals()[dname] = dist_class = type(dname, (ScipyUnivariate,), dict())
    dist_class.dist = dist
    dist_class.param_specs = tuple(spec)
    export(dist_class)
