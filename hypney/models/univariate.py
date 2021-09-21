import inspect
import typing as ty

import eagerpy as ep
from eagerpy.astensor import _get_module_name
import numpy as np
from scipy import stats

import hypney

export, __all__ = hypney.exporter()

##
# Univariate scipy.stats (and jax.scipy.stats) distributions
##


class ScipyUnivariate(hypney.Model):
    scipy_dist: ty.Union[stats.rv_continuous, stats.rv_discrete]
    param_specs = hypney.RATE_LOC_SCALE_PARAMS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dists = dict(scipy=self.scipy_dist)

    def _dist_params(self, params):
        return {k: v for k, v in params.items() if k != hypney.DEFAULT_RATE_PARAM.name}

    @property
    def distname(self):
        return self.scipy_dist.name

    def dist(self):
        """Return distribution from library appropriate to self.data"""
        if isinstance(self.data, ep.NumPyTensor) or self.data is None:
            return self.scipy_dist

        modname = _get_module_name(self.data.raw)
        if modname.startswith("jax"):
            modname = "jax"

        if modname in self._dists:
            return self._dists[modname]

        if modname == "jax":
            if not hasattr(ep.jax.scipy.stats, self.distname):
                raise NotImplementedError(f"{self.distname} not implemented in JAX")
            result = getattr(ep.jax.scipy.stats, self.distname)

        elif modname == "torch":
            import torch  # noqa

            dname = TORCH_DISTRIBUTION_NAMES.get(
                self.distname, self.distname.capitalize()
            )
            if not hasattr(torch.distributions, dname):
                raise NotImplementedError(
                    f"{dname} not implemented in torch.distributions"
                )
            result = TorchDistributionWrapper(
                getattr(torch.distributions, dname), self.defaults
            )

        elif modname == "tensorflow":
            import tensorflow_probability as tfp  # noqa

            dname = TFP_DISTRIBUTION_NAMES.get(
                self.distname, self.distname.capitalize()
            )
            if not hasattr(tfp.distributions, dname):
                raise NotImplementedError(
                    f"{dname} not implemented in tensorflow_probability"
                )
            result = TFPDistributionWrapper(
                getattr(tfp.distributions, dname), self.defaults
            )

        else:
            raise ValueError(f"Unkown tensor from module {modname}")

        self._dists[modname] = result
        return result

    # Methods using data

    def _rvs(self, size: int, params: dict) -> ep.TensorType:
        return self.scipy_dist.rvs(size=size, **self._dist_params(params))[:, None]

    def _pdf(self, params: dict) -> ep.TensorType:
        dist = self.dist()
        pdf = dist.pdf if hasattr(dist, "pdf") else dist.pmf
        return ep.astensor(pdf(self.data[:, 0].raw, **self._dist_params(params)))

    def _cdf(self, params: dict) -> np.ndarray:
        return ep.astensor(
            self.dist().cdf(self.data[:, 0].raw, **self._dist_params(params))
        )

    # Methods not using data

    def _mean(self, params):
        return self.dist().mean(**self._dist_params(params))

    def _std(self, params):
        return self.dist().std(**self._dist_params(params))


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
    dist_class.scipy_dist = dist
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

        self.scipy_dist = stats.rv_histogram((histogram, bin_edges),)
        super().__init__(*args, **kwargs)


##
# Tensorflow / Pytorch support
##

# TODO: We must check that the parametrizations of the distributions
# are consistent between scipy, jax, torch, tf...
# Cannot just assume this; it will silently corrupt users' results if not!

TORCH_DISTRIBUTION_NAMES = dict(
    # Distributions with different names in torch.distributions
    # Several other distributions are also supported
    # but their names differ only by capitalization (cauchy, chi2, ...)
    norm="Normal",
    epon="Exponential",
    f="FisherSnedecor",
    gumbel_r="Gumbel",
    halfcauchy="HalfCauchy",
    halfnorm="HalfNormal",
    lognorm="LogNormal",
    t="StudentT",
    vonmises="VonMises",
    weibull_min="Weibull",
    binom="Binomial",
    geom="Geometric",
    nbinom="NegativeBinomial",
)


TFP_DISTRIBUTION_NAMES = dict(
    norm="Normal",
    epon="Exponential",
    gumbel_r="Gumbel",
    halfcauchy="HalfCauchy",
    halfnorm="HalfNormal",
    invgamma="InverseGamma",
    lognorm="LogNormal",
    t="StudentT",
    # triang='Triangular',   # different parametrization
    # truncnorm='TruncatedNormal',   # different parametrization?
    vonmises="VonMises",
    weibull_min="Weibull",
    betabinom="BetaBinomial",
    binom="Binomial",
    geom="Geometric",
    nbinom="NegativeBinomial",
)


class TorchDistributionWrapper:
    """Wrap torch / tf_probability distributions in a scipy-stats like API
    including data scaling for distributions not taking loc and scale
    """

    def __init__(self, dist, default_params):
        self.dist = dist
        self.default_params = default_params

        sig_params = inspect.signature(dist).parameters
        self.patch_loc = ("loc" in default_params) and ("loc" not in sig_params)
        self.patch_scale = ("scale" in default_params) and ("scale" not in sig_params)

    def _patch_params(self, params, data_tensor):
        if self.patch_loc:
            x0 = params["loc"]
            params = {k: v for k, v in params.items() if k != "loc"}
        else:
            x0 = 0

        if self.patch_scale:
            x_scale = params["scale"]
            params = {k: v for k, v in params.items() if k != "scale"}
        else:
            x_scale = 1

        params = self._patch_param_dtypes(params, data_tensor)
        return params, x0, x_scale

    @staticmethod
    def _patch_param_dtypes(params, data_tensor):
        # For pytorch, casting param values to tensors explicitly
        # seems not to be needed. It doesn't improve speed either.
        return params

    def pdf(self, data, **params):
        params, x0, x_scale = self._patch_params(params, data)
        return self.dist(**params).log_prob((data - x0) / x_scale).exp() / x_scale

    def cdf(self, data, **params):
        params, x0, x_scale = self._patch_params(params)
        return self.dist(**params).cdf((data - x0) / x_scale)

    # TODO: test these!

    def mean(self, **params):
        return self.dist(**params).mean

    def std(self, **params):
        return self.dist(**params).stddev


class TFPDistributionWrapper(TorchDistributionWrapper):
    @staticmethod
    def _patch_param_dtypes(params, data_tensor):
        # tfp casts parameters to float32 by default, which gives an error
        # if the data is a float64 tensor. Hence we must cast the params
        # explicitly to the right type of tensor.
        import tensorflow as tf

        return {
            k: tf.convert_to_tensor(v, dtype=data_tensor.dtype)
            for k, v in params.items()
        }

    def pdf(self, data, **params):
        params, x0, x_scale = self._patch_params(params, data)
        return self.dist(**params).prob((data - x0) / x_scale) / x_scale

    # TODO: test these!

    def mean(self, **params):
        return self.dist(**params).mean()

    def std(self, **params):
        return self.dist(**params).stddev()
