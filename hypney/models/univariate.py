import inspect

import eagerpy as ep
from eagerpy.astensor import _get_module_name
import numpy as np
from scipy import stats as scipy_stats

import hypney
from hypney import DEFAULT_OBSERVABLE

export, __all__ = hypney.exporter()


class UnivariateDistribution(hypney.Model):
    scipy_name: str = None
    # jax name is assumed equal to scipy name
    tfp_name: str = None
    torch_name: str = None

    param_specs = hypney.RATE_LOC_SCALE_PARAMS
    observables = (DEFAULT_OBSERVABLE,)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(self, "scipy_dist"):
            # Needed for rv_histogram
            self._dists = dict(scipy=self.scipy_dist)
        else:
            # Use the distribution name. This is more flexible; scipy might add
            # distributions across versions.
            self._dists = dict(scipy=getattr(scipy_stats, self.scipy_name, None))

    def _dist_params(self, params):
        # Extract raw tensor, and remove rate parameter
        # (scipy dists have no such concept)
        return {
            k: v.raw for k, v in params.items() if k != hypney.DEFAULT_RATE_PARAM.name
        }

    def dist_for_data(self):
        """Return distribution from library appropriate to self.data"""
        if isinstance(self.data, ep.NumPyTensor) or self.data is None:
            return self._dists["scipy"]

        modname = _get_module_name(self.data.raw)
        if modname.startswith("jax"):
            modname = "jax"

        if modname in self._dists:
            return self._dists[modname]

        if modname == "jax":
            if not hasattr(ep.jax.scipy.stats, self.scipy_name):
                raise NotImplementedError(f"{self.distname} not implemented in JAX")
            result = getattr(ep.jax.scipy.stats, self.distname)

        elif modname == "torch":
            import torch  # noqa

            if not self.torch_name or not hasattr(torch.distributions, self.torch_name):
                raise NotImplementedError(
                    f"{self.__class__.__name__} distribution not available in PyTorch"
                )
            result = TorchDistributionWrapper(
                getattr(torch.distributions, self.torch_name),
                defaults=self.defaults,
                transform=self.torch_param_transform,
            )

        elif modname == "tensorflow":
            import tensorflow_probability as tfp  # noqa

            if not self.tfp_name or not hasattr(tfp.distributions, self.tfp_name):
                raise NotImplementedError(
                    f"{self.__class__.__name__} distribution not available in TensorFlow Probability"
                )
            result = TFPDistributionWrapper(
                getattr(tfp.distributions, self.tfp_name),
                defaults=self.defaults,
                transform=self.tf_param_transform,
            )

        else:
            raise RuntimeError(f"Unkown tensor from module {modname}?!")

        self._dists[modname] = result
        return result

    def torch_param_transform(self, params):
        return params

    def tf_param_transform(self, params):
        return self.torch_param_transform(params)

    # Methods using data / quantiles

    def _rvs(self, size: int, params: dict) -> ep.TensorType:
        # Simulation requires the scipy dist
        if not "scipy" in self._dists:
            raise NotImplementedError(
                f"This version of scipy does not have {self.scipy_name}"
            )
        return self._dists["scipy"].rvs(size=size, **self._dist_params(params))[
            ..., None
        ]

    def _logpdf(self, params: dict) -> ep.TensorType:
        dist = self.dist_for_data()
        logpdf = dist.logpdf if hasattr(dist, "logpdf") else dist.logpmf
        return ep.astensor(logpdf(self.data[..., 0].raw, **self._dist_params(params)))

    def _pdf(self, params: dict) -> ep.TensorType:
        dist = self.dist_for_data()
        pdf = dist.pdf if hasattr(dist, "pdf") else dist.pmf
        return ep.astensor(pdf(self.data[..., 0].raw, **self._dist_params(params)))

    def _cdf(self, params: dict) -> np.ndarray:
        return ep.astensor(
            self.dist_for_data().cdf(self.data[..., 0].raw, **self._dist_params(params))
        )

    def _ppf(self, params: dict) -> np.ndarray:
        return ep.astensor(
            self.dist_for_data().ppf(self.quantiles.raw, **self._dist_params(params))
        )

    # Methods not using data

    def _mean(self, params):
        return self._to_tensor(self.dist_for_data().mean(**self._dist_params(params)))

    def _std(self, params):
        return self._to_tensor(self.dist_for_data().std(**self._dist_params(params)))


class UnivariateDiscreteDistribution(UnivariateDistribution):

    observables = (DEFAULT_OBSERVABLE._replace(integer=True),)


##
# Tensorflow / Pytorch support
##


class TorchDistributionWrapper:
    """Wrap torch / tf_probability distributions in a scipy-stats like API
    including data scaling for distributions not taking loc and scale
    """

    def __init__(self, dist, defaults, transform):
        self.dist = dist
        self.defaults = transform(defaults)
        self.transform = transform

        signature_params = inspect.signature(dist).parameters
        self.patch_loc = ("loc" in self.defaults) and ("loc" not in signature_params)
        self.patch_scale = ("scale" in self.defaults) and (
            "scale" not in signature_params
        )

    def _patch_params(self, params, data_tensor):
        params = self.transform(params)

        # Model ensured params were all tensors of the same shape
        if not params:
            param_shape = tuple()
        else:
            param_shape = params[list(params.keys())[0]].shape

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

    def logpdf(self, data, **params):
        import torch

        params, x0, x_scale = self._patch_params(params, data)
        return (
            self.dist(**params).log_prob((data - x0) / x_scale)
            - torch.tensor(x_scale).log()
        )

    def pdf(self, data, **params):
        params, x0, x_scale = self._patch_params(params, data)
        return self.dist(**params).log_prob((data - x0) / x_scale).exp() / x_scale

    def cdf(self, data, **params):
        params, x0, x_scale = self._patch_params(params)
        return self.dist(**params).cdf((data - x0) / x_scale)

    # TODO: test these!

    def ppf(self, quantiles, **params):
        params, x0, x_scale = self._patch_params(params)
        result = self.dist(**params).icdf(quantiles)
        return result * x_scale + x0

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

    def logpdf(self, data, **params):
        import tensorflow as tf

        params, x0, x_scale = self._patch_params(params, data)
        return self.dist(**params).log_prob((data - x0) / x_scale) - tf.log(x_scale)

    def pdf(self, data, **params):
        params, x0, x_scale = self._patch_params(params, data)
        return self.dist(**params).prob((data - x0) / x_scale) / x_scale

    # TODO: test these!

    def ppf(self, quantiles, **params):
        return self.icdf(quantiles, **params)

    def mean(self, **params):
        return self.dist(**params).mean()

    def std(self, **params):
        return self.dist(**params).stddev()
