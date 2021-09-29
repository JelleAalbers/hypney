from copy import copy
import functools

import eagerpy as ep
import numpy as np

import hypney
from hypney import NotChanged

export, __all__ = hypney.exporter()


@export
class Statistic:
    model: hypney.Model  # Model of the data
    dist: hypney.Model = None  # Model of the statistic; takes same parameters

    def __init__(self, model: hypney.Model, data=hypney.NotChanged, dist=None):
        self.model = model
        self._set_dist(dist)
        self._set_data(data)

    def _set_dist(self, dist: hypney.Model):
        if dist is hypney.NotChanged:
            return
        if dist is None:
            if hasattr(self, "_build_dist"):
                # Build a distribution automatically
                self.dist = hypney.models.Reparametrized(
                    self._build_dist(),
                    transform_params=self._dist_params,
                    param_specs=self.model.param_specs,
                )
        else:
            # Use the distribution passed by the user.
            # Make it take the model params, then ignore ones it does not depend on.
            filter_params = functools.partial(
                _filter_params, allowed_names=dist.param_names
            )
            self.dist = hypney.models.Reparametrized(
                dist,
                transform_params=filter_params,
                param_specs=self.model.param_specs,
            )

    def _set_data(self, data):
        if data is not hypney.NotChanged:
            self.model = self.model(data=data)
        # self.data is just self.model.data, see below
        if self.data is not None:
            self._init_data()

    @property
    def data(self) -> ep.Tensor:
        return self.model.data

    def validate_data(self, data):
        return self.model.validate_data(data)

    def _init_data(self):
        """Initialize self.data (either from construction or data change)"""
        pass

    def set(self, data=NotChanged, dist=NotChanged):
        """Return a statistic with possibly changed data or distribution"""
        if data is NotChanged and dist is NotChanged:
            return self
        new_self = copy(self)
        new_self._set_data(data)
        new_self._set_dist(dist)
        return new_self

    def __call__(self, data=NotChanged, params: dict = None, **kwargs):
        return self.compute(data=data, params=params, **kwargs)

    def compute(self, data=NotChanged, params: dict = None, **kwargs):
        return self.compute_(data=data, params=params, **kwargs).raw

    def compute_(self, data=NotChanged, params: dict = None, **kwargs) -> ep.TensorType:
        self = self.set(data)
        if self.data is None:
            raise ValueError("Must provide data")

        params = self.model.validate_params(params, **kwargs)
        return self._compute(params)

    def _compute(self, params):
        raise NotImplementedError

    def rvs(self, size=1, params=None, transform=np.asarray, **kwargs) -> np.ndarray:
        """Return statistic evaluated on simulated data,
        generated from model with params

        Args:
         - size: number of toys to draw
         - params, **kwargs: parameters at which to simulate toys
         - transform: run numpy data through this function before passing
            it to statistic. Useful to convert to an autograd library,
            e.g. torch.from_numpy / tf.convert_to_tensor.
        """
        params = self.model.validate_params(params, **kwargs)
        results = np.zeros(size)
        for i in range(size):
            sim_data = transform(self.model._simulate(params=params))
            results[i] = self.compute(data=sim_data, params=params)
        return results

    ## Distribution

    def _dist_params(self, params):
        """Return distribution params given model params"""
        # Default assumption is that distribution is parameter-free
        return dict()

    def dist_from_toys(self, params=None, n_toys=1000, transform=np.asarray, **kwargs):
        """Return an estimated distribution of the statistic given params
        from running simulations.

        Note: kwargs are passed to hypney.models.from_samples, pass params as dict!
        """
        # Use a *lot* of bins by default, since we're most interested
        # in the cdf/ppf
        kwargs.setdefault("bin_count_multiplier", 10)
        toys = transform(self.rvs(n_toys, params=params))
        dist = hypney.models.from_samples(toys, **kwargs)
        # Remove standard loc/scale/rate params
        # to avoid confusion with model parameters
        return dist.freeze()

    def interpolate_dist_from_toys(self, anchors: dict, progress=True, **kwargs):
        assert isinstance(anchors, dict), "Pass a dict of sequences as anchors"
        return hypney.models.Interpolation(
            functools.partial(self.dist_from_toys, **kwargs),
            anchors,
            progress=progress,
        )


@export
class IndependentStatistic(Statistic):
    """Statistic depending only on data, not on any parameters.

    The distribution may still depend on parameters.

    For speed, the result will be precomputed as soon as data is known.
    """

    def _init_data(self):
        # Precompute result on the data.
        self._result = self._compute()
        super()._init_data()

    def compute(self, params: dict = None, data=NotChanged):
        self = self.set(data)
        if self.data is None:
            raise ValueError("Must provide data")
        return self._result

    def _compute(self):
        raise NotImplementedError


def _filter_params(params, allowed_names):
    # just because pickle doesn't like lambdas...
    return {name: params[name] for name in allowed_names}
