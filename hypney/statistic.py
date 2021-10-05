from copy import copy
import functools
import warnings

import eagerpy as ep
import numpy as np

import hypney
from hypney import NotChanged

export, __all__ = hypney.exporter()


@export
class Statistic:
    model: hypney.Model  # Model of the data
    dist: hypney.Model = None  # Model of the statistic; takes same parameters

    ##
    # Initialization
    ##

    def __init__(
        self,
        model: hypney.Model,
        data=NotChanged,
        params=NotChanged,
        dist=None,
        **kwargs,
    ):
        self.model = model
        self._set_dist(dist)
        if data is NotChanged:
            # Do not bypass _set_data; if the model has data,
            # we'll want to run _init_data on it
            data = self.model.data
        self._set_data(data)
        self._set_defaults(params, **kwargs)

    def _set_dist(self, dist: hypney.Model):
        if dist is NotChanged:
            return

        if dist is None:
            if hasattr(self, "_build_dist"):
                # Build a distribution automatically
                dist = self._build_dist()
                transform_params = self._dist_params
        else:
            # Use the user's distribution, but make it take all the model's params,
            # ignoring params the dist does not depend on.
            transform_params = functools.partial(
                _filter_params, allowed_names=dist.param_names
            )

        if dist is not None:
            # Make the dist take the same parameters as the model,
            # even if it depends on fewer / different parameters.
            param_specs = []
            for p in self.model.param_specs:
                # Set anchors from dist if available, copy the rest of the param
                # spec -- especially defaults! -- from model
                new_spec = p
                if p.name in dist.param_names:
                    dist_spec = dist.param_spec_for(p.name)
                    if dist_spec.anchors and not p.anchors:
                        new_spec = p._replace(anchors=dist_spec.anchors)
                param_specs.append(new_spec)

            self.dist = dist.reparametrize(
                transform_params=transform_params, param_specs=param_specs,
            )

        # if dist is None and we can't build_dist, leave dist None;
        # some estimators will complain.

    def _set_data(self, data):
        if data is NotChanged:
            return
        self.model = self.model(data=data)
        if self.model.data is not None:
            self._init_data()

    def _init_data(self):
        """Initialize self.data (either from construction or data change)"""
        pass

    @property
    def data(self) -> ep.Tensor:
        return self.model.data

    def _set_defaults(self, params=NotChanged, **kwargs):
        self.model = self.model(params=params, **kwargs)

    def set(self, data=NotChanged, dist=NotChanged, params=NotChanged, **kwargs):
        """Return a statistic with possibly changed data or distribution"""
        if (
            data is NotChanged
            and dist is NotChanged
            and params is NotChanged
            and not kwargs
        ):
            return self
        new_self = copy(self)
        new_self._set_defaults(params, **kwargs)
        new_self._set_dist(dist)
        new_self._set_data(data)
        return new_self

    ##
    # Computation
    ##

    def __call__(self, data=NotChanged, dist=NotChanged, params=NotChanged, **kwargs):
        return self.set(data=data, params=params, **kwargs)

    def compute(self, data=NotChanged, params: dict = None, **kwargs) -> ep.TensorType:
        self = self.set(data=data)
        return self.model._scalar_method(self._compute, params=params, **kwargs)

    def _compute(self, params):
        raise NotImplementedError

    ##
    # Simulation
    ##

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
        # Set defaults once to avoid re-validation
        self = self.set(params=params, **kwargs)

        results = np.zeros(size)
        for i in range(size):
            sim_data = transform(self.model._simulate(params=self.model.defaults))
            try:
                results[i] = self.compute(data=sim_data)
            except Exception as e:
                warnings.warn(f"Exception during test statistic evaluation: {e}")
                results[i] = float("nan")
        return results

    ##
    # Distribution
    ##

    def _dist_params(self, params):
        """Return distribution params given model params"""
        # Default assumption is that distribution is parameter-free
        return dict()

    def dist_from_toys(self, params=NotChanged, n_toys=1000, transform=np.asarray,
                       options=None, **kwargs):
        """Return an estimated distribution of the statistic given params
        from running simulations.

        """
        if options is None:
            options = dict()
        # Use a *lot* of bins by default, since we're most interested
        # in the cdf/ppf
        options.setdefault("bin_count_multiplier", 10)

        # Set defaults before simulation; helps provide e.g. better minimizer guesses
        self = self.set(params=params, **kwargs)
        print(self, params, kwargs, self.model.defaults)
        toys = self.rvs(n_toys, transform=transform)
        print("dist_from_toys called with ", params, kwargs, " produced average toy ", toys.mean())


        dist = hypney.models.from_samples(toys, **options)
        # Remove standard loc/scale/rate params
        # to avoid confusion with model parameters
        return dist.freeze()

    def interpolate_dist_from_toys(
        self, anchors: dict, progress=True, methods="ppf", **kwargs
    ):
        assert isinstance(anchors, dict), "Pass a dict of sequences as anchors"
        return hypney.models.Interpolation(
            functools.partial(self.dist_from_toys, **kwargs),
            anchors,
            progress=progress,
            methods=methods,
        )


def _filter_params(params, allowed_names):
    # just because pickle doesn't like lambdas...
    return {name: params[name] for name in allowed_names}
