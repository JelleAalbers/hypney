from copy import copy
from concurrent.futures import ProcessPoolExecutor
import functools
import gzip
import os
from pathlib import Path
import pickle
import warnings

import eagerpy as ep
import numpy as np

import hypney
from hypney import NotChanged

export, __all__ = hypney.exporter()


@export
class Statistic:
    model: hypney.Model  # Model of the data
    _dist: hypney.Model = None  # Model of the statistic; takes same parameters

    @property
    def dist(self):
        # Just so people won't assign it by accident...
        # pass dist=... in __init__ instead.
        return self._dist

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
                # Statistic has a default distribution
                dist = self._build_dist()
            else:
                # Leave self.dist at None (some estimators will complain)
                assert self._dist is None
                return
        if isinstance(dist, (str, Path)):
            # Load distribution from a pickle
            _open = gzip.open if str(dist).endswith(".gz") else open
            with _open(dist) as f:
                dist = pickle.load(f)

        if self._has_redefined("_dist_params"):
            # For some statistics (e.g. count), distributions take different
            # parameters than the model, specified by _dist_params.
            # Thus, wrap dist in an appropriate reparametrization.
            # Unfortunately, there is no easy way to preserve any anchors...
            dist = dist.reparametrize(
                transform_params=self._dist_params, param_specs=self.model.param_specs,
            )

        # Ensure dist has same defaults as models
        self._dist = dist(params=self.model.defaults)

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

    def _has_redefined(self, method_name, from_base=None):
        """Returns if method_name is redefined from Statistic.method_name"""
        if from_base is None:
            from_base = Statistic
        f = getattr(self, method_name)
        if not hasattr(f, "__func__"):
            return True
        return f.__func__ is not getattr(from_base, method_name)

    ##
    # Computation
    ##

    def __call__(self, data=NotChanged, dist=NotChanged, params=NotChanged, **kwargs):
        return self.set(data=data, params=params, **kwargs)

    def compute(self, data=NotChanged, params: dict = None, **kwargs) -> ep.TensorType:
        self = self.set(data=data)
        if self.data is None:
            raise ValueError("Data must be set first")
        return self.model._scalar_method(self._compute, params=params, **kwargs)

    def _compute(self, params):
        # data has shape ([n_datasets?], n_events, n_observables)
        # params have shape ([batch_shape], 1)
        # result has to be shape ([n_datasets?], [batch_shape], 1)
        raise NotImplementedError

    ##
    # Simulation
    ##

    def rvs(
        self,
        size=1,
        params=NotChanged,
        transform=np.asarray,
        nan_on_exception=False,
        simulate_from_model=None,
        **kwargs,
    ) -> np.ndarray:
        """Return statistic evaluated on simulated data,
        generated from model with params

        Args:
         - size: number of toys to draw
         - params, **kwargs: parameters at which to simulate toys
         - transform: run numpy data through this function before passing
            it to statistic. Useful to convert to an autograd library,
            e.g. torch.from_numpy / tf.convert_to_tensor.
         - simulate_from_model: alternate model to simulate data from,
            if different from the model the statistic uses for evaluation.
         - pass_params_to_sim_model: If True, pass any params and **kwargs
            also to the model given in simulate_from_model.
        """
        if simulate_from_model is None:
            simulate_from_model = self.model

        # Set defaults once to avoid re-validation
        self = self.set(params=params, **kwargs)

        results = np.zeros(size)
        for i in range(size):
            sim_data = transform(simulate_from_model.simulate())
            if nan_on_exception:
                try:
                    results[i] = self.compute(data=sim_data)
                except Exception as e:
                    warnings.warn(f"Exception during test statistic evaluation: {e}")
                    results[i] = float("nan")
            else:
                results[i] = self.compute(data=sim_data)
        return results

    ##
    # Distribution
    ##

    def _dist_params(self, params):
        """Return distribution params given model params"""
        return params

    def dist_from_toys(
        self,
        params=NotChanged,
        n_toys=1000,
        transform=np.asarray,
        options=None,
        nan_on_exception=False,
        simulate_from_model=None,
        pass_params_to_sim_model=False,
        **kwargs,
    ):
        """Return an estimated distribution of the statistic given params
        from running simulations.

        """
        if options is None:
            options = dict()

        # Set defaults before simulation; helps provide e.g. better minimizer guesses
        self = self.set(params=params, **kwargs)

        if simulate_from_model is None:
            simulate_from_model = self.model
        if pass_params_to_sim_model:
            simulate_from_model = simulate_from_model(params=params, **kwargs)

        toys = self.rvs(
            n_toys,
            transform=transform,
            nan_on_exception=nan_on_exception,
            simulate_from_model=simulate_from_model,
        )

        dist = hypney.models.ppf_from_samples(toys, **options)
        # Remove all parameters (to avoid confusion with model parameters)
        return dist.freeze()

    def interpolate_dist_from_toys(
        self, anchors: dict, progress=True, methods="ppf", map=map, **kwargs,
    ):
        """Estimate this statistic's distribution by Monte Carlo.

        This draws toys at a grid specified by the anchors.
        By default, we then interpolate the ppf, since this is what you need
        for confidence interval setting.

        Arguments:
         - anchors: dict {param_name: anchor_values} of parameter values at
            which to run dist_from_toys. For multiple parameter, a grid is built
         - map: map function for task distribution. Can use e.g.
            ProcessPoolExecutor.map instead of the default map.
         - progress: whether to show a progress bar
         - methods: which distribution methods to interpolate
         - simulate_from_model: alternate model to simulate toys from.
            This will be passed the anchor parameters.

        """
        assert isinstance(anchors, dict), "Pass a dict of sequences as anchors"

        # Without this, simulate_from_model would be the same at each anchor.
        # Can't image why someone would want that, but allow it to be overriden
        # anyway...
        kwargs.setdefault("pass_params_to_sim_model", True)

        if not self._has_redefined("_dist_params"):
            model_builder = functools.partial(self.dist_from_toys, **kwargs)
            anchors = anchors
        else:
            # Build a distribution that takes the _dist_params
            # rather than the model's params.

            # Compute anchors transformed by self._dist_params
            if len(anchors) > 1:
                raise NotImplementedError(
                    "Multi-parameter interpolation not supported if _dist_params is nontrivial"
                )
                # (Since we'd have to transform the whole grid of anchors.
                #  Even if the transformation is simple enough to allow this,
                #  we don't have that grid here yet)
            # (back and forth to tensor necessary to support dist_params that
            #  do calls -- e.g. Count's dist calls to model._rate)
            param_tensors = {k: self.model._to_tensor(v) for k, v in anchors.items()}
            dist_anchors = self._dist_params(param_tensors)
            dist_anchors = {k: v.numpy().tolist() for k, v in dist_anchors.items()}

            # Set up transformation dictionary
            # from new (dist) to old (model) anchors
            dist_pname = list(dist_anchors.keys())[0]
            model_pname = list(anchors.keys())[0]
            dist_to_model_anchor = dict(
                zip(dist_anchors[dist_pname], tuple(anchors[model_pname]))
            )

            # The interpolator will accept the new (dist) anchors
            # To know which model to place where, model_builder must transform
            # from the new (dist) anchors back to the old (model) anchors.
            # (Cannot inline _transformed_model_builder, would break pickle.)
            anchors = dist_anchors
            model_builder = functools.partial(
                _transformed_model_builder,
                self=self,
                model_pname=model_pname,
                dist_pname=dist_pname,
                dist_to_model_anchor=dist_to_model_anchor,
                **kwargs,
            )

        return hypney.models.Interpolation(
            model_builder, anchors, progress=progress, map=map, methods=methods,
        ).fix_except(anchors.keys())

    def with_stored_dist(
        self,
        dist_filename,
        n_toys=None,
        rate_anchors=hypney.DEFAULT_RATE_GRID,
        max_workers=None,
        build_if_not_found=True,
        dist_dir="cached_dists",
        **kwargs,
    ):
        """Return statistic with distribution loaded from cache_dir,
        or rebuilt from toy mc if file does not exist

        TODO: describe rate anchors or generalize
        """
        if n_toys is None:
            n_toys = 10_000
        if max_workers is None:
            max_workers = min(32, os.cpu_count() - 4)
        rate_anchors = list(rate_anchors)

        dist_dir = Path(f"./{dist_dir}/")
        dist_dir.mkdir(exist_ok=True)
        dist_filename = dist_dir / f"{dist_filename}_{n_toys}.pkl.gz"

        if dist_filename.exists():
            with gzip.open(dist_filename) as f:
                return self.set(dist=pickle.load(f))

        elif not build_if_not_found:
            raise FileNotFoundError(f"No cached distribution at {dist_filename}")

        else:
            mu_min, mu_max = [f(rate_anchors) for f in (min, max)]
            print(
                f"Building distribution {dist_filename}, {n_toys} toys,"
                f"mu in [{mu_min}, {mu_max}]"
            )
            if max_workers > 1:
                with ProcessPoolExecutor(max_workers=max_workers) as exc:
                    dist = self.interpolate_dist_from_toys(
                        anchors=dict(rate=rate_anchors),
                        n_toys=n_toys,
                        map=exc.map,
                        **kwargs,
                    )
            else:
                dist = self.interpolate_dist_from_toys(
                    anchors=dict(rate=rate_anchors), n_toys=n_toys, **kwargs
                )
            with gzip.open(dist_filename, mode="wb") as f:
                pickle.dump(dist, f)
            return self.set(dist=dist)


def _transformed_model_builder(
    dist_params, *, self, model_pname, dist_pname, dist_to_model_anchor, **kwargs
):
    model_params = {model_pname: dist_to_model_anchor[dist_params[dist_pname]]}
    return self.dist_from_toys(params=model_params, **kwargs)
