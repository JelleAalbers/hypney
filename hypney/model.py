from copy import copy
import itertools
import functools
import typing as ty

import eagerpy as ep
import numpy as np

import hypney
from hypney import NotChanged
import hypney.utils.eagerpy as ep_util

export, __all__ = hypney.exporter()


@export
class Model:
    name: str = ""
    param_specs: ty.Tuple[hypney.Parameter] = (hypney.DEFAULT_RATE_PARAM,)
    observables: ty.Tuple[hypney.Observable] = (hypney.DEFAULT_OBSERVABLE,)
    data: ep.Tensor = None
    quantiles: ep.Tensor = None
    _backend_name = None

    _data_is_single_scalar = False
    _quantiles_is_single_scalar = False

    def param_spec_for(self, pname):
        for p in self.param_specs:
            if p.name == pname:
                return p
        raise ValueError(f"Unknown parameter {pname}")

    @property
    def n_dim(self):
        return len(self.observables)

    @property
    def param_names(self):
        return tuple([p.name for p in self.param_specs])

    @property
    def defaults(self):
        return {p.name: p.default for p in self.param_specs}

    @property
    def sample_shape(self):
        # Assumes event_shape is length-1
        if self.data is None:
            return (1,)
        return self.data.shape[:-1]

    @property
    def backend(self):
        if not self._backend_name:
            return None
        # Can't sore backend module, that's not picklable
        return getattr(ep, self._backend_name)

    ##
    # Initialization
    ##

    def __init__(
        self,
        *,
        name=NotChanged,
        data=None,
        params=NotChanged,  # Really defaults...
        param_specs=NotChanged,
        observables=NotChanged,
        quantiles=NotChanged,
        validate_defaults=True,
        backend=None,
        **kwargs,
    ):
        if name is not NotChanged:
            self.name = name

        # These have default class attributes
        if param_specs is not NotChanged:
            self.param_specs = param_specs
        if observables is not NotChanged:
            self.observables = observables

        if self._backend_name:
            # backend is already fixed
            if backend and backend != self._backend_name:
                raise ValueError(
                    f"backend is {self._backend_name}, cannot change to {backend}"
                )
        elif backend:
            self._backend_name = backend
        else:
            # Infer backend from data / quantiles. If we can't, it's numpy.
            for q in data, quantiles:
                try:
                    self._backend_name = ep_util.tensorlib(q).__name__
                    break
                except Exception:
                    continue
            else:
                self._backend_name = "numpy"

        self._set_data(data)
        self._set_defaults(params, validate_defaults=validate_defaults, **kwargs)
        self._set_quantiles(quantiles)

    def _set_data(self, data=hypney.NotChanged):
        if data is hypney.NotChanged:
            return
        if data is None:
            if self.data is not None:
                raise ValueError("Cannot reset data to None")
            else:
                self.data = None
                return
        data = self.validate_data(data)
        self.data = data
        self._init_data()

    def _init_data(self):
        """Initialize self.data (either from construction or data change)"""
        pass

    def _set_quantiles(self, quantiles=hypney.NotChanged):
        if quantiles is hypney.NotChanged:
            return
        if quantiles is None:
            if self.quantiles is not None:
                raise ValueError("Cannot reset quantiles to None")
            else:
                self.quantiles = None
                return
        quantiles = self.validate_quantiles(quantiles)
        self.quantiles = quantiles
        self._init_quantiles()

    def _init_quantiles(self):
        """Initialize self.quantiles (either from construction or data change)"""
        pass

    def _set_defaults(
        self, params: dict = NotChanged, validate_defaults=True, **kwargs
    ):
        if params is NotChanged:
            params = dict()
        if params is None:
            raise ValueError("None is not valid for params")
        if validate_defaults and not params:
            params = self.defaults
        if not params and not kwargs:
            return
        new_defaults = self.validate_params(params, **kwargs)
        self.param_specs = tuple(
            [p._replace(default=new_defaults[p.name]) for p in self.param_specs]
        )

    def _has_redefined(self, method_name, from_base=None):
        """Returns if method_name is redefined from Model.method_name"""
        if from_base is None:
            from_base = Model
        f = getattr(self, method_name)
        if not hasattr(f, "__func__"):
            return True
        return f.__func__ is not getattr(from_base, method_name)

    ##
    # Creating new models from old
    ##

    def __call__(self, **kwargs):
        return self.set(**kwargs)

    def set(
        self,
        *,
        name=NotChanged,
        data=NotChanged,
        quantiles=NotChanged,
        params=NotChanged,
        **kwargs,
    ):
        """Return a model with possibly changed name, defaults, data, or parameters"""
        if (
            name is NotChanged
            and data is NotChanged
            and quantiles is NotChanged
            and (params is NotChanged or not params)
            and not kwargs
        ):
            return self
        if params is None:
            raise ValueError("None is invalid as params...")

        new_self = copy(self)
        Model.__init__(
            new_self,
            name=name,
            data=data,
            quantiles=quantiles,
            # Only re-validate the defaults if new defaults were actually given
            validate_defaults=params != NotChanged and _merge_dicts(params, kwargs),
            params=params,
            **kwargs,
        )
        return new_self

    def mix_with(self, *others):
        return hypney.models.mixture(*((self,) + others))

    def __add__(self, other):
        return self.mix_with(other)

    def tensor_with(self, *others):
        return hypney.models.tensor_product(*((self,) + others))

    def __pow__(self, other):
        return self.tensor_with(other)

    def reparametrize(self, transform_params: callable, *args, **kwargs):
        return hypney.models.Reparametrized(
            self, transform_params=transform_params, *args, **kwargs
        )

    def cut(self, *args, **kwargs):
        """Return new model with observables cut to a rectangular region

        Args: left-right boundaries, specified in one of many legal ways.
        Just try some :-)
        """
        if args and kwargs:
            raise ValueError("Specify either keyword or position arguments")
        if args:
            if len(args) == 1:
                cut = args[0]
            else:
                cut = args
        if kwargs:
            cut = kwargs
        return hypney.models.CutModel(self, cut)

    def shift_and_scale(self, shift=0.0, scale=1):
        """Return model for data that has been shifted, then scaled,
        by constants.
        """
        return hypney.models.TransformedDataModel(self, shift=shift, scale=scale)

    def shift(self, shift=0.0):
        """Return model for data that has been shifted
        """
        return self.shift_and_scale(shift=shift, scale=1)

    def scale(self, scale=0.0):
        """Return model for data that has been scaled
        """
        return self.shift_and_scale(shift=0, scale=scale)

    def normalized_data(self):
        """Return model for data that was normalized using the current model's
        mean and standard deviation.
        """
        return self.shift_and_scale(shift=-self.mean(), scale=1 / self.std())

    def fix(self, params=None, **kwargs):
        """Return model with parameters in fix fixed

        Args:
         - params: sequence of parameter names to fix, or dict of parameters
            to fix to specific values.

        Other keyword arguments will be added to params.
        """
        if not params and not kwargs:
            return self
        if isinstance(params, str):
            params = (params,)
        if isinstance(params, (list, tuple)):
            params = {pname: self.defaults[pname] for pname in params}
        params = self.validate_params(params, **kwargs, set_defaults=False)
        return self.reparametrize(
            param_specs=tuple([p for p in self.param_specs if p.name not in params]),
            transform_params=functools.partial(_merge_dicts, params),
        )

    def fix_except(self, keep=tuple()):
        """Return new model with only parameters named in keep;
        other parameters will be fixed to their defaults.

        Args:
         - keep: sequence of parameters that should remain
        """
        if isinstance(keep, str):
            keep = (keep,)
        return self.fix([pname for pname in self.param_names if pname not in keep])

    def freeze(self, keep=tuple()):
        """Return new model that takes no parameters.
        All parameters are fixed to their defaults
        """
        return self.fix_except()

    ##
    # Input validation
    ##

    def validate_params(self, params: dict = None, set_defaults=True, **kwargs) -> dict:
        """Return dictionary of parameters for the model

        Args:
         - params: Dictionary of parameters
         - set_defaults: Whether missing parameters should be set
            to their defaults (default True).

        Other keyword arguments are merged with params.
        """
        if params is None:
            params = dict()
        if not isinstance(params, dict):
            raise ValueError(f"Params must be a dict, got {type(params)}")
        params = _merge_dicts(params, kwargs)

        if set_defaults:
            if not params:
                # No params passed at all; we can just return defaults..
                # .. if the params have been passed through validate once,
                # to convert them into tensors.
                if self.defaults and not isinstance(
                    self.defaults[self.param_names[0]], ep.Tensor
                ):
                    raise ValueError("Defaults have not been validated!")
                return self.defaults
            for p in self.param_specs:
                params.setdefault(p.name, p.default)

        # Flag spurious parameters
        spurious = set(params.keys()) - set(self.param_names)
        if spurious:
            raise ValueError(f"Unknown parameters {spurious} passed")

        # Convert all params to tensors
        params = {pname: self._to_tensor(x) for pname, x in params.items()}

        # Find a common batch shape.
        # Walk through shapes from right to left, taking max() of dimsizes,
        # and taking 1 for missing entries.
        param_shapes_rev = [tuple(reversed(x.shape)) for x in params.values()]
        batch_shape = tuple(
            reversed(
                [max(x) for x in itertools.zip_longest(*param_shapes_rev, fillvalue=1)]
            )
        )
        # Cast all param tensors to the common batch shape
        params = {k: ep_util.broadcast_to(v, batch_shape) for k, v in params.items()}
        return params

    def _batch_shape(self, params):
        if not len(params):
            return tuple()
        else:
            return params[self.param_names[0]].shape

    def _expand_params(self, params):
        """"Return params expanded to (batch_shape, [1] * len(sample_shape))
        """
        # Add len(sample_shape) 1s
        # to param shape for broadcasting with data.
        for _ in self.sample_shape:
            params = {pname: x[..., None] for pname, x in params.items()}

        return params

    def _validate_data_or_quantiles(self, x):
        if x is None:
            raise ValueError("None is not valid as data/quantiles")
        # Shorthand data specifications
        try:
            len(x)
        except TypeError:
            # Int/float like
            was_scalar = True
            x = [x]
        else:
            was_scalar = False
        if isinstance(x, (list, tuple)):
            x = self._to_tensor(x)
        x = ep.astensor(x)
        if ep_util.tensorlib(x) != self.backend:
            raise ValueError(
                f"Data and quantiles must be {self.backend.__name__} array/tensors"
            )
        return x, was_scalar

    def validate_data(self, data) -> ep.TensorType:
        """Return eagerpy tensor from data
        """
        data, self._data_is_single_scalar = self._validate_data_or_quantiles(data)
        if len(data.shape) == 1:
            data = data[:, None]
        observed_dim = data.shape[-1]
        if self.n_dim != observed_dim:
            raise ValueError(
                f"Data should have {self.n_dim} observables per event, got {observed_dim}"
            )
        return data

    def _to_tensor(self, x):
        return ep_util.to_tensor(x, tensorlib=self.backend or "numpy")

    def validate_quantiles(self, quantiles) -> ep.TensorType:
        """Return an (n_events) eagerpy tensor from quantiles
        """
        quantiles, self._quantiles_is_single_scalar = self._validate_data_or_quantiles(
            quantiles
        )
        # Note min <= max, since there could be only one unique quantile
        assert 0 <= quantiles.min() <= quantiles.max() <= 1
        return quantiles

    ##
    # Simulation. These always return numpy arrays, not eagerpy tensors.
    ##

    @property
    def simulate_partially_efficient(self):
        return hasattr(self, "simulate_p_keep") and hasattr(
            self, "rate_before_efficiencies"
        )

    def simulate(self, params: dict = None, **kwargs) -> np.ndarray:
        params = self.validate_params(params, **kwargs)
        return self._simulate(params)

    def _simulate(self, params) -> np.ndarray:
        if self.simulate_partially_efficient:
            print("Untested!")
            mu = self.rate_before_efficiencies(params)
            n = np.random.poisson(mu)
            events, p_keep = self.simulate_p_keep(params, size=n)
            events = events[np.random.rand(n) < p_keep]
            return events

        else:
            mu = self._rate(params)
            n = np.random.poisson(mu.raw)
            data = self._rvs(size=n, params=params)
            assert len(data) == n

        return data

    def rvs(self, size: int = 1, params: dict = None, **kwargs) -> np.ndarray:
        params = self.validate_params(params, **kwargs)
        if self.simulate_partially_efficient:
            # Could simulate an excess of events, remove unneeded ones?
            raise NotImplementedError
        return self._rvs(size, params)

    def _rvs(self, size: int, params: dict):
        raise NotImplementedError

    ##
    # Methods using data / quantiles
    ##

    def _tensor_method(
        self,
        # Have to pass the method *name*, since self will change
        name,
        params,
        _input_name="data",
        data=NotChanged,
        quantiles=NotChanged,
        **kwargs,
    ):
        params = self.validate_params(params, **kwargs)
        self = self(data=data, quantiles=quantiles)
        if getattr(self, _input_name) is None:
            raise ValueError(f"Provide {_input_name}")

        expanded_params = self._expand_params(params)

        # returns (batch_shape, sample_shape)
        result = getattr(self, "_" + name)(expanded_params)

        if getattr(self, f"_{_input_name}_is_single_scalar"):
            if self._batch_shape(params):
                # Remove sample_shape = 1 from end
                result = result[..., 0]
            else:
                result = result.item()

        return ep_util.ensure_raw(result)

    # External functions: flexible input, returm value has same type as data.

    def logpdf(self, data=NotChanged, params: dict = None, **kwargs) -> ep.TensorType:
        return self._tensor_method("logpdf", data=data, params=params, **kwargs)

    def pdf(self, data=NotChanged, params: dict = None, **kwargs) -> ep.TensorType:
        return self._tensor_method("pdf", data=data, params=params, **kwargs)

    def diff_rate(
        self, data=NotChanged, params: dict = None, **kwargs
    ) -> ep.TensorType:
        return self._tensor_method("diff_rate", data=data, params=params, **kwargs)

    def cdf(self, data=NotChanged, params: dict = None, **kwargs) -> ep.TensorType:
        return self._tensor_method("cdf", data=data, params=params, **kwargs)

    def log_diff_rate(
        self, data=NotChanged, params: dict = None, **kwargs
    ) -> ep.TensorType:
        return self._tensor_method("log_diff_rate", data=data, params=params, **kwargs)

    def ppf(self, quantiles=NotChanged, params: dict = None, **kwargs) -> ep.TensorType:
        return self._tensor_method(
            "ppf", _input_name="quantiles", quantiles=quantiles, params=params, **kwargs
        )

    # Internal methods: Take and return eagerpy tensors,
    # params have been validated and are (batch_dim, 1) tensors

    def _logpdf(self, params: dict):
        if self._has_redefined("_log_diff_rate"):
            return self._log_diff_rate(params) - self._log_rate(params)
        if self._has_redefined("_pdf") or self._has_redefined("_diff_rate"):
            return ep.log(self._pdf(params))
        raise NotImplementedError(
            "Model should implement _pdf, _logpdf, _diff_rate or _log_diff_rate"
        )

    def _pdf(self, params: dict):
        if self._has_redefined("_diff_rate"):
            return self._diff_rate(params) / self._rate(params)
        return ep.exp(self._logpdf(params))

    def _log_diff_rate(self, params: dict):
        if self._has_redefined("_logpdf"):
            return self._logpdf(params=params) + self._log_rate(params=params)
        return ep.log(self._diff_rate(params=params))

    def _diff_rate(self, params: dict):
        if self._has_redefined("_pdf"):
            return self._pdf(params=params) * self._rate(params=params)
        return ep.exp(self._log_diff_rate(params=params))

    def _cdf(self, params: dict):
        raise NotImplementedError

    def _ppf(self, params: dict):
        raise NotImplementedError

    ##
    # Methods not using data; return (batch_dims,) tensor
    # or scalar if batch_dims == tuple()
    ##

    def _scalar_method(self, method, params, **kwargs):
        params = self.validate_params(params, **kwargs)

        # Scalar methods (e.g. from statistics) might still touch data
        # / be used together with tensor methods that do.
        expanded_params = self._expand_params(params)

        # should return (batch_shape, ones(sample_shape))...
        result = method(expanded_params)

        # ... or a scalar, in case the method didn't use params at all
        # if so, broadcast result as a tensor
        batch_shape = self._batch_shape(params)
        if not hasattr(result, "shape") or not len(result.shape):
            result = (
                result
                * self._expand_params(dict(dummy=self.backend.ones(batch_shape)))[
                    "dummy"
                ]
            )

        # Remove ones(sample_shape) and eagerpy wrapper
        for _ in self.sample_shape:
            result = result[..., 0]
        result = ep_util.ensure_raw(result)

        if not batch_shape:
            # Un-tensorify
            return result.item()
        return result

    # External functions: validate params if needed

    def rate(self, params: dict = None, **kwargs) -> float:
        return self._scalar_method(self._rate, params, **kwargs)

    def mean(self, params: dict = None, **kwargs) -> float:
        return self._scalar_method(self._mean, params, **kwargs)

    def var(self, params: dict = None, **kwargs) -> float:
        return self._scalar_method(self._var, params, **kwargs)

    def std(self, params: dict = None, **kwargs) -> float:
        return self._scalar_method(self._std, params, **kwargs)

    # Internal functions

    def _rate(self, params: dict):
        return params[hypney.DEFAULT_RATE_PARAM.name]

    def _log_rate(self, params: dict):
        return self._rate(params).log()

    def _mean(self, params: dict):
        return NotImplementedError

    def _var(self, params: dict):
        return self._std(params) ** 2

    def _std(self, params: dict):
        return NotImplementedError

    ##
    # Plotting
    ##

    def _plot(self, method, x=None, params=None, auto_labels=True, **kwargs):
        """Plots result of Model.method"""
        import matplotlib.pyplot as plt

        discrete = self.observables[0].integer

        if self.n_dim > 1:
            raise NotImplementedError("Plotting only implemented for 1d models")
        if x is None:
            # Sensible default limits
            quantiles = np.array([0.01, 0.05, 0.95, 0.99])
            try:
                qs = self.ppf(quantiles, params=params)
            except NotImplementedError:
                # ppf not implemented, use simulation to estimate range roughly
                sim_data = self.rvs(size=10_000, params=params)
                qs = np.percentile(sim_data[:, 0], 100 * quantiles)
            qs = dict(zip(quantiles, qs))

            low = qs[0.01] - 2 * (qs[0.05] - qs[0.01])
            high = qs[0.99] + 2 * (qs[0.99] - qs[0.95])

            x = np.linspace(low, high, 500)

            if discrete:
                x = np.unique(np.round(x))

        y = getattr(self, method)(x, params=params)
        if discrete:
            # Use plt.hist rather than a line with drawstyle='steps', to
            # ensure the extreme bins also get drawn completely
            kwargs.setdefault("histtype", "step")
            offset = 0 if method == "cdf" else 0.5
            plt.hist(
                x,
                bins=np.concatenate([x - offset, [x[-1] + 1 - offset]]),
                weights=y,
                **kwargs,
            )
        else:
            plt.plot(x, y, **kwargs)

        if auto_labels:
            plt.xlabel(self.observables[0].name)
            x = self.observables[0].name
            labels = dict(
                pdf=f"P({x})" + ("" if discrete else f" d{x}"),
                cdf=f"Fraction of data below",
                diff_rate=f"Events" + ("" if discrete else f" / d{x}"),
            )
            plt.ylabel(labels[method])

    def plot_pdf(self, x=None, params=None, auto_labels=True, **kwargs):
        return self._plot("pdf", x=x, params=params, auto_labels=auto_labels, **kwargs)

    def plot_diff_rate(self, x=None, params=None, auto_labels=True, **kwargs):
        return self._plot(
            "diff_rate", x=x, params=params, auto_labels=auto_labels, **kwargs
        )

    def plot_cdf(self, x=None, params=None, auto_labels=True, **kwargs):
        return self._plot("cdf", x=x, params=params, auto_labels=auto_labels, **kwargs)


@export
class WrappedModel(Model):
    """A model wrapping another model"""

    _orig_model: Model

    def __init__(self, orig_model: Model = hypney.NotChanged, *args, **kwargs):
        if orig_model is not hypney.NotChanged:
            # No need to make a copy now; any attempted state change
            # (set data, change defaults...) will trigger that
            self._orig_model = orig_model
        kwargs.setdefault("observables", orig_model.observables)
        kwargs.setdefault("param_specs", orig_model.param_specs)
        # TODO: this causes _orig_model._init_data to often be run twice...
        # But without it self.data could be None while self._orig_model.data
        # would be set
        kwargs.setdefault("data", orig_model.data)
        kwargs.setdefault("quantiles", orig_model.quantiles)
        kwargs.setdefault("backend", orig_model._backend_name)
        super().__init__(*args, **kwargs)


def _merge_dicts(x, y):
    return {**x, **y}
