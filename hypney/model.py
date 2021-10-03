from copy import copy
import functools
import typing as ty

import eagerpy as ep
import numpy as np

import hypney
from hypney import NotChanged
from hypney.utils.eagerpy import ensure_raw

export, __all__ = hypney.exporter()


@export
class Model:
    name: str = ""
    param_specs: ty.Tuple[hypney.ParameterSpec] = (hypney.DEFAULT_RATE_PARAM,)
    observables: ty.Tuple[hypney.Observable] = (hypney.DEFAULT_OBSERVABLE,)
    data: ep.Tensor = None
    quantiles: ep.Tensor = None

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

    ##
    # Initialization
    ##

    def __init__(
        self,
        *,
        name="",
        data=None,
        params=NotChanged,  # Really defaults...
        param_specs=NotChanged,
        observables=NotChanged,
        quantiles=None,
        **kwargs,
    ):
        self.name = name

        # These have default class attributes
        if param_specs is not NotChanged:
            self.param_specs = param_specs
        if observables is not NotChanged:
            self.observables = observables

        self._validate_and_set_defaults(params, **kwargs)
        self._set_data(data)
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

    def _validate_and_set_defaults(
        self, new_defaults: dict = NotChanged, **kwargs: dict
    ):
        if new_defaults == NotChanged:
            new_defaults = dict()
        new_defaults = self.validate_params(new_defaults, **kwargs)
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

    def set(
        self,
        *,
        name=NotChanged,
        data=NotChanged,
        params=NotChanged,
        quantiles=NotChanged,
        fix=None,
        fix_except=None,
        **kwargs,
    ):
        """Return a model with possibly changed name, defaults, data, or parameters"""
        if (
            name is NotChanged
            and data is NotChanged
            and quantiles is NotChanged
            and not params
            and not kwargs
            and fix is None
            and fix_except is None
        ):
            return self
        new_self = copy(self)
        if name is not NotChanged:
            new_self.name = name
        new_self._validate_and_set_defaults(params, **kwargs)
        new_self._set_data(data)
        new_self._set_quantiles(quantiles)
        if fix is not None:
            new_self = new_self.fix(fix)
            if fix_except is not None:
                raise ValueError("Provide either fix or fix_except, not both")
        if fix_except is not None:
            new_self = new_self.fix_except(fix)
        return new_self

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
        """Return new model with parameters in fix fixed

        Args:
         - params: sequence of parameter names to fix, or dict of parameters
            to fix to specific values.

        Other keyword arguments will be added to params.
        """
        if params is None:
            params = dict()
        if isinstance(params, str):
            params = (params,)
        if isinstance(params, (list, tuple)):
            params = {pname: self.defaults[pname] for pname in params}
        return self._fix(_merge_dicts(params, kwargs))

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

    def _fix(self, fix):
        fix = self.validate_params(fix, set_defaults=False)
        return self.reparametrize(
            param_specs=tuple([p for p in self.param_specs if p.name not in fix]),
            transform_params=functools.partial(_merge_dicts, fix),
        )

    ##
    # Input validation
    ##

    def validate_params(self, params: dict = None, set_defaults=True, **kwargs) -> dict:
        """Return dictionary of parameters for the model.

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
            for p in self.param_specs:
                params.setdefault(p.name, p.default)

        # Flag spurious parameters
        spurious = set(params.keys()) - set(self.param_names)
        if spurious:
            raise ValueError(f"Unknown parameters {spurious} passed")

        return params

    def validate_data(self, data) -> ep.TensorType:
        """Return an (n_events, n_observables) eagerpy tensor from data
        """
        if data is None:
            raise ValueError("None is not valid as data")
        # Shorthand data specifications
        try:
            len(data)
        except TypeError:
            # Int/float like
            self._data_is_single_scalar = True
            data = [data]
        if isinstance(data, (list, tuple)):
            if self.data is None:
                data = np.asarray(data)
            else:
                # Preserve existing tensor type
                data = hypney.utils.eagerpy.to_tensor(data, match_type=self.data)
        if len(data.shape) == 1:
            data = data[:, None]
        data = ep.astensor(data)

        observed_dim = data.shape[1]
        if self.n_dim != observed_dim:
            raise ValueError(
                f"Data should have {self.n_dim} observables per event, got {observed_dim}"
            )
        return data

    def validate_quantiles(self, quantiles) -> ep.TensorType:
        """Return an (n_events) eagerpy tensor from quantiles
        """
        # TODO: much of this duplicates validate_data...
        if quantiles is None:
            raise ValueError("None is not valid as quantiles")
        try:
            len(quantiles)
        except TypeError:
            self._quantiles_is_single_scalar = True
            quantiles = [quantiles]
        if isinstance(quantiles, (list, tuple)):
            if self.data is None:
                quantiles = np.asarray(quantiles)
            else:
                quantiles = hypney.utils.eagerpy.to_tensor(
                    quantiles, match_type=self.data
                )
        # Note min <= max, maybe there is only one unique quantile
        assert 0 <= quantiles.min() <= quantiles.max() <= 1
        quantiles = ep.astensor(quantiles)
        return quantiles

    ##
    # Simulation. These functions return numpy arrays, not eagerpy tensors.
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
            n = np.random.poisson(mu)
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
    #   * _x: Internal function.
    #       Takes and returns eagerpy tensors.
    #       Uses self.data, assumes params have been validated.
    #   * x_: 'Friendly' function for use in other hypney classes.
    #       Flexible input, returns eagerpy tensors.
    #   * x: External function, for users to call directly.
    #       Flexible input, returm value has same type as data.
    ##

    # TODO: make these decorators? See if pickle is OK with that..
    def _return_like_data(self, result):
        if self._data_is_single_scalar:
            return result.item()
        return result

    def _return_like_quantiles(self, result):
        if self._quantiles_is_single_scalar:
            return result.item()
        return result

    def logpdf(self, data=NotChanged, params: dict = None, **kwargs):
        return ensure_raw(self.logpdf_(data=data, params=params, **kwargs))

    def logpdf_(self, data=NotChanged, params: dict = None, **kwargs) -> ep.TensorType:
        params = self.validate_params(params, **kwargs)
        self = self(data=data)
        if self.data is None:
            raise ValueError("Provide data")
        return self._return_like_data(self._logpdf(params))

    def _logpdf(self, params: dict):
        if self._has_redefined("_log_diff_rate"):
            return self._log_diff_rate(params) - self._log_rate(params)
        if self._has_redefined("_pdf") or self._has_redefined("_diff_rate"):
            return ep.log(self._pdf(params))
        raise NotImplementedError(
            "Model should implement _pdf, _logpdf, _diff_rate or _log_diff_rate"
        )

    def pdf(self, data=NotChanged, params: dict = None, **kwargs):
        return ensure_raw(self.pdf_(data=data, params=params, **kwargs))

    def pdf_(self, data=NotChanged, params: dict = None, **kwargs) -> ep.TensorType:
        params = self.validate_params(params, **kwargs)
        self = self(data=data)
        if self.data is None:
            raise ValueError("Provide data")
        return self._return_like_data(self._pdf(params))

    def _pdf(self, params: dict):
        if self._has_redefined("_diff_rate"):
            return self._diff_rate(params) / self._rate(params)
        return ep.exp(self._logpdf(params))

    def log_diff_rate(self, data=NotChanged, params: dict = None, **kwargs):
        return ensure_raw(self.log_diff_rate_(data=data, params=params, **kwargs))

    def log_diff_rate_(
        self, data=NotChanged, params: dict = None, **kwargs
    ) -> ep.TensorType:
        params = self.validate_params(params, **kwargs)
        self = self(data=data)
        if self.data is None:
            raise ValueError("Provide data")
        return self._return_like_data(self._log_diff_rate(params))

    def _log_diff_rate(self, params: dict):
        if self._has_redefined("_logpdf"):
            return self._logpdf(params=params) + self._log_rate(params=params)
        return ep.log(self._diff_rate(params=params))

    def diff_rate(self, data=NotChanged, params: dict = None, **kwargs):
        return ensure_raw(self.diff_rate_(data=data, params=params, **kwargs))

    def diff_rate_(
        self, data=NotChanged, params: dict = None, **kwargs
    ) -> ep.TensorType:
        params = self.validate_params(params, **kwargs)
        self = self(data=data)
        if self.data is None:
            raise ValueError("Provide data")
        return self._return_like_data(self._diff_rate(params))

    def _diff_rate(self, params: dict):
        if self._has_redefined("_pdf"):
            return self._pdf(params=params) * self._rate(params=params)
        return ep.exp(self._log_diff_rate(params=params))

    def cdf(self, data=NotChanged, params: dict = None, **kwargs):
        return ensure_raw(self.cdf_(data=data, params=params, **kwargs))

    def cdf_(self, data=NotChanged, params: dict = None, **kwargs) -> ep.TensorType:
        params = self.validate_params(params, **kwargs)
        self = self(data=data)
        if self.data is None:
            raise ValueError("Provide data")
        return self._return_like_data(self._cdf(params))

    def _cdf(self, params: dict):
        raise NotImplementedError

    def ppf(self, quantiles=NotChanged, params: dict = None, **kwargs):
        return ensure_raw(self.ppf_(quantiles=quantiles, params=params, **kwargs))

    def ppf_(
        self, quantiles=NotChanged, params: dict = None, **kwargs
    ) -> ep.TensorType:
        params = self.validate_params(params, **kwargs)
        self = self(quantiles=quantiles)
        if self.quantiles is None:
            raise ValueError("Provide quantiles")
        return self._return_like_quantiles(self._ppf(params))

    def _ppf(self, params: dict):
        raise NotImplementedError

    ##
    # Methods not using data; return a simple float
    # As above, _x are internal functions (use self.data)
    # whereas x are external functions (do input validation, set data if needed)
    ##

    def rate(self, params: dict = None, **kwargs) -> float:
        params = self.validate_params(params, **kwargs)
        return self._rate(params)

    def _rate(self, params: dict):
        return params[hypney.DEFAULT_RATE_PARAM.name]

    def _log_rate(self, params: dict):
        return hypney.utils.eagerpy.to_tensor(
            self.rate(params), match_type=self.data
        ).log()

    def mean(self, params: dict = None, **kwargs) -> float:
        params = self.validate_params(params, **kwargs)
        return self._mean(params)

    def _mean(self, params: dict):
        return NotImplementedError

    def var(self, params: dict = None, **kwargs) -> float:
        params = self.validate_params(params, **kwargs)
        return self._var(params)

    def _var(self, params: dict):
        return self._std(params) ** 2

    def std(self, params: dict = None, **kwargs) -> float:
        params = self.validate_params(params, **kwargs)
        return self._std(params)

    def _std(self, params: dict):
        return NotImplementedError

    ##
    # Plotting
    ##

    def _plot(self, method, x=None, params=None, auto_labels=True, **kwargs):
        """Plots differential rate of model"""
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

    def __init__(self, orig_model=hypney.NotChanged, *args, **kwargs):
        if orig_model is not hypney.NotChanged:
            # No need to make a copy now; any attempted state change
            # (set data, change defaults...) will trigger that
            self._orig_model = orig_model
        kwargs.setdefault("observables", self._orig_model.observables)
        kwargs.setdefault("param_specs", self._orig_model.param_specs)
        # TODO: this causes _orig_model._init_data to often be run twice...
        # But without it self.data could be None while self._orig_model.data
        # would be set
        kwargs.setdefault("data", self._orig_model.data)
        kwargs.setdefault("quantiles", self._orig_model.quantiles)
        super().__init__(*args, **kwargs)


def _merge_dicts(x, y):
    return {**x, **y}
