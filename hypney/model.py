from copy import copy
import functools
import itertools
import math
import typing as ty

import eagerpy as ep
import numpy as np

import hypney
from hypney import NotChanged

export, __all__ = hypney.exporter()


@export
class Model:
    name: str = ""
    param_specs: ty.Tuple[hypney.ParameterSpec] = (hypney.DEFAULT_RATE_PARAM,)
    observables: ty.Tuple[hypney.Observable] = (hypney.DEFAULT_OBSERVABLE,)
    cut: ty.Union[hypney.NoCut, tuple] = hypney.NoCut
    data: ep.Tensor = None
    quantiles: ep.Tensor = None

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
        name="",
        data=None,
        params=NotChanged,  # Really defaults...
        param_specs=NotChanged,
        observables=NotChanged,
        cut=NotChanged,
        quantiles=None,
        **kwargs,
    ):
        self.name = name

        # These have default class attributes
        if param_specs is not NotChanged:
            self.param_specs = param_specs
        if observables is not NotChanged:
            self.observables = observables
        if cut is not NotChanged:
            self._set_cut(cut)

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

    def _set_cut(self, cut):
        if cut is NotChanged:
            return
        self.cut = self.validate_cut(cut)
        self._init_cut()

    def _init_cut(self):
        """Called during initialization, if cut is to be frozen"""
        pass

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

    def __add__(self, other):
        return hypney.models.Mixture(self, other)

    def __pow__(self, other):
        return hypney.models.TensorProduct(self, other)

    def set(
        self,
        *,
        name=NotChanged,
        data=NotChanged,
        params=NotChanged,
        cut=NotChanged,
        quantiles=NotChanged,
        fix=None,
        fix_except=None,
        **kwargs,
    ):
        """Return a model with possibly changed name, defaults, data, or parameters"""
        if (
            name is NotChanged
            and data is NotChanged
            and cut is NotChanged
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
        new_self._set_cut(cut)
        new_self._set_quantiles(quantiles)
        if fix is not None:
            new_self = new_self.fix(fix)
            if fix_except is not None:
                raise ValueError("Provide either fix or fix_except, not both")
        if fix_except is not None:
            new_self = new_self.fix_except(fix)
        return new_self

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
        other paramters will be fixed to their defaults.

        Args:
         - keep: sequence of parameters that should remain
        """
        if isinstance(keep, str):
            keep = (keep,)
        return self.fix([pname for pname in self.param_names if pname not in keep])

    def _fix(self, fix):
        fix = self.validate_params(fix, set_defaults=False)
        return hypney.models.TransformParams(
            orig_model=self,
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
            data = [data]
        if isinstance(data, (list, tuple)):
            data = np.asarray(data)
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
            quantiles = [quantiles]
        if isinstance(quantiles, (list, tuple)):
            if self.data is None:
                quantiles = np.asarray(quantiles)
            else:
                quantiles = hypney.utils.eagerpy.sequence_to_tensor(
                    quantiles, match_type=self.data
                )
        # Note min <= max, maybe there is only one unique quantile
        assert 0 <= quantiles.min() <= quantiles.max() <= 1
        quantiles = ep.astensor(quantiles)
        return quantiles

    def validate_cut(self, cut):
        """Return a valid cut, i.e. NoCut or tuple of (l, r) tuples for each observable.
        """
        if cut is hypney.NoCut:
            return cut
        if cut is None:
            raise ValueError("None is not a valid cut, use NoCut")
        if isinstance(cut, (list, np.ndarray, ep.Tensor)):
            cut = tuple(cut)
        if not isinstance(cut, tuple):
            raise ValueError("Cut should be a tuple")
        if len(cut) == 2 and len(self.observables) == 1:
            cut = (cut,)
        if len(cut) != len(self.observables):
            raise ValueError("Cut should have same length as observables")
        if any([len(c) != 2 for c in cut]):
            raise ValueError("Cut should be a tuple of 2-tuples")
        return cut

    ##
    # Simulation. These functions return numpy arrays, not eagerpy tensors.
    ##

    @property
    def simulate_partially_efficient(self):
        return hasattr(self, "simulate_p_keep") and hasattr(
            self, "rate_before_efficiencies"
        )

    def simulate(self, params: dict = None, *, cut=NotChanged, **kwargs) -> np.ndarray:
        params = self.validate_params(params, **kwargs)
        return self(cut=cut)._simulate(params)

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
            data = self.apply_cut_(data).raw

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
    #       Uses self.data and self.cut, assumes params have been validated.
    #   * x_: 'Friendly' function for use in other hypney classes.
    #       Flexible input, returns eagerpy tensors.
    #   * x: External function, for users to call directly.
    #       Flexible input, returm value has same type as data.
    ##

    def apply_cut(self, data=NotChanged, cut=NotChanged):
        return self.apply_cut_(data=data, cut=cut).raw

    def apply_cut_(self, data=NotChanged, cut=NotChanged):
        return self(data=data, cut=cut)._apply_cut()

    def _apply_cut(self):
        if self.cut == hypney.NoCut:
            return self.data
        passed = 1 + 0 * self.data[:, 0]
        for dim_i, (l, r) in enumerate(self.cut):
            passed *= (l <= self.data[:, dim_i]) & (self.data[:, dim_i] < r)
        return self.data[passed]

    def diff_rate(
        self, data=NotChanged, params: dict = None, *, cut=NotChanged, **kwargs
    ):
        return self.diff_rate_(data=data, params=params, cut=cut, **kwargs).raw

    def diff_rate_(
        self, data=NotChanged, params: dict = None, *, cut=NotChanged, **kwargs
    ) -> ep.TensorType:
        params = self.validate_params(params, **kwargs)
        self = self(data=data, cut=cut)
        if self.data is None:
            raise ValueError("Provide data")
        return self._diff_rate(params)

    def _diff_rate(self, params: dict):
        if not self._has_redefined("_pdf"):
            raise NotImplementedError(
                "Can't compute pdf of a Model implementing "
                "neither _pdf nor _diff_rate"
            )
        return self._pdf(params=params) * self._rate(params=params)

    def pdf(self, data=NotChanged, params: dict = None, *, cut=NotChanged, **kwargs):
        return self.pdf_(data=data, params=params, cut=cut, **kwargs).raw

    def pdf_(
        self, data=NotChanged, params: dict = None, *, cut=NotChanged, **kwargs
    ) -> ep.TensorType:
        params = self.validate_params(params, **kwargs)
        self = self(data=data, cut=cut)
        if self.data is None:
            raise ValueError("Provide data")
        return self._pdf(params)

    # TODO: test _has_redefined error is raised
    def _pdf(self, params: dict):
        if not self._has_redefined("_diff_rate"):
            raise NotImplementedError(
                "Can't compute pdf of a Model implementing "
                "neither _pdf nor _diff_rate"
            )
        return self._diff_rate(params) / self._rate(params)

    def cdf(self, data=NotChanged, params: dict = None, *, cut=NotChanged, **kwargs):
        return self.cdf_(data=data, params=params, cut=cut, **kwargs).raw

    def cdf_(
        self, data=NotChanged, params: dict = None, *, cut=cut, **kwargs
    ) -> ep.TensorType:
        params = self.validate_params(params, **kwargs)
        self = self(data=data, cut=cut)
        if self.data is None:
            raise ValueError("Provide data")
        return self._cdf(params)

    def _cdf(self, params: dict):
        raise NotImplementedError

    def ppf(
        self, quantiles=NotChanged, params: dict = None, *, cut=NotChanged, **kwargs
    ):
        return self.ppf_(quantiles=quantiles, params=params, cut=cut, **kwargs).raw

    def ppf_(
        self, quantiles=NotChanged, params: dict = None, *, cut=cut, **kwargs
    ) -> ep.TensorType:
        params = self.validate_params(params, **kwargs)
        self = self(quantiles=quantiles, cut=cut)
        if self.quantiles is None:
            raise ValueError("Provide quantiles")
        return self._ppf(params)

    def _ppf(self, params: dict):
        raise NotImplementedError

    ##
    # Methods not using data; return a simple float
    # As above, _x are internal functions (use self.data and self.cut)
    # whereas x are external functions (do input validation, set data and cut if needed)
    ##

    def rate(self, params: dict = None, *, cut=NotChanged, **kwargs) -> float:
        params = self.validate_params(params, **kwargs)
        return self(cut=cut)._rate(params)

    def _rate(self, params: dict):
        return params[hypney.DEFAULT_RATE_PARAM.name] * self._cut_efficiency(params)

    def mean(self, params: dict = None, *, cut=NotChanged, **kwargs) -> float:
        params = self.validate_params(params, **kwargs)
        return self(cut=cut)._mean(params)

    def _mean(self, params: dict):
        return NotImplementedError

    def var(self, params: dict = None, *, cut=NotChanged, **kwargs) -> float:
        params = self.validate_params(params, **kwargs)
        return self(cut=cut)._var(params)

    def _var(self, params: dict):
        return self._std(params) ** 2

    def std(self, params: dict = None, *, cut=NotChanged, **kwargs) -> float:
        params = self.validate_params(params, **kwargs)
        return self(cut=cut)._std(params)

    def _std(self, params: dict):
        return NotImplementedError

    def cut_efficiency(self, params: dict = None, cut=NotChanged, **kwargs) -> float:
        params = self.validate_params(params, **kwargs)
        return self(cut=cut)._cut_efficiency(params)

    def _cut_efficiency(self, params: dict):
        if self.cut is hypney.NoCut:
            return 1.0
        if not hasattr(self, "cdf"):
            raise NotImplementedError("Nontrivial cuts require a cdf")
        # Evaluate CDF at rectangle endpoints, add up with alternating signs,
        # always + in upper right.
        # TODO: Not sure this is correct for n > 2!
        # (for n=3 looks OK, for higher n I can't draw/visualize)
        lower_sign = (-1) ** (len(self.cut))
        signs, points = zip(
            *[
                (
                    math.prod(indices),
                    [c[int(0.5 * j + 0.5)] for (c, j) in zip(self.cut, indices)],
                )
                for indices in itertools.product(
                    *[[lower_sign, -lower_sign]] * len(self.cut)
                )
            ]
        )
        return ((signs) * self.cdf_(data=points, params=params)).sum()


def _merge_dicts(x, y):
    return {**x, **y}
