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
class Model(hypney.DataContainer):
    name: str = ""
    param_specs: ty.Tuple[hypney.ParameterSpec] = (hypney.DEFAULT_RATE_PARAM,)
    observables: ty.Tuple[hypney.Observable] = (hypney.DEFAULT_OBSERVABLE,)
    cut: ty.Union[hypney.NoCut, tuple] = hypney.NoCut

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
        param_specs=NotChanged,
        observables=NotChanged,
        cut=NotChanged,
        **new_defaults,
    ):
        self.name = name

        # These have default class attributes
        if param_specs is not NotChanged:
            self.param_specs = param_specs
        if observables is not NotChanged:
            self.observables = observables
        if cut is not NotChanged:
            self._set_cut(cut)

        self._validate_and_set_defaults(new_defaults)
        super().__init__(data=data)

    def _validate_and_set_defaults(self, new_defaults: dict):
        new_defaults = self.validate_params(new_defaults)
        self.param_specs = tuple(
            [p._replace(default=new_defaults[p.name]) for p in self.param_specs]
        )

    # TODO: freeze is a poor name, can change things twice...
    # copy is worse since we may not copy
    def freeze(
        self,
        name=NotChanged,
        data=NotChanged,
        cut=NotChanged,
        fix=None,
        keep=None,
        **new_defaults,
    ):
        """Return a model with possibly changed name, defaults, data, or parameters"""
        if (
            name is NotChanged
            and data is NotChanged
            and cut is NotChanged
            and not new_defaults
            and fix is None
            and keep is None
        ):
            return self
        new_self = copy(self)
        if name is not NotChanged:
            new_self.name = name
        new_self._validate_and_set_defaults(new_defaults)
        if data is not NotChanged:
            new_self._set_data(data)
        if cut is not NotChanged:
            new_self._set_cut(cut)
        new_self = new_self.filter_params(fix=fix, keep=keep)
        return new_self

    def _has_redefined(self, method_name, from_base=None):
        """Returns if method_name is redefined from Model.method_name"""
        if from_base is None:
            from_base = Model
        f = getattr(self, method_name)
        if not hasattr(f, "__func__"):
            return True
        return f.__func__ is not getattr(from_base, method_name)

    def _set_cut(self, cut):
        self.cut = self.validate_cut(cut)
        self._init_cut()

    def _init_cut(self):
        """Called during initialization, if cut is to be frozen"""
        pass

    ##
    # Creating new models from old
    ##

    def __call__(self, *args, **kwargs):
        return self.freeze(*args, **kwargs)

    def __add__(self, other):
        return hypney.models.Mixture(self, other)

    def __pow__(self, other):
        return hypney.models.TensorProduct(self, other)

    def filter_params(self, fix=None, keep=None):
        """Return new model with parameters in fix fixed

        Args:
         - fix: sequence of parameter names to fix, or dict of parameters
            to fix to specific values
         - free: sequence of parameter names to keep free, others will be
            fixed to their defaults.

        Either fix or free may be specified, but not both.
        """
        if fix is None and keep is None:
            return self
        fix = self._process_fix_keep(fix, keep)
        return hypney.models.TransformedModel(
            orig_model=self,
            param_specs=tuple([p for p in self.param_specs if p.name not in fix]),
            transform_params=functools.partial(_merge_dicts, fix),
        )

    def _process_fix_keep(self, fix=None, keep=None):
        if keep is not None:
            if fix is not None:
                raise ValueError("Specify either free or fix, not both")
            if isinstance(keep, str):
                keep = (keep,)
            fix = [pname for pname in self.param_names if pname not in keep]

        if fix is None:
            fix = dict()
        if isinstance(fix, str):
            fix = (fix,)
        if isinstance(fix, (tuple, list)):
            fix = {pname: self.defaults[pname] for pname in fix}

        fix = self.validate_params(fix, set_defaults=False)
        return fix

    ##
    # Input validation
    ##

    def validate_params(self, params: dict, set_defaults=True) -> dict:
        if params is None:
            params = dict()
        if not isinstance(params, dict):
            raise ValueError(f"Params must be a dict, got {type(params)}")

        if set_defaults:
            for p in self.param_specs:
                params.setdefault(p.name, p.default)

        # Bounds check
        for p in self.param_specs:
            if p.name not in params:
                continue
            val = params[p.name]
            if not p.min <= params[p.name] < p.max:
                raise ValueError(
                    f"{val} is out of bounds {(p.min, p.max)} for {p.name}"
                )

        # Flag spurious parameters
        spurious = set(params.keys()) - set(self.param_names)
        if spurious:
            raise ValueError(f"Unknown parameters {spurious} passed")

        return params

    def validate_data(self, data) -> ep.TensorType:
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

    def validate_cut(self, cut):
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
    # Simulation
    ##

    @property
    def simulate_partially_efficient(self):
        return hasattr(self, "simulate_p_keep") and hasattr(
            self, "rate_before_efficiencies"
        )

    def simulate(self, params: dict = None, *, cut=NotChanged) -> np.ndarray:
        params = self.validate_params(params)
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
            data = self._rvs(params, size=n)
            assert len(data) == n
            data = self.apply_cut(data).raw

        return data

    def rvs(self, params: dict = None, size: int = 1) -> np.ndarray:
        if self.simulate_partially_efficient:
            # Could simulate an excess of events, remove unneeded ones?
            raise NotImplementedError
        params = self.validate_params(params)
        return self._rvs(params, size)

    def _rvs(self, params: dict, size: int):
        raise NotImplementedError

    ##
    # Methods using data
    ##

    def apply_cut(self, data=NotChanged, cut=NotChanged):
        return self(data=data, cut=cut)._apply_cut()

    def _apply_cut(self):
        if self.cut == hypney.NoCut:
            return self.data
        passed = 1 + 0 * self.data[:, 0]
        for dim_i, (l, r) in enumerate(self.cut):
            passed *= (l <= self.data[:, dim_i]) & (self.data[:, dim_i] < r)
        return self.data[passed]

    def diff_rate(
        self, params: dict = None, data: ep.TensorType = NotChanged, *, cut=NotChanged
    ) -> ep.TensorType:
        params = self.validate_params(params)
        return self(data=data, cut=cut)._diff_rate(params)

    def _diff_rate(self, params: dict):
        if not self._has_redefined("_pdf"):
            raise NotImplementedError(
                "Can't compute pdf of a Model implementing "
                "neither _pdf nor _diff_rate"
            )
        return self._pdf(params=params) * self._rate(params=params)

    def pdf(
        self, params: dict = None, data: ep.TensorType = NotChanged
    ) -> ep.TensorType:
        params = self.validate_params(params)
        return self(data=data)._pdf(params)

    def _pdf(self, params: dict):
        if not self._has_redefined("_diff_rate"):
            raise NotImplementedError(
                "Can't compute pdf of a Model implementing "
                "neither _pdf nor _diff_rate"
            )
        return self._diff_rate(self.data, params) / self.rate(params)

    def cdf(
        self, params: dict = None, data: ep.TensorType = NotChanged
    ) -> ep.TensorType:
        params = self.validate_params(params)
        return self(data=data)._cdf(params)

    def _cdf(self, params: dict):
        raise NotImplementedError

    ##
    # Methods not using data
    ##

    def rate(self, params: dict = None, cut=NotChanged) -> ep.TensorType:
        params = self.validate_params(params)
        return self._rate(params) * self(cut=cut).cut_efficiency(params=params)

    def _rate(self, params: dict):
        return params[hypney.DEFAULT_RATE_PARAM.name]

    def cut_efficiency(
        self, params: dict = None, cut=NotChanged,
    ):
        params = self.validate_params(params)
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
        return ((signs) * self.cdf(params=params, data=points)).sum()


def _merge_dicts(x, y):
    # Lambda function / closure wouldn't pickle..
    return {**x, **y}
