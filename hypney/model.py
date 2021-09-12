from copy import copy
import itertools
import typing as ty

import numpy as np

import hypney
from hypney import NotChanged

export, __all__ = hypney.exporter()


@export
class Model(hypney.Element):

    observables: ty.Tuple[hypney.Observable] = (hypney.DEFAULT_OBSERVABLE,)
    cut: ty.Union[hypney.NoCut, tuple] = hypney.NoCut

    @property
    def n_dim(self):
        return len(self.observables)

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

        self._set_defaults(new_defaults)
        self._set_data(data)

    def __call__(
        self, name=NotChanged, data=NotChanged, cut=NotChanged, **new_defaults
    ):
        """Return a model with possibly changed name, defaults, or data"""
        if (
            name is NotChanged
            and data is NotChanged
            and cut is NotChanged
            and not new_defaults
        ):
            return self
        new_self = copy(self)
        if name is not NotChanged:
            new_self.name = name
        new_self._set_defaults(new_defaults)
        if data is not NotChanged:
            new_self._set_data(data)
        if cut is not NotChanged:
            new_self._set_cut(cut)
        return new_self

    def __add__(self, other):
        return hypney.Mixture(self, other)

    def _has_redefined(self, method_name):
        """Returns if method_name is redefined from Model.method_name"""
        f = getattr(self, method_name)
        if not hasattr(f, "__func__"):
            return True
        return f.__func__ is not getattr(Model, method_name)

    def _set_cut(self, cut):
        self.cut = self.validate_cut(cut)
        self._init_cut()

    def _init_cut(self):
        """Called during initialization, if cut is to be frozen"""
        pass

    ##
    # Input validation
    ##

    def validate_data(self, data: np.ndarray) -> np.ndarray:
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
        if isinstance(cut, (list, np.ndarray)):
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

    def _simulate(self, params):
        if self.simulate_partially_efficient:
            mu = self.rate_before_efficiencies(params)
            n = np.random.poisson(mu)
            events, p_keep = self.simulate_p_keep(params, size=n)
            events = events[np.random.rand(n) < p_keep]
            return events

        else:
            mu = self.rate(params)
            n = np.random.poisson(mu)
            data = self.rvs(params, size=n)
            assert len(data) == n
            data = self.apply_cut(data)

        return data

    ##
    # Methods taking data
    ##

    def diff_rate(
        self, params: dict = None, data: np.ndarray = NotChanged, *, cut=NotChanged
    ) -> np.ndarray:
        params = self.validate_params(params)
        return self(data=data, cut=cut)._diff_rate(params)

    def _diff_rate(self, params: dict):
        if not self._has_redefined("_pdf"):
            raise NotImplementedError(
                "Can't compute pdf of a Model implementing "
                "neither _pdf nor _diff_rate"
            )
        return self._pdf(params=params) * self._rate(params=params)

    def apply_cut(self, data=NotChanged, cut=NotChanged):
        return self(data=data, cut=cut)._apply_cut()

    def _apply_cut(self):
        if self.cut is hypney.NoCut:
            return self.data
        passed = np.ones(len(self.data), np.bool_)
        for dim_i, (l, r) in enumerate(self.cut):
            passed *= (l <= self.data[:, dim_i]) & (self.data[:, dim_i] < r)
        return self.data[passed]

    def pdf(self, params: dict = None, data: np.ndarray = NotChanged) -> np.ndarray:
        params = self.validate_params(params)
        return self(data=data)._pdf(params)

    def _pdf(self, params: dict):
        if not self._has_redefined("_diff_rate"):
            raise NotImplementedError(
                "Can't compute pdf of a Model implementing "
                "neither _pdf nor _diff_rate"
            )
        return self._diff_rate(self.data, params) / self.rate(params)

    def cdf(self, params: dict = None, data: np.ndarray = NotChanged) -> np.ndarray:
        params = self.validate_params(params)
        return self(data=data)._cdf(params)

    def _cdf(self, params: dict):
        raise NotImplementedError

    ##
    # Methods not taking data
    ##

    def rate(self, params: dict = None, cut=NotChanged) -> np.ndarray:
        params = self.validate_params(params)
        return self._rate(params) * self(cut=cut).cut_efficiency(params=params)

    def _rate(self, params: dict):
        return params[hypney.DEFAULT_RATE_PARAM.name]

    def rvs(self, params: dict = None, size: int = 1) -> np.ndarray:
        params = self.validate_params(params)
        return self._rvs(params, size)

    def _rvs(self, params: dict, size: int):
        raise NotImplementedError

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
                    np.prod(indices),
                    [c[int(0.5 * j + 0.5)] for (c, j) in zip(self.cut, indices)],
                )
                for indices in itertools.product(
                    *[[lower_sign, -lower_sign]] * len(self.cut)
                )
            ]
        )
        return np.sum(np.array(signs) * self.cdf(params=params, data=points))
