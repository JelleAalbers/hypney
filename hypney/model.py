from copy import copy
import itertools
import typing as ty

import numpy as np

import hypney

export, __all__ = hypney.exporter(also_export=("DEFAULT_OBSERVABLE",))


@export
class Observable(ty.NamedTuple):
    """Description of a observable space: name and limits"""

    name: str
    min: float = -float("inf")
    max: float = float("inf")
    # Whether only integer values are allowed
    integer: bool = False


DEFAULT_OBSERVABLE = Observable(name="x", min=-float("inf"), max=float("inf"))


@export
class NotChanged:
    """Default argument where None would be ambiguous"""
    pass


@export
class NoCut:
    pass


@export
class Model(hypney.Element):

    observables: ty.Tuple[Observable] = (DEFAULT_OBSERVABLE,)
    cut: ty.Union[NoCut, tuple] = NoCut

    @property
    def n_dim(self):
        return len(self.observables)

    # Initialization

    def __init__(
        self, name="", data=None, param_specs=None, observables=None, cut=NoCut, **new_defaults
    ):
        self.name = name

        # These are often set as class attributes, so are allowed to be None
        if param_specs is not None:
            self.param_specs = param_specs
        if observables is not None:
            self.observables = observables

        self._set_defaults(new_defaults)
        self._set_cut(cut)
        self._set_data(data)

    def __call__(self, name=NotChanged, data=NotChanged, cut=NotChanged, **new_defaults):
        """Return a model with possibly changed name, defaults, or data"""
        # If the user explicitly sets data=None, it would be ambiguous:
        # should we return a model with the same default, or with data 'unset'?
        # Hence the funny _not_changed default argument instead of None
        if name is NotChanged and data is NotChanged and cut is NotChanged and not new_defaults:
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

    def _has_redefined(self, method_name):
        """Returns if method_name is redefined from Model.method_name"""
        f = getattr(self, method_name)
        if not hasattr(f, "__func__"):
            return True
        return f.__func__ is not getattr(Model, method_name)

    def _set_cut(self, cut):
        self.cut = self.validate_cut(cut)
        self.init_cut()

    def init_cut(self):
        """Called during initialization, if cut is to be frozen"""
        pass

    # Validation

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
        if cut is NoCut:
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

    @property
    def simulator_gives_efficiency(self):
        return hasattr(self, "simulate_partially_efficient") and hasattr(
            self, "rate_before_efficiencies"
        )

    def simulate(self, params: dict = None, *, cut=NotChanged) -> np.ndarray:
        params = self.validate_params(params)

        if self.simulator_gives_efficiency:
            mu = self.rate_before_efficiencies(params)
            n = np.random.poisson(mu)
            events, p_keep = self(cut=cut).simulate_partially_efficient(params, size=n)
            events = events[np.random.rand(n) < p_keep]
            return events

        else:
            mu = self.rate(params)
            n = np.random.poisson(mu)
            data = self.rvs(params, size=n)
            assert len(data) == n
            data = self.apply_cut(data, cut)

        return data

    # Methods taking data
    # _ methods do not take data

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
        if self.cut is NoCut:
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

    # Methods not taking data

    def rate(self, params: dict = None, cut=NotChanged) -> np.ndarray:
        params = self.validate_params(params)
        return self._rate(params) * self(cut=cut).cut_efficiency(params=params)

    def _rate(self, params: dict):
        return params[hypney.DEFAULT_RATE_PARAM.name]

    def rvs(self, params: dict = None, size: int = 1) -> np.ndarray:
        params = self.validate_params(params)
        return self.rvs(params, size)

    def _rvs(params, size):
        raise NotImplementedError

    def cut_efficiency(
        self, params=None, cut=NotChanged,
    ):
        params = self.validate_params(params)
        return self(cut=cut)._cut_efficiency(params)

    def _cut_efficiency(self, params):
        if self.cut is NoCut:
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

    def __add__(self, other):
        return hypney.Mixture(self, other)
