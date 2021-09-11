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
class Model(hypney.Element):

    observables: ty.Tuple[Observable] = (DEFAULT_OBSERVABLE,)

    @property
    def n_dim(self):
        return len(self.observables)

    # Initialization

    def __init__(self, name="", data=None, params=None, observables=None, **new_defaults):
        self.name = name
        if params is not None:
            self.param_specs = params
        if observables is not None:
            self.observables = observables
        self._set_defaults(new_defaults)
        self._set_data(data)

    def __call__(self, name=None, data=None, **new_defaults):
        """Return a model with possibly changed name, defaults, or data"""
        if name is None and data is None and not new_defaults:
            return self
        new_self = copy(self)
        if name is not None:
            new_self.name = name
        new_self._set_defaults(new_defaults)
        new_self._set_data(data)
        return new_self

    # Validation

    def validate_data(self, data: np.ndarray) -> np.ndarray:
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
        if cut is None:
            return cut
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

    def simulate(self, params: dict = None, *, cut=None) -> np.ndarray:
        params = self.validate_params(params)

        if self.simulator_gives_efficiency:
            mu = self.rate_before_efficiencies(params)
            n = np.random.poisson(mu)
            events, p_keep = self.simulate_partially_efficient(n, params, cut=cut)
            events = events[np.random.rand(n) < p_keep]
            return events

        else:
            mu = self.rate(params)
            n = np.random.poisson(mu)
            data = self.rvs(n, params)
            assert len(data) == n
            data = self.cut(data, cut)

        return data

    # Methods taking data
    # _ methods do not take data

    def diff_rate(
        self, params: dict = None, data: np.ndarray = None, *, cut=None
    ) -> np.ndarray:
        params = self.validate_params(params)
        return self(data=data)._diff_rate(params, cut=cut)

    def _diff_rate(self, params: dict, cut=None):
       return self._pdf(params=params) * self.rate(params, cut=cut)

    def cut(self, data=None, cut=None):
        cut = self.validate_cut(cut)
        return self(data=data)._cut(cut=cut)

    def _cut(self, cut):
        if cut is None:
            return self.data
        passed = np.ones(len(self.data), np.bool_)
        for dim_i, (l, r) in enumerate(cut):
            passed *= (l <= self.data[:, dim_i]) & (self.data[:, dim_i] < r)
        return self.data[passed]

    def pdf(self, params: dict = None, data: np.ndarray=None) -> np.ndarray:
        params = self.validate_params(params)
        return self(data=data)._pdf(params)

    def _pdf(self, params: dict):
        if self.__class__._diff_rate is Model._diff_rate:
            raise NotImplementedError(
                "Can't compute pdf of a Model implementing "
                "neither _pdf nor _diff_rate")
        return self._diff_rate(self.data, params) / self.rate(params)

    def cdf(self, params: dict = None, data: np.ndarray = None) -> np.ndarray:
        params = self.validate_params(params)
        return self(data=data)._cdf(params)

    def _cdf(self, params: dict):
        raise NotImplementedError

    # Methods not taking data

    def rate(self, params: dict = None, cut=None) -> np.ndarray:
        params = self.validate_params(params)
        return params["rate"] * self.cut_efficiency(cut, params)

    def rvs(self, n: int, params: dict = None) -> np.ndarray:
        raise NotImplementedError

    def cut_efficiency(self, cut=None, params: dict = None):
        params = self.validate_params(params)
        cut = self.validate_cut(cut)
        if cut is None:
            return 1.0
        if not hasattr(self, "cdf"):
            raise NotImplementedError("Nontrivial cuts require a cdf")
        # Evaluate CDF at rectangle endpoints, add up with alternating signs,
        # always + in upper right.
        # TODO: Not sure this is correct for n > 2!
        # (for n=3 looks OK, for higher n I can't draw/visualize)
        lower_sign = (-1) ** (len(cut))
        signs, points = zip(
            *[
                (
                    np.prod(indices),
                    [c[int(0.5 * j + 0.5)] for (c, j) in zip(cut, indices)],
                )
                for indices in itertools.product(
                    *[[lower_sign, -lower_sign]] * len(cut)
                )
            ]
        )
        return np.sum(np.array(signs) * self.cdf(params=params, data=points))

    def __add__(self, other):
        return hypney.Mixture(self, other)
