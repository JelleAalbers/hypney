from copy import copy
import itertools
import typing as ty

import numpy as np

import hypney

export, __all__ = hypney.exporter()


@export
class Observable(ty.NamedTuple):
    """Description of a observable space: name and limits"""

    name: str
    min: float = -float("inf")
    max: float = float("inf")


@export
class Model(hypney.Element):

    observables: ty.Tuple[Observable] = (
        Observable(name="x", min=-float("inf"), max=float("inf")),
    )

    @property
    def n_dim(self):
        return len(self.observables)

    def __init__(self, name="", **params):
        self.name = name
        self._set_defaults(params)

    def _set_defaults(self, params):
        params = self.validate_params(params)
        self.param_specs = tuple(
            [
                hypney.ParameterSpec(
                    p.name, default=params[p.name], min=p.min, max=p.max
                )
                for p in self.param_specs
            ]
        )

    def __call__(self, **params):
        """Return a new model with different parameter defaults"""
        new_self = copy(self)
        new_self._set_defaults(params)
        return new_self

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

    def diff_rate(
        self, data: np.ndarray, params: dict = None, *, cut=None
    ) -> np.ndarray:
        params = self.validate_params(params)
        data = self.validate_data(data)
        return self.pdf(data, params) * self.rate(params, cut=cut)

    def cut(self, data, cut=None):
        cut = self.validate_cut(cut)
        data = self.validate_data(data)
        if cut is None:
            return data
        passed = np.ones(len(data), np.bool_)
        for dim_i, (l, r) in enumerate(cut):
            passed *= (l <= data[:, dim_i]) & (data[:, dim_i] < r)
        return data[passed]

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
        return np.sum(np.array(signs) * self.cdf(points, params))

    def __add__(self, other):
        return hypney.Mixture(self, other)

    # Model should implement one of pdf or diff_rate, else recursion error
    def pdf(self, data: np.ndarray, params: dict = None) -> np.ndarray:
        data = self.validate_data(data)
        params = self.validate_params(params)
        return self.diff_rate(data, params) / self.rate(params)

    def rate(self, params: dict = None, cut=None) -> np.ndarray:
        params = self.validate_params(params)
        return params["rate"] * self.cut_efficiency(cut, params)

    def rvs(self, n: int, params: dict = None) -> np.ndarray:
        raise NotImplementedError


Model.differential_rate = Model.diff_rate
Model.cut_eff = Model.cut_efficiency
