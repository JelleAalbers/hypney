import itertools
import typing as ty

import numpy as np

import hypney as hp


export, __all__ = hp.exporter()


class ParameterSpec(ty.NamedTuple):
    """Description of a parameter: name, default, and limits"""

    name: str
    default: float = 0.0
    min: float = -float("inf")
    max: float = float("inf")


class Observable(ty.NamedTuple):
    """Description of a observable space: name and limits"""

    name: str
    min: float = -float("inf")
    max: float = float("inf")


@export
class Model:
    param_specs: ty.Tuple[ParameterSpec]
    observables: ty.Tuple[Observable]

    @property
    def param_names(self):
        return tuple([p.name for p in self.param_specs])

    def __init__(self, **params):
        params = self.validate_params(params)
        self.param_specs = tuple(
            [
                ParameterSpec(p.name, default=params[p.name], min=p.min, max=p.max)
                for p in self.param_specs
            ]
        )

    def validate_params(self, params: dict) -> dict:
        if params is None:
            params = dict()
        for p in self.param_specs:
            params.setdefault(p.name, p.default)
        spurious = set(params.keys()) - set(self.param_names)
        if spurious:
            raise ValueError(f"Unknown parameters {spurious} passed")
        return params

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
            data = data[None, :]

        expected_dim = len(self.observables)
        observed_dim = data.shape[1]
        if expected_dim != observed_dim:
            raise ValueError(
                "Data should have {expected_dim} observables per event, got {observed_dim}"
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
        if isinstance(cut, tuple) and len(cut) == len(self.observables):
            if all([len(c) == 2 for c in cut]):
                return cut
            raise ValueError("Cut should be a tuple of 2-tuples")
        return cut

    @property
    def simulator_gives_efficiency(self):
        return hasattr(self, "simulate_partially_efficient") and hasattr(
            self, "expected_count_before_efficiencies"
        )

    def simulate(self, params: dict = None, *, cut=None) -> np.ndarray:
        params = self.validate_params(params)

        if self.simulator_gives_efficiency:
            mu = self.expected_count_before_efficiencies(params)
            n = np.random.poisson(mu)
            events, p_keep = self.simulate_partially_efficient(n, params, cut=cut)
            events = events[np.random.rand(n) < p_keep]
            return events

        else:
            mu = self.expected_count(params)
            n = np.random.poisson(mu)
            events = self.simulate_n(n, params, cut=cut)
            assert len(events) == n  # TODO: allow acceptance

        return events

    def diff_rate(self, data: np.ndarray, params=None, *, cut=None) -> np.ndarray:
        params = self.validate_params(params)
        data = self.validate_data(data)
        return self.pdf(data, params) * self.expected_count(params, cut=cut)

    def simulate_n(self, n: int, params: dict = None, *, cut=None) -> np.ndarray:
        params = self.validate_params(params)
        raise NotImplementedError

    def expected_count(self, params: dict = None, cut=None) -> np.ndarray:
        params = self.validate_params(params)
        raise NotImplementedError

    def pdf(self, data: np.ndarray, params: dict = None) -> np.ndarray:
        data = self.validate_data(data)
        params = self.validate_params(params)
        raise NotImplementedError

    def cut_efficiency(self, params: dict = None, cut=None):
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


@export
class Uniform(Model):
    observables = (Observable("x", 0, 1),)
    param_specs = (ParameterSpec("expected_events", 0),)

    def expected_count(self, params: dict = None, *, cut=None) -> np.ndarray:
        params = self.validate_params(params)
        return self.cut_efficiency(cut) * (params["expected_events"])

    def simulate_n(self, n: int, params: dict = None, *, cut=None) -> np.ndarray:
        params = self.validate_params(params)
        if cut is None:
            obs = self.observables[0]
            low, high = obs.min, obs.max
        else:
            cut = self.validate_cut(cut)
            low, high = cut[0]
        return (low + np.random.rand(n) * (high - low))[:, None]

    def pdf(self, data: np.ndarray, params: dict = None) -> np.ndarray:
        params = self.validate_params(params)
        data = self.validate_data(data)
        return np.ones(len(data))

    def cdf(self, data: np.ndarray, params: dict = None) -> np.ndarray:
        params = self.validate_params(params)
        params = self.validate_data(data)
        return data[:, 0]
