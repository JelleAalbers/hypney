import collections
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
    share = False


class Observable(ty.NamedTuple):
    """Description of a observable space: name and limits"""

    name: str
    min: float = -float("inf")
    max: float = float("inf")


@export
class Model:
    name: str = ""
    param_specs: ty.Tuple[ParameterSpec]
    observables: ty.Tuple[Observable]

    @property
    def param_names(self):
        return tuple([p.name for p in self.param_specs])

    @property
    def n_dim(self):
        return len(self.observables)

    def __init__(self, name="", **params):
        self.name = name

        # TODO: should we store defaults separately? might be easier...
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
        if not isinstance(params, dict):
            raise ValueError(f"Params must be a dict, got {type(params)}")
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

    def diff_rate(
        self, data: np.ndarray, params: dict = None, *, cut=None
    ) -> np.ndarray:
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

    def cut_efficiency(self, cut, params: dict = None):
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
        return Mixture(self, other)


@export
class Mixture(Model):
    models: ty.Tuple[Model]
    model_names: ty.Tuple[str]
    weights: np.ndarray

    # param_mapping   (mname -> (pname in model, pname in mixture))
    param_mapping: ty.Dict[str, ty.Tuple[str, str]]

    def __init__(self, *models):
        assert len(models) > 1

        # If any of the models are mixtures, grab underlying models
        _models = []
        for m in models:
            if isinstance(m, Mixture):
                _models.extend(m.models)
            else:
                _models.append(m)
        models = _models

        self.models = tuple(models)
        self.model_names = [m.name if m.name else f"m{i}" for i, m in enumerate(models)]

        # Construct new domain / support / observables.
        assert all(
            [m.n_dim == models[0].n_dim for m in models]
        ), "Can't mix models of different dimensionality"
        new_obs = []
        for obs_i in range(models[0].n_dim):
            obs_0 = models[0].observables[obs_i]
            assert all(
                [m.observables[obs_i].name == obs_0.name for m in models]
            ), "Can't mix models with different observable names"
            new_min = min([m.observables[obs_i].min for m in models])
            new_max = max([m.observables[obs_i].max for m in models])
            new_obs.append(Observable(name=obs_0.name, min=new_min, max=new_max))
        self.observables = tuple(new_obs)

        # Construct parameter spec of mixture.
        # Clashing unshared parameter names are renamed modelname_paramname.
        # For shared params, defaults and bounds are taken from
        # the earliest model in the mixture.
        all_names = sum([list(m.param_names) for m in models], [])
        name_count = collections.Counter(all_names)
        unique = [pn for pn, count in name_count.items() if count == 1]
        specs = []
        pmap = dict()
        seen = []
        for m, mname in zip(self.models, self.model_names):
            pmap[mname] = []
            for p in m.param_specs:
                if p.name in unique or p.share:
                    pmap[mname].append((p.name, p.name))
                    if p.name not in seen:
                        specs.append(p)
                        seen.append(p.name)
                else:
                    new_name = mname + "_" + p.name
                    pmap[mname].append((p.name, new_name))
                    specs.append(
                        ParameterSpec(
                            name=new_name, min=p.min, max=p.max, default=p.default
                        )
                    )
        self.param_mapping = pmap
        self.param_specs = tuple(specs)

        super().__init__()

    def iter_models_params(self, params):
        for m, param_map in zip(self.models, self.param_mapping.values()):
            yield m, {
                pname_in_model: params[pname_in_mixture]
                for pname_in_model, pname_in_mixture in param_map
            }

    def expected_count_per_model(self, params: dict = None, *, cut=None) -> np.ndarray:
        params = self.validate_params(params)
        return np.array(
            [m.expected_count(ps, cut=cut) for m, ps in self.iter_models_params(params)]
        )

    def expected_count(self, params: dict = None, *, cut=None) -> np.ndarray:
        return sum(self.expected_count_per_model(params, cut=cut))

    def f_per_model(self, params):
        mus = self.expected_count_per_model(params)
        return mus / mus.sum()

    def pdf(self, data: np.ndarray, params: dict = None) -> np.ndarray:
        params = self.validate_params(params)
        return np.average(
            [m.pdf(data, ps) for m, ps in self.iter_models_params(params)],
            axis=0,
            weights=self.f_per_model(params),
        )

    def cdf(self, data: np.ndarray, params: dict = None) -> np.ndarray:
        # TODO: check.. and we're duplicating most of pdf...
        params = self.validate_params(params)
        return np.average(
            [m.cdf(data, ps) for m, ps in self.iter_models_params(params)],
            axis=0,
            weights=self.f_per_model(params),
        )

    def diff_rate(
        self, data: np.ndarray, params: dict = None, *, cut=None
    ) -> np.ndarray:
        params = self.validate_params(params)
        return np.sum(
            [
                m.diff_rate(data, ps, cut=cut)
                for m, ps in self.iter_models_params(params)
            ],
            axis=0,
        )

    def simulate(self, params: dict = None, *, cut=None) -> np.ndarray:
        params = self.validate_params(params)
        return np.concatenate(
            [
                m.simulate(params=ps, cut=cut)
                for m, ps in self.iter_models_params(params)
            ],
            axis=0,
        )


@export
class Uniform(Model):
    observables = (Observable("x", 0, 1),)
    param_specs = (ParameterSpec("expected_events", 0),)

    def expected_count(self, params: dict = None, *, cut=None) -> np.ndarray:
        params = self.validate_params(params)
        return self.cut_efficiency(cut, params) * (params["expected_events"])

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
        data = self.validate_data(data)
        return data[:, 0]
