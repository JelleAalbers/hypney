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
        """Return a new model with different defaults"""
        params = self.validate_params(params)
        return self.__class__(name=self.name, **params)

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
            data = self.simulate_n(n, params)
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
        return Mixture(self, other)

    # Model should implement one of pdf or diff_rate, else recursion error
    def pdf(self, data: np.ndarray, params: dict = None) -> np.ndarray:
        data = self.validate_data(data)
        params = self.validate_params(params)
        return self.diff_rate(data, params) / self.rate(params)

    def rate(self, params: dict = None, cut=None) -> np.ndarray:
        params = self.validate_params(params)
        return params["rate"] * self.cut_efficiency(cut, params)

    def simulate_n(self, n: int, params: dict = None) -> np.ndarray:
        raise NotImplementedError


Model.differential_rate = Model.diff_rate
Model.cut_eff = Model.cut_efficiency


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

        self.param_specs, self.param_mapping = hypney.combine_param_specs(
            models, self.model_names
        )

        super().__init__(name="mix_" + "_".join(self.model_names))

    def iter_models_params(self, params):
        for m, param_map in zip(self.models, self.param_mapping.values()):
            yield m, {
                pname_in_model: params[pname_in_mixture]
                for pname_in_model, pname_in_mixture in param_map
            }

    def rate_per_model(self, params: dict = None, *, cut=None) -> np.ndarray:
        params = self.validate_params(params)
        return np.array(
            [m.rate(ps, cut=cut) for m, ps in self.iter_models_params(params)]
        )

    def rate(self, params: dict = None, *, cut=None) -> np.ndarray:
        return sum(self.rate_per_model(params, cut=cut))

    def f_per_model(self, params):
        mus = self.rate_per_model(params)
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

    def simulate_n(self, n: int, params: dict = None) -> np.ndarray:
        params = self.validate_params(params)
        n_from = np.random.multinomial(n, self.f_per_model(params))
        return np.concatenate(
            [
                m.simulate_n(_n, params=ps)
                for _n, (m, ps) in zip(n_from, self.iter_models_params(params))
            ]
        )
