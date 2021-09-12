from types import DynamicClassAttribute
import typing as ty

import numpy as np

import hypney

export, __all__ = hypney.exporter()


@export
class Mixture(hypney.Model):
    models: ty.Tuple[hypney.Model]
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
            new_obs.append(hypney.Observable(name=obs_0.name, min=new_min, max=new_max))
        self.observables = tuple(new_obs)

        self.param_specs, self.param_mapping = hypney.combine_param_specs(
            models, self.model_names
        )

        super().__init__(name="mix_" + "_".join(self.model_names))

    def init_data(self):
        self.models = tuple([m(data=self.data) for m in self.models])
        super().init_data()

    def init_cut(self):
        self.models = tuple([m(cut=self.cut) for m in self.models])
        super().init_cut()

    def iter_models_params(self, params):
        for m, param_map in zip(self.models, self.param_mapping.values()):
            yield m, {
                pname_in_model: params[pname_in_mixture]
                for pname_in_model, pname_in_mixture in param_map
            }

    def rate_per_model(self, params: dict) -> np.ndarray:
        return np.array(
            [m._rate(ps) for m, ps in self.iter_models_params(params)]
        )

    def _rate(self, params: DynamicClassAttribute) -> np.ndarray:
        return sum(self.rate_per_model(params))

    def f_per_model(self, params):
        mus = self.rate_per_model(params)
        return mus / mus.sum()

    def _pdf(self, params: dict) -> np.ndarray:
        return np.average(
            [m._pdf(params=ps) for m, ps in self.iter_models_params(params)],
            axis=0,
            weights=self.f_per_model(params),
        )

    def _cdf(self, params: dict) -> np.ndarray:
        return np.average(
            [m._cdf(params=ps) for m, ps in self.iter_models_params(params)],
            axis=0,
            weights=self.f_per_model(params),
        )

    def _diff_rate(self, params: dict) -> np.ndarray:
        return np.sum(
            [m._diff_rate(params=ps) for m, ps in self.iter_models_params(params)],
            axis=0,
        )

    def simulate(self, params: dict, *, cut=hypney.NotChanged) -> np.ndarray:
        params = self.validate_params(params)
        return np.concatenate(
            [
                m(cut=cut).simulate(params=ps)
                for m, ps in self.iter_models_params(params)
            ],
            axis=0,
        )

    def _rvs(self, params: dict, size: int = 1) -> np.ndarray:
        n_from = np.random.multinomial(size, self.f_per_model(params))
        return np.concatenate(
            [
                m._rvs(params=ps, size=_n)
                for _n, (m, ps) in zip(n_from, self.iter_models_params(params))
            ]
        )
