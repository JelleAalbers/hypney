import typing as ty

import numpy as np

import hypney

export, __all__ = hypney.exporter()


class AssociativeCombination(hypney.Model):
    models: ty.Tuple[hypney.Model]
    model_names: ty.Tuple[str]

    # param_mapping   (mname -> (pname in model, pname in combination))
    param_mapping: ty.Dict[str, ty.Tuple[str, str]]

    def __init__(self, *models):
        assert len(models) > 1

        # Exploit associativity: if any of the models are combinations of
        # the same type, grab underlying models and combine them
        _models = []
        for m in models:
            if isinstance(m, self.__class__):
                _models.extend(m.models)
            else:
                _models.append(m)
        models = _models

        self.models = tuple(models)
        self.model_names = [m.name if m.name else f"m{i}" for i, m in enumerate(models)]

        self.observables = self._init_observables()

        self.param_specs, self.param_mapping = hypney.combine_param_specs(
            models, self.model_names
        )

        super().__init__(
            name=self.__class__.__name__ + "_" + "_".join(self.model_names)
        )

    def _iter_models_params(self, params):
        for m, param_map in zip(self.models, self.param_mapping.values()):
            yield m, {
                pname_in_model: params[pname_in_mixture]
                for pname_in_model, pname_in_mixture in param_map
            }


@export
class Mixture(AssociativeCombination):
    def _init_observables(self):
        assert all(
            [m.n_dim == self.models[0].n_dim for m in self.models]
        ), "Can't mix models of different dimensionality"
        new_obs = []
        for obs_i in range(self.models[0].n_dim):
            obs_0 = self.models[0].observables[obs_i]
            assert all(
                [m.observables[obs_i].name == obs_0.name for m in self.models]
            ), "Can't mix models with different observable names"
            new_min = min([m.observables[obs_i].min for m in self.models])
            new_max = max([m.observables[obs_i].max for m in self.models])
            new_obs.append(hypney.Observable(name=obs_0.name, min=new_min, max=new_max))
        return tuple(new_obs)

    def _init_cut(self):
        self.models = tuple([m(cut=self.cut) for m in self.models])
        super()._init_cut()

    def _init_data(self):
        self.models = tuple([m(data=self.data) for m in self.models])
        super()._init_data()

    ##
    # Simulation
    ##

    def _rvs(self, params: dict, size: int = 1) -> np.ndarray:
        n_from = np.random.multinomial(size, self._f_per_model(params))
        return np.concatenate(
            [
                m._rvs(params=ps, size=_n)
                for _n, (m, ps) in zip(n_from, self._iter_models_params(params))
            ]
        )

    ##
    # Main statistical methods
    ##

    def _rate(self, params: dict) -> np.ndarray:
        return sum(self._rate_per_model(params))

    def _pdf(self, params: dict) -> np.ndarray:
        return np.average(
            [m._pdf(params=ps) for m, ps in self._iter_models_params(params)],
            axis=0,
            weights=self._f_per_model(params),
        )

    def _cdf(self, params: dict) -> np.ndarray:
        return np.average(
            [m._cdf(params=ps) for m, ps in self._iter_models_params(params)],
            axis=0,
            weights=self._f_per_model(params),
        )

    def _diff_rate(self, params: dict) -> np.ndarray:
        return np.sum(
            [m._diff_rate(params=ps) for m, ps in self._iter_models_params(params)],
            axis=0,
        )

    ##
    # Helpers
    ##

    def _rate_per_model(self, params: dict) -> np.ndarray:
        return np.array([m._rate(ps) for m, ps in self._iter_models_params(params)])

    def _f_per_model(self, params):
        mus = self._rate_per_model(params)
        return mus / mus.sum()


@export
class TensorProduct(AssociativeCombination):
    def _init_observables(self):
        new_obs = []
        for m in self.models:
            new_obs.extend(list(m.observables))
        return tuple(new_obs)

    def _obs_splits(self):
        # TODO: check off by one
        return np.cumsum([len(m.observables) for m in self.models])

    def _init_data(self):
        _data_list = np.split(self.data, self._obs_splits(), axis=-1)
        self.models = tuple([m(data=d) for m, d in zip(self.models, _data_list)])
        super()._init_data()

    def _init_cut(self):
        _cut_list = np.split(self.cut, self._obs_splits())
        self.models = tuple([m(cut=c) for m, c in zip(self.models, _cut_list)])
        super()._init_cut()

    def _rvs(self, params: dict, size: int) -> np.ndarray:
        return np.concatenate(
            [m.rvs(params=ps, size=size) for m, ps in self._iter_models_params(params)],
            axis=-1,
        )

    def _rate(self, params: dict):
        # First model controls the rate
        # TODO: remove default rate param from later models
        m, ps = next(self._iter_models_params(params))
        return m._rate(ps)

    def _pdf(self, params: dict):
        return np.product(
            [m._pdf(ps) for m, ps in self._iter_models_params(params)], axis=0
        )

    def _cdf(self, params: dict):
        return np.product(
            [m._cdf(ps) for m, ps in self._iter_models_params(params)], axis=0
        )
