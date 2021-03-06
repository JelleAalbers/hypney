import itertools
import collections
import typing as ty

import eagerpy as ep
import numpy as np

import hypney

export, __all__ = hypney.exporter()


class AssociativeCombination(hypney.Model):
    """Model formed from other models using some associative operation"""

    models: ty.Tuple[hypney.Model]
    model_names: ty.Tuple[str]

    # param_mapping   (mname -> (pname in model, pname in combination))
    param_mapping: ty.Dict[str, ty.Tuple[str, str]]

    def __init__(self, models: hypney.Model, share=tuple(), **kwargs):
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

        self.param_specs, self.param_mapping = combine_param_specs(
            models, self.model_names, share=share
        )

        super().__init__(
            name=self.__class__.__name__ + "_" + "_".join(self.model_names), **kwargs
        )

    def _init_observables(self):
        raise NotImplementedError

    def _iter_models_params(self, params):
        for m, param_map in zip(self.models, self.param_mapping.values()):
            yield m, {
                pname_in_model: params[pname_in_mixture]
                for pname_in_model, pname_in_mixture in param_map
            }

    @staticmethod
    def stack_axis0(xs):
        """Stack list of results from low-level methods along axis=0

        xs are broadcasted to whichever has the largest number of dimensions.

        (One model may use tensor-valued params, while another has no params;
         this causes some elements to have/lack extra batch_size dimensions)

        TODO: this needs more testing, plenty of edge cases possible...
        """
        argmax = max(range(len(xs)), key=lambda i: len(xs[i].shape))
        y = [hypney.utils.eagerpy.broadcast_to(x, xs[argmax].shape) for x in xs]
        return ep.stack(y, axis=0)


@export
class Mixture(AssociativeCombination):
    """Model that is a mixture of other models;
    that is, events from all constituent models are observed simultaneously.
    """

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
            new_obs.append(obs_0._replace(name=obs_0.name, min=new_min, max=new_max))
        return tuple(new_obs)

    def _init_data(self):
        self.models = tuple([m(data=self.data) for m in self.models])
        super()._init_data()

    def _rvs(self, size: int, params: dict) -> np.ndarray:
        if size == 0:
            return np.zeros((0, len(self.observables)))
        n_from = np.random.multinomial(size, self._f_per_model(params).numpy())
        result = np.concatenate(
            [
                m._rvs(size=_n, params=ps)
                for _n, (m, ps) in zip(n_from, self._iter_models_params(params))
            ]
        )
        # Ensure event sources are mixed, do not just put all of the first
        # in front of the array.
        np.random.shuffle(result)
        return result

    ##
    # Methods using data / quantiles
    ##

    # TODO: logpdf

    def _pdf(self, params: dict) -> ep.TensorType:
        # (n_models, {batch_shape}, n_events)
        # _pdf eats expanded batch shape
        pdf_per_model = self.stack_axis0(
            [m._pdf(params=ps) for m, ps in self._iter_models_params(params)],
        )
        return hypney.utils.eagerpy.average(
            pdf_per_model,
            # _f_per_model is (n_models, {expanded batch_shape})
            self._f_per_model(params),
            axis=0,
        )

    def _cdf(self, params: dict) -> ep.TensorType:
        # (n_models, {batch_shape}, n_events)
        cdf_per_model = self.stack_axis0(
            [m._cdf(params=ps) for m, ps in self._iter_models_params(params)],
        )
        return hypney.utils.eagerpy.average(
            cdf_per_model, weights=self._f_per_model(params), axis=0
        )

    def _diff_rate(self, params: dict) -> ep.TensorType:
        return ep.sum(
            self.stack_axis0(
                [m._diff_rate(params=ps) for m, ps in self._iter_models_params(params)]
            ),
            axis=0,
        )

    def _ppf(self, params: dict) -> ep.TensorType:
        raise NotImplementedError("Have to look up how to do this")

    ##
    # Methods not using data
    ##

    def _rate(self, params: dict) -> ep.TensorType:
        return ep.sum(self._rate_per_model(params), axis=0)

    def _mean(self, params: dict) -> ep.TensorType:
        # Average of the means
        means = self._mean_per_model(params)
        return ep.sum(means * self._f_per_model(params), axis=0)

    def _std(self, params: dict) -> ep.TensorType:
        means = self._mean_per_model(params)
        s2s = self._var_per_model(params)
        ps = self._f_per_model(params)
        # See e.g. https://stats.stackexchange.com/a/16609
        var = ep.sum(ps * (s2s + means ** 2), axis=0) - ep.sum(ps * means, axis=0) ** 2
        return var ** 0.5

    def _min(self, params):
        supports = self._support_per_model(params)
        return supports[:, 0, ...].min(axis=0)

    def _max(self, params):
        supports = self._support_per_model(params)
        return supports[:, 1, ...].max(axis=0)

    ##
    # Helpers
    ##

    def _rate_per_model(self, params: dict) -> ep.TensorType:
        # This will give (n_models, {expanded_batch_shape})
        # (assuming params is in expanded batch shape)
        return self.stack_axis0(
            [m._rate(ps) for m, ps in self._iter_models_params(params)]
        )

    def _mean_per_model(self, params: dict) -> ep.TensorType:
        return self.stack_axis0(
            [m._mean(ps) for m, ps in self._iter_models_params(params)]
        )

    def _var_per_model(self, params: dict) -> ep.TensorType:
        return self.stack_axis0(
            [m._var(ps) for m, ps in self._iter_models_params(params)]
        )

    def _f_per_model(self, params):
        mus = self._rate_per_model(params)
        return mus / mus.sum(axis=0)

    def _support_per_model(self, params: dict):
        # (n_models, 2, {expanded_batch_shape})
        assert self.n_dim == 1, "Support methods work for univariate dists"
        return self.stack_axis0(
            [
                ep.stack([m._min(params=ps), m._max(params=ps)], axis=0)
                for m, ps in self._iter_models_params(params)
            ]
        )


@export
class TensorProduct(AssociativeCombination):
    """Model for which constituent models describe independent observables
    observed simultaneously for each event.
    (e.g. one model for energy, another for time)
    The first model will control the overall event rate.
    """

    def _init_observables(self):
        new_obs = []
        for m in self.models:
            new_obs.extend(list(m.observables))
        return tuple(new_obs)

    def _obs_splits(self):
        # TODO: check off by one
        # cumsum would have been easier, but we don't know the tensor type here
        return list(itertools.accumulate([len(m.observables) for m in self.models]))

    def _init_data(self):
        _data_list = hypney.utils.eagerpy.split(self.data, self._obs_splits(), axis=-1)
        self.models = tuple([m(data=d) for m, d in zip(self.models, _data_list)])
        super()._init_data()

    def _rvs(self, size: int, params: dict) -> np.ndarray:
        return np.concatenate(
            [
                m._rvs(size=size, params=ps)
                for m, ps in self._iter_models_params(params)
            ],
            axis=-1,
        )

    def _rate(self, params: dict):
        # First model controls the rate
        # TODO: remove default rate param from later models
        m, ps = next(self._iter_models_params(params))
        return m._rate(ps)

    def _logpdf(self, params: dict):
        return ep.sum(
            self.stack_axis0(
                [m._logpdf(ps) for m, ps in self._iter_models_params(params)]
            )
        )

    def _pdf(self, params: dict):
        return ep.prod(
            self.stack_axis0([m._pdf(ps) for m, ps in self._iter_models_params(params)])
        )

    def _cdf(self, params: dict):
        return ep.prod(
            self.stack_axis0([m._cdf(ps) for m, ps in self._iter_models_params(params)])
        )

    def _ppf(self, params: dict) -> ep.TensorType:
        raise NotImplementedError("I think ppf isn't uniquely defined in >1 dim")


@export
def mixture(*models, **kwargs):
    return Mixture(models, **kwargs)


@export
def combine_param_specs(
    elements: ty.Sequence[hypney.Model], names=None, share=tuple(),
):
    """Return param spec, mapping for new model made of other models
    Mapping is name -> (old name, new name)

    Clashing unshared parameter names are renamed elementname_paramname
    For shared params, defaults and bounds are taken from
    the earliest model in the combination
    """
    if names is None:
        names = [e.name if e.name else str(i) for i, e in enumerate(elements)]
    if isinstance(share, str):
        share = (share,)
    all_names = sum([list(m.param_names) for m in elements], [])
    name_count = collections.Counter(all_names)
    unique = [pn for pn, count in name_count.items() if count == 1]
    specs = []
    pmap: ty.Dict[str, list] = dict()
    seen = []
    for m, name in zip(elements, names):
        pmap[name] = []
        for p in m.param_specs:
            if p.name in unique or p.name in share:
                pmap[name].append((p.name, p.name))
                if p.name not in seen:
                    specs.append(p)
                    seen.append(p.name)
            else:
                new_name = name + "_" + p.name
                pmap[name].append((p.name, new_name))
                specs.append(
                    hypney.Parameter(
                        name=new_name, min=p.min, max=p.max, default=p.default
                    )
                )
    return tuple(specs), pmap


@export
def tensor_product(*models, **kwargs):
    return TensorProduct(models, **kwargs)
