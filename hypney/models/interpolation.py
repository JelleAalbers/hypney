from functools import partial, partialmethod
import itertools
import numpy as np
import typing as ty

import eagerpy as ep

import hypney

export, __all__ = hypney.exporter()


@export
class Interpolation(hypney.Model):
    """Model which interpolates other models, depending on parameters.

    The pdf, cdf and ppf are linearly interpolated. You should only
    use one of these; the other two will generally be inconsistent!
    """

    data_methods_to_interpolate = "pdf logpdf cdf diff_rate".split()
    other_methods_to_interpolate = "rate mean std".split()
    anchor_models: ty.Dict[tuple, hypney.Model]

    @property
    def interpolated_params(self):
        return [p for p in self.param_specs if p.anchors]

    def __init__(
        self,
        # Called with params (dict of scalars), outputs model
        model_builder: callable,
        param_specs: ty.Union[tuple, dict],
        methods: ty.Tuple,
        progress=False,
        map=map,
        *args,
        **kwargs
    ):
        if isinstance(methods, str):
            methods = (methods,)
        self.methods = methods
        if isinstance(param_specs, dict):
            # Shorthand parameter spec just for this model,
            # only anchors given.
            param_specs = tuple(
                [
                    hypney.Parameter(
                        name=pname,
                        # Use (left) middle anchor as the default
                        default=anchors[(len(anchors) - 1) // 2],
                        anchors=anchors,
                    )
                    for pname, anchors in param_specs.items()
                ]
            )

        interpolated_params = [p for p in param_specs if p.anchors]
        param_names = [p.name for p in interpolated_params]
        anchor_values = [p.anchors for p in interpolated_params]

        # List of all possible anchor vals (long 1d list of parameter tuples)
        anchor_grid = list(itertools.product(*anchor_values))
        # List of possible anchor params (long 1d list of param dicts)
        anchor_grid_kwargs = [
            dict(zip(param_names, anchor_vals)) for anchor_vals in anchor_grid
        ]

        progress_iter = hypney.utils.misc.progress_iter(
            progress, desc="Building models", total=len(anchor_grid)
        )
        self.anchor_models = dict(
            zip(anchor_grid, progress_iter(map(model_builder, anchor_grid_kwargs)))
        )

        self.interp_maker = hypney.utils.interpolation.InterpolatorBuilder(
            anchor_values
        )
        self._interpolators = dict()

        # Add specs of any non-interpolated params
        # TODO: this is tricky. I guess idea is allowing rate as a parameter
        # in a shape interpolation. Works if we interpolate pdf, not diff_rate.
        # In general, shouldn't other params be forced at constants?
        self._some_model = next(iter(self.anchor_models.values()))
        param_specs = param_specs + tuple(
            [
                p
                for p in self._some_model.param_specs
                if p.name not in [q.name for q in param_specs]
            ]
        )

        # Build interpolators for methods that don't take data
        for method_name in self.other_methods_to_interpolate:
            self._maybe_build_interpolator(
                method_name, tensorlib=self._some_model.backend
            )

        # Can't call this earlier; may trigger interpolator building
        # if data is given on init
        super().__init__(
            *args,
            param_specs=param_specs,
            observables=self._some_model.observables,
            **kwargs
        )

    def _init_data(self):
        # Replace anchor models with ones with data set
        self.anchor_models = {
            anchor: model(data=self.data)
            for anchor, model in self.anchor_models.items()
        }

        # Build interpolators for methods that take data (i.e. use self.data)
        for method_name in self.data_methods_to_interpolate:
            # Only build interpolator if method was redefined in anchor models
            # (e.g. if anchors define _pdf, don't interpolate _diff_rate too)
            if self._some_model._has_redefined("_" + method_name):
                self._maybe_build_interpolator(
                    method_name, tensorlib=self._some_model.backend
                )

        super()._init_data()

    def _init_quantiles(self):
        self.anchor_models = {
            anchor: model(quantiles=self.quantiles)
            for anchor, model in self.anchor_models.items()
        }
        if self._has_redefined("_ppf"):
            self._maybe_build_interpolator("ppf", tensorlib=self._some_model.backend)

    def _maybe_build_interpolator(self, itp_name: str, tensorlib):
        if itp_name not in self.methods:
            # Shouldn't / don't have to interpolate this method
            return
        self._interpolators[itp_name] = self.interp_maker.make_interpolator(
            # itp_name=itp_name does not work! Confusing...
            partial(self._call_anchor_method, itp_name),
            tensorlib=tensorlib,
        )

    def _params_to_anchor_tensor(self, params):
        """Return (batch_shape, n_params) tensor given dict of param tensors
        """
        return ep.stack(
            [params[p.name] for p in self.param_specs if p.anchors], axis=-1
        )

    def _call_interpolator(self, itp_name, params: dict = None):
        if not itp_name in self._interpolators:
            # No interpolator was built, call a fallback method
            # (e.g. Model._log_differential_rate, which will may call _pdf
            #  for which we may have an interpolator)
            model = super()
            return getattr(model, "_" + itp_name)(params)

        # Stack parameters to make
        # (batch_shape, n_params) tensor
        anchor_tensor = self._params_to_anchor_tensor(params)

        # Call interpolator to get result
        # not sure if [0] undoes 'expanded' batch shape or some
        # interpolator convention... but vectorization tests pass
        result = self._interpolators[itp_name](anchor_tensor)[0]
        return result

    def _params_to_anchor_tuple(self, params):
        # TODO: this will fail for non-trivial batch_shape!
        return tuple([params[p.name].raw.item() for p in self.param_specs if p.anchors])

    def _call_anchor_method(self, method_name, param_tuple):
        """Call Model.method_name for anchor model at params"""
        # Which anchor to call?
        anchor_model = self.anchor_models[param_tuple]

        # Call appropriate internal method with valid param dict
        params = anchor_model.validate_params(
            {
                p.name: v
                for p, v in zip(self.interpolated_params, param_tuple)
                # TODO: Test below needed since param may already be frozen?
                if p.name in anchor_model.defaults
            }
        )
        return getattr(anchor_model, "_" + method_name)(params=params)

    def _rvs(self, size: int, params: dict):
        anchor = self._params_to_anchor_tuple(params)
        if anchor not in self.anchor_models:
            # Dig into interpolator to get weight for each anchor,
            # then mix rvs call with different weights?
            # Sounds tricky.
            raise NotImplementedError("Can only simulate at anchor models")
        return self.anchor_models[anchor]._rvs(size=size)

    def _simulate(self, params: dict = None):
        anchor = self._params_to_anchor_tuple(params)
        if anchor not in self.anchor_models:
            raise NotImplementedError("Can only simulate at anchor models")
        return self.anchor_models[anchor]._simulate(params)


for method_name in (
    Interpolation.data_methods_to_interpolate
    + Interpolation.other_methods_to_interpolate
    + ["ppf",]
):
    setattr(
        Interpolation,
        "_" + method_name,
        partialmethod(Interpolation._call_interpolator, method_name),
    )
del method_name
