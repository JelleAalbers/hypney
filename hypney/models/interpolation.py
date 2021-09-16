from functools import partial, partialmethod
import itertools
import typing as ty

import numpy as np

import hypney

export, __all__ = hypney.exporter()


@export
class Interpolation(hypney.Model):
    """Model which interpolates between other models, depending on parameters.
        The pdf, cdf, rate, etc. are interpolators, evaluated at anchor points
    """

    data_methods_to_interpolate = "_pdf _cdf _diff_rate".split()
    other_methods_to_interpolate = "_rate ".split()

    def __init__(
        self,
        # Called with params, outputs model
        model_builder: callable,
        param_specs: ty.Union[tuple, dict],
        *args,
        **kwargs
    ):
        if isinstance(param_specs, dict):
            # Shorthand parameter spec, only anchors given.
            param_specs = tuple(
                [
                    hypney.ParameterSpec(
                        name=pname,
                        # Use (left) middle anchor as the default
                        default=anchors[(len(anchors) - 1) // 2],
                        anchors=anchors,
                    )
                    for pname, anchors in param_specs.items()
                ]
            )

        # Create the grid of anchor models
        # TODO: support lazy map / multiprocessing? Progress bar?
        # Star morphing?
        # Some way for user to extend this?
        param_names = [p.name for p in param_specs]
        anchor_values = [p.anchors for p in param_specs if p.anchors]
        self.anchor_models = {
            anchor_vals: model_builder(dict(zip(param_names, anchor_vals)))
            for anchor_vals in itertools.product(*anchor_values)
        }
        self._some_model = next(iter(self.anchor_models.values()))

        self.interp_maker = hypney.GridInterpolator(anchor_values)
        self._interpolators = dict()

        # Add specs of any non-interpolated params
        # TODO: check other models agree on these?
        param_specs = param_specs + tuple(
            [
                p
                for p in self._some_model.param_specs
                if p.name not in [q.name for q in param_specs]
            ]
        )

        # Can't call this earlier; may trigger interpolator building
        # if data is given on init
        super().__init__(
            *args,
            param_specs=param_specs,
            observables=self._some_model.observables,
            **kwargs
        )

        # For methods that don't take data, we can build the interpolators
        # right now
        for method_name in self.other_methods_to_interpolate:
            if hasattr(self._some_model, method_name):
                self._build_interpolator(method_name)

    def _init_data(self):
        # Update anchor models to ones with appropriate data & cut
        self.anchor_models = {
            anchor: model(data=self.data, cut=self.cut)
            for anchor, model in self.anchor_models.items()
        }

        # Build interpolators for methods that take data (i.e. use self.data)
        for method_name in self.data_methods_to_interpolate:
            # Only build interpolator if method was redefined from Model
            if self._has_redefined(method_name):
                self._build_interpolator(method_name)

        super()._init_data()

    def _init_cut(self):
        if self.data is not None:
            self._init_data()
        super()._init_cut()

    def _build_interpolator(self, itp_name: str):
        # Make sure to call non-underscored methods of the anchor models,
        # so they fill in their default params.
        # (especially convenient for non-interpolated params. Otherwise we'd
        #  need quite some complexity in _call_anchor_method / _params_to_anchor_tuple)
        method_name = itp_name[1:] if itp_name.startswith("_") else itp_name

        self._interpolators[itp_name] = self.interp_maker.make_interpolator(
            # method_name=method_name does not work! Confusing...
            partial(self._call_anchor_method, method_name)
        )

    def _call_interpolator(
        self, itp_name, params: dict = None, *, cut=hypney.NotChanged
    ):
        if cut is not hypney.NotChanged:
            self = self(cut=cut)
        if not itp_name in self._interpolators:
            # No interpolator was built e.g. diff_rate when pdf and rate known.
            return getattr(super(), itp_name)(params)
        params = self.validate_params(params)
        return self._interpolators[itp_name]([self._params_to_anchor_tuple(params)])[0]

    def _params_to_anchor_tuple(self, params):
        return tuple([params[p.name] for p in self.param_specs if p.anchors])

    def _call_anchor_method(self, method_name, param_tuple):
        """Call Model.method_name for anchor model at params"""
        return getattr(self.anchor_models[param_tuple], method_name)()

    def _rvs(self, params: dict = None, size: int = 1) -> np.ndarray:
        anchor = self._params_to_anchor_tuple(params)
        if anchor not in self.anchor_models:
            # Dig into interpolator to get weight for each anchor,
            # then mix rvs call with different weights?
            # Sounds tricky.
            raise NotImplementedError("Can only simulate at anchor models")
        return self.anchor_models[anchor]._rvs(size=size)

    def _simulate(self, params: dict = None) -> np.ndarray:
        anchor = self._params_to_anchor_tuple(params)
        if anchor not in self.anchor_models:
            raise NotImplementedError("Can only simulate at anchor models")
        return self.anchor_models[anchor]._simulate(params)


for method_name in (
    Interpolation.data_methods_to_interpolate
    + Interpolation.other_methods_to_interpolate
):
    setattr(
        Interpolation,
        method_name,
        partialmethod(Interpolation._call_interpolator, method_name),
    )
