from functools import partial, partialmethod
import itertools
import typing as ty

import numpy as np

import hypney

export, __all__ = hypney.exporter()


@export
class Interpolation(hypney.Model):

    data_methods_to_interpolate = "_pdf _cdf _diff_rate".split()
    other_methods_to_interpolate = "rate mean std".split()

    def __init__(
        self,
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

    def init_data(self):
        # Update anchor models to ones with data
        self.anchor_models = {
            anchor: model(data=self.data)
            for anchor, model in self.anchor_models.items()
        }

        # Build interpolators for methods that take data (i.e. use self.data)
        for method_name in self.data_methods_to_interpolate:
            # Only build interpolator if method was redefined from Model
            # ()
            if getattr(self._some_model, method_name).__func__ != getattr(
                hypney.Model, method_name
            ):
                self._build_interpolator(method_name, method_name[1:])

    def _build_interpolator(self, itp_name, method_name=None):
        if method_name is None:
            method_name = itp_name
        self._interpolators[itp_name] = self.interp_maker.make_interpolator(
            # method_name=method_name does not work! Confusing...
            partial(self._anchor_method_getter, method_name)
        )

    def _call_interpolator(self, itp_name, params: dict = None, **kwargs):
        if not hasattr(self._some_model, itp_name):
            raise AttributeError
        params = self.validate_params(params)
        # Ensure params are ordered correctly
        params = np.stack([params[pname] for pname in self.defaults])
        return self._interpolators[itp_name](params)[0]

    def _param_tuple(self, params):
        return tuple([params[p.name] for p in self.param_specs])

    def _anchor_method_getter(self, method_name, param_tuple):
        """Call Model.method_name for anchor model at params
        """
        return getattr(self.anchor_models[param_tuple], method_name)()

    def rvs(self, params: dict = None, size: int = 1) -> np.ndarray:
        params = self.validate_params(params)
        anchor = self._param_tuple(params)
        if anchor not in self.anchor_models:
            # Dig into interpolator to get weight for each anchor,
            # then mix rvs call with different weights?
            # Sounds tricky.
            raise NotImplementedError("Can only simulate at anchor models")
        return self.anchor_models[anchor].rvs(size=size)


for method_name in (
    Interpolation.data_methods_to_interpolate
    + Interpolation.other_methods_to_interpolate
):
    setattr(
        Interpolation,
        method_name,
        partialmethod(Interpolation._call_interpolator, method_name),
    )
