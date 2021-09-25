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

    data_methods_to_interpolate = "pdf cdf diff_rate".split()
    other_methods_to_interpolate = "rate mean std".split()
    anchor_models: ty.Dict[tuple, hypney.Model]

    def __init__(
        self,
        # Called with params, outputs model
        model_builder: callable,
        param_specs: ty.Union[tuple, dict],
        *args,
        standardize=False,
        **kwargs
    ):
        self.standardize = standardize
        if isinstance(param_specs, dict):
            # Shorthand parameter spec just for this model,
            # only anchors given.
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

        if self.standardize:
            # Build models taking standardized data
            self.rescaled_anchor_models = {
                anchor: hypney.models.NormalizedData(model)
                for anchor, model in self.anchor_models.items()
            }

        self.interp_maker = hypney.utils.interpolation.InterpolatorBuilder(
            anchor_values
        )
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

        # Build interpolators for methods that don't take data
        # (we will change tensorlib type as needed in init_data)
        for method_name in self.other_methods_to_interpolate:
            self._build_interpolator(method_name, tensorlib=ep.numpy)

        # Can't call this earlier; may trigger interpolator building
        # if data is given on init
        super().__init__(
            *args,
            param_specs=param_specs,
            observables=self._some_model.observables,
            **kwargs
        )

    def _init_data(self):
        # Update anchor models to ones with appropriate data
        # TODO: what about cut?
        self.anchor_models = {
            anchor: model(data=self.data)
            for anchor, model in self.anchor_models.items()
        }

        # Build interpolators for methods that take data (i.e. use self.data)
        tensorlib = hypney.utils.eagerpy.tensorlib(self.data)
        for method_name in self.data_methods_to_interpolate:
            # Only build interpolator if method was redefined from Model
            if self._has_redefined("_" + method_name):
                self._build_interpolator(method_name, tensorlib=tensorlib)

        # TODO: change tensorlib of non-data-method interpolate non-data interpolators
        for method_name in self.other_methods_to_interpolate:
            self._build_interpolator(method_name, tensorlib=tensorlib)

        super()._init_data()

    def _init_quantiles(self):
        self.anchor_models = {
            anchor: model(quantiles=self.quantiles)
            for anchor, model in self.anchor_models.items()
        }
        tensorlib = hypney.utils.eagerpy.tensorlib(self.quantiles)
        if self._has_redefined("_ppf"):
            self._build_interpolator("ppf", tensorlib=tensorlib)

    def _init_cut(self):
        if not isinstance(self.data, ep.NumPyTensor):
            raise NotImplementedError("Interpolation only works on numpy data")
        if self.data is not None:
            self._init_data()
        super()._init_cut()

    def _build_interpolator(self, itp_name: str, tensorlib):
        self._interpolators[itp_name] = self.interp_maker.make_interpolator(
            # itp_name=itp_name does not work! Confusing...
            partial(self._call_anchor_method, itp_name),
            tensorlib=tensorlib,
        )

    def _call_interpolator(
        self, itp_name, params: dict = None, *, cut=hypney.NotChanged
    ):
        if cut is not hypney.NotChanged:
            self = self(cut=cut)
        if not itp_name in self._interpolators:
            # No interpolator was built (e.g. diff_rate when pdf and rate known)
            return getattr(super(), "_" + itp_name)(params)
        params = self.validate_params(params)
        anchor_tuple = hypney.utils.eagerpy.sequence_to_tensor(
            self._params_to_anchor_tuple(params),
            match_type=self.data if self.data is not None else np.zeros(0),
        )

        result = self._interpolators[itp_name](anchor_tuple[None, :])[0]
        if itp_name in self.data_methods_to_interpolate or itp_name == "ppf":
            # Vector result
            return ep.astensor(result)
        else:
            # Scalar result
            return result

    def _params_to_anchor_tuple(self, params):
        return tuple([params[p.name] for p in self.param_specs if p.anchors])

    def _call_anchor_method(self, method_name, param_tuple):
        """Call Model.method_name for anchor model at params"""
        anchor_models = (
            self.rescaled_anchor_models if self.standardize else self.anchor_models
        )
        # Make sure to call non-underscored methods of the anchor models,
        # so they fill in their default params.
        # (especially convenient for non-interpolated params. Otherwise we'd
        #  need quite some complexity in _call_anchor_method / _params_to_anchor_tuple)

        # We do have to call method_name + _, since we want to preserve eagerpy
        if method_name in self.data_methods_to_interpolate or method_name == "ppf":
            method_name = method_name + "_"
        return getattr(anchor_models[param_tuple], method_name)()

    def _rvs(self, size: int, params: dict) -> ep.TensorType:
        anchor = self._params_to_anchor_tuple(params)
        if anchor not in self.anchor_models:
            # Dig into interpolator to get weight for each anchor,
            # then mix rvs call with different weights?
            # Sounds tricky.
            raise NotImplementedError("Can only simulate at anchor models")
        return self.anchor_models[anchor]._rvs(size=size)

    def _simulate(self, params: dict = None) -> ep.TensorType:
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
        # See command in _call_anchor_method about underscore
        "_" + method_name,
        partialmethod(Interpolation._call_interpolator, method_name),
    )
del method_name
