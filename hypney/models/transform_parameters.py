import hypney

export, __all__ = hypney.exporter()


@export
class TransformParameters(hypney.Model):
    """Give new parameters to a model, given a function
    mapping the new to the old parameters.
    """

    _wrapped_model: hypney.Model

    def __init__(
        self,
        *args,
        wrapped_model=hypney.NotChanged,
        transform_params=hypney.NotChanged,
        **kwargs
    ):
        if transform_params is not hypney.NotChanged:
            self._transform_params = transform_params
        if wrapped_model is not hypney.NotChanged:
            self._wrapped_model = wrapped_model
        super().__init__(*args, observables=self._wrapped_model.observables, **kwargs)

    def validate_params(self, params: dict, set_defaults) -> dict:
        # Validate parameters in our own / the new parameter spec
        params = super().validate_params(params, set_defaults=set_defaults)
        # Transform params, validate in the original model
        params_orig = self._transform_params(params)
        return self._wrapped_model.validate_params(params_orig)

    def _transform_params(self, params: dict):
        raise NotImplementedError


@export
def transform_parameters(model: hypney.Model, transform_f: callable):
    return TransformParameters(
        wrapped_model=model,
        transform_params=transform_f,
        param_specs=model.param_specs,
    )
