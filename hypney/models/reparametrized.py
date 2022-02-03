from math import prod
from functools import wraps
from types import MethodType

import hypney

export, __all__ = hypney.exporter()


@export
class Reparametrized(hypney.WrappedModel):
    """A model which transforms parameters, then feeds them to another model

    Args (beyond those of Model):
     - orig_model: original model taking transformed parameters
     - transform_params: function mapping dict of params the new model takes
        to dict of params the old model takes
    """

    def _transform_params(self, params: dict):
        return params

    # Initialization

    def __init__(self, *args, transform_params=hypney.NotChanged, **kwargs):
        if transform_params is not hypney.NotChanged:
            self._transform_params = transform_params
        super().__init__(*args, **kwargs)

    # Simulation
    # these are not vectorized, no need to restore batch shape.

    def _simulate(self, params):
        return self._orig_model._simulate(self._transform_params(params))

    def _rvs(self, size: int, params: dict):
        return self._orig_model._rvs(size=size, params=self._transform_params(params))


def _wrapped_method(method_name):
    @wraps(method_name)
    def wrapped(self, params):
        # Just to avoid confusion
        old_params = params
        del params

        method = getattr(self._orig_model, method_name)

        # Get new parameters in common shape
        new_params = self._transform_params(old_params)
        new_params = self._to_common_shape(new_params)

        result = method(new_params)

        # Enforce the old params' batch shape on the result
        new_batch_shape = self._batch_shape(new_params)
        old_batch_shape = self._batch_shape(old_params)
        if new_batch_shape == old_batch_shape:
            # Already good, usual case
            pass
        elif not new_batch_shape or [x == 1 for x in new_batch_shape]:
            # This happens if transform fills in scalar-ish defaults

            # Remove 1s from new batch axes
            for _ in len(new_batch_shape):
                result = result[0]
            base_shape = result.shape

            # Repeat along old batch axes
            result = result.tile(prod(old_batch_shape))
            result.reshape(list(old_batch_shape) + list(base_shape))
        else:
            raise NotImplementedError(
                f"Don't know how to impose old batch shape {old_batch_shape} "
                f"on result with new batch shape {old_batch_shape}"
            )
        return result

    return wrapped


for method_name in "_logpdf _pdf _cdf _ppf _mean _rate _std _min _max".split():
    setattr(Reparametrized, method_name, _wrapped_method(method_name))
