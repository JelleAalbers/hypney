import warnings

import eagerpy as ep
import numpy as np
from scipy import optimize

import hypney
import hypney.utils.eagerpy as ep_util

export, __all__ = hypney.exporter()


@export
class Minimum(hypney.Estimator):
    sign = 1
    return_result = "point"

    def __init__(
        self,
        *args,
        options=None,
        method=None,
        return_result=None,
        on_failure="error",
        **kwargs,
    ):
        if options is None:
            options = dict()
        if return_result:
            self.return_result = return_result
        self.on_failure = on_failure
        self.options = options
        self.method = method
        super().__init__(*args, **kwargs)

    def _compute(self):
        history = []
        guess = np.array(
            [
                hypney.utils.eagerpy.ensure_numpy_float(p.default)
                for p in self._free_params()
            ]
        )
        bounds = [(p.min, p.max) for p in self._free_params()]

        if isinstance(self.stat.data, ep.NumPyTensor):
            jac = None

            def fun(params):
                nonlocal history
                result = self.sign * self.stat.compute(
                    params=self._param_sequence_to_dict(params)
                )
                history.append([params, result, None])
                return result

        else:
            jac = True

            def _fun(params):
                result, grad = ep.value_and_grad(
                    lambda param_tensor: self.sign
                    * self.stat.model._to_tensor(
                        self.stat.compute(
                            # [:,None] ensures batch_shape = 1, so Model._scalar_method
                            # won't un-tensorify the result...
                            params=self._param_sequence_to_dict(param_tensor[:, None])
                        )
                    ),
                    params,
                )
                return result, grad

            if isinstance(ep.astensor(self.stat.data), ep.JAXTensor):
                # Can't just ep.jax.jit, it tries to wrap the output
                # (i.e. the compiled function) as an eagerpy tensor...
                import jax

                _fun = jax.jit(_fun)

            # JAX, and possibly other libraries, change the precision
            # to 32-bit without asking. Scipy.optimize doesn't expect this
            # and would throw a strange error if we don't cast back to 64:
            # "failed to initialize intent(inout) array -- input not
            #  fortran contiguous -- expected elsize=8 but got 4"
            def fun(params):
                nonlocal history
                result, grad = [
                    ep_util.np64(x)
                    for x in _fun(ep_util.to_tensor(params, match_type=self.stat.data))
                ]
                history.append([params, result, grad])
                return result, grad

        result = optimize.minimize(
            fun=fun,
            jac=jac,
            x0=guess,
            bounds=bounds,
            options=self.options,
            method=self.method,
        )

        if not result.success:
            msg = f"Optimizer failed: {result.message}"
            if self.on_failure == "error":
                raise ValueError(msg)
            elif self.on_failure == "warning":
                warnings.warn(msg)
            elif self.on_failure == "silent":
                pass
            else:
                raise ValueError(f"Unknown on_failure {self.on_failure}")

        bestfit = self._param_sequence_to_dict(result.x)
        value = result.fun * self.sign

        if self.return_result == "point":
            return bestfit
        elif self.return_result == "point_and_value":
            return bestfit, value
        elif self.return_result == "value":
            return value
        elif self.return_result == "optresult":
            return result
        elif self.return_result == "optresult_and_history":
            return result, history
        else:
            raise ValueError(f"Unknown return_result {self.return_result}")


@export
class MinimumAndValue(Minimum):
    return_result = "point_and_value"


@export
class MaximumAndValue(MinimumAndValue):
    sign = -1
    return_result = "point_and_value"


@export
class Maximum(Minimum):
    sign = -1
