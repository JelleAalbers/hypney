import eagerpy as ep
import numpy as np
from scipy import optimize

import hypney
import hypney.utils.eagerpy as ep_util

export, __all__ = hypney.exporter()


@export
class MinimumAndValue(hypney.Estimator):
    sign = 1

    def _compute(self):
        guess = np.array([hypney.utils.eagerpy.ensure_numpy_float(p.default) for p in self._free_params()])
        bounds = [(p.min, p.max) for p in self._free_params()]

        if isinstance(self.stat.data, ep.NumPyTensor):
            jac = None

            def fun(params):
                return self.sign * self.stat.compute(
                    params=self._param_sequence_to_dict(params)
                )

        else:
            jac = True

            def _fun(params):
                result, grad = ep.value_and_grad(
                    lambda param_tensor: self.sign
                    * self.stat.compute(
                        params=self._param_sequence_to_dict(param_tensor)
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
            fun = lambda params: [
                ep_util.np64(x)
                for x in _fun(ep_util.to_tensor(params, match_type=self.stat.data))
            ]

        result = optimize.minimize(fun=fun, jac=jac, x0=guess, bounds=bounds)

        if result.success:
            return self._param_sequence_to_dict(result.x), result.fun * self.sign
        raise ValueError(f"Optimizer failed: {result.message}")


@export
class Minimum(MinimumAndValue):
    def _compute(self):
        return super()._compute()[0]


@export
class MaximumAndValue(MinimumAndValue):
    sign = -1


@export
class Maximum(Minimum):
    sign = -1
