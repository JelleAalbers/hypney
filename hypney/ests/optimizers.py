import eagerpy as ep
import numpy as np
from scipy import optimize

import hypney

export, __all__ = hypney.exporter()


@export
class Minimum(hypney.Estimator):
    sign = 1

    def _compute(self, stat):
        guess = np.array([p.default for p in self._free_params()])
        bounds = [(p.min, p.max) for p in self._free_params()]

        if isinstance(stat.data, ep.NumPyTensor):
            jac = None

            def fun(params):
                result = (
                    self.sign
                    * stat(params=self._param_sequence_to_dict(params)).numpy()
                )
                return result

        else:
            jac = True

            def fun(params):
                result, grad = ep.value_and_grad(
                    lambda param_tensor: self.sign
                    * stat(params=self._param_sequence_to_dict(param_tensor)),
                    hypney.sequence_to_tensor(params, match_type=stat.data),
                )
                return result.numpy(), grad.numpy()

        result = optimize.minimize(fun=fun, jac=jac, x0=guess, bounds=bounds)

        if result.success:
            return self._param_sequence_to_dict(result.x)
        raise ValueError(f"Optimizer failed: {result.message}")


@export
class Maximum(Minimum):
    sign = -1
