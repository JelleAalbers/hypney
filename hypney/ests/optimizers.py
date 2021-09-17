import eagerpy as ep
import numpy as np
from scipy import optimize

import hypney

export, __all__ = hypney.exporter()


@export
class Minimum(hypney.Estimator):
    sign = 1

    def _free_params(self):
        return [p for p in self.stat.param_specs if p.name not in self.fix]

    def __call__(self, data):
        stat = self.stat.freeze(data)
        guess = np.array([p.default for p in self._free_params()])
        bounds = [(p.min, p.max) for p in self._free_params()]

        if isinstance(stat.data, ep.NumPyTensor):
            jac = None

            def fun(params):
                return stat(params=self._param_array_to_dict(params))

        else:
            jac = True

            def fun(params):
                result, grad = ep.value_and_grad(
                    lambda param_tensor: stat(
                        params=self._param_sequence_to_dict(param_tensor)
                    ),
                    hypney.sequence_to_tensor(params, match_type=stat.data),
                )
                return result.numpy(), grad.numpy()

        result = optimize.minimize(fun=fun, jac=jac, x0=guess, bounds=bounds)

        if result.success:
            return result.x
        raise ValueError(f"Optimizer failed: {result.message}")

    def _param_sequence_to_dict(self, x):
        params = {p.name: x[i] for i, p in enumerate(self._free_params())}
        return {**params, **self.fix}


@export
class Maximum(Minimum):
    sign = -1
