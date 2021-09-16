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

        if hasattr(self.statistic, "compute_with_grad"):
            jac = True

            def fun(params):
                return stat.compute_with_grad(
                    params=self._array_to_params(params),
                    grad_wrt=self.stat.param_names,
                )

        else:
            jac = None

            def fun(params):
                return stat(params=self._array_to_params(params))

        result = optimize.minimize(fun=fun, jac=jac, x0=guess, bounds=bounds)

        if result.success:
            return result.x
        raise ValueError(f"Optimizer failed: {result.message}")

    def _array_to_params(self, x: np.ndarray):
        params = {p.name: x[i] for i, p in enumerate(self._free_params())}
        return {**params, **self.fix}


@export
class Maximum(Minimum):
    sign = -1
