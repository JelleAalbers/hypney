import numpy as np
from scipy import optimize

import hypney

export, __all__ = hypney.exporter()


@export
class Estimator:
    def __init__(self, statistic):
        self.statistic = statistic


@export
class Minimum(Estimator):
    sign = 1

    def __call__(self, data):
        guess = np.array([p.default for p in self.statistic.params])
        bounds = [(p.min, p.max) for p in self.statistic.params]

        if hasattr(self.statistic, "compute_with_grad"):
            jac = True

            def fun(params):
                return self.statistic.compute_with_grad(
                    data=data,
                    params=self.param_array(params),
                    grad_wrt=self.statistic.param_names,
                )

        else:
            jac = None

            def fun(params):
                return self.statistic(data=data, params=self.param_array(params))

        result = optimize.minimize(fun=fun, jac=jac, x0=guess, bounds=bounds)

        if result.success:
            return result.x
        raise ValueError(f"Optimizer failed: {result.message}")


@export
class Maximum(Minimum):
    sign = -1
