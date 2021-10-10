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
        method=("L-BFGS-B", "powell"),
        return_result=None,
        on_failure="raise",
        autograd=None,
        **kwargs,
    ):
        self.on_failure = on_failure

        if options is None:
            options = dict()
        self.options = options

        if return_result:
            self.return_result = return_result

        if isinstance(method, str):
            method = (method,)
        self.methods = method

        super().__init__(*args, **kwargs)

        if autograd is None:
            autograd = self.stat.model._backend_name != "numpy"
        self.autograd = autograd

    def in_bounds(self, params):
        if not isinstance(params, dict):
            params = self._param_sequence_to_dict(params)
        for p in self._free_params():
            v = params[p.name]
            if not p.min <= v <= p.max:
                return False
        return True

    def _compute(self):
        # Try different methods; complaining only if last one fails.
        for i, method in enumerate(self.methods):
            final_method = i == len(self.methods) - 1
            on_failure = self.on_failure if final_method else "raise"
            try:
                return self._minimize(method, on_failure=on_failure)
            except Exception:
                if final_method:
                    raise
                continue

    def _minimize(self, method, on_failure):
        history = []
        guess = np.array(
            [
                hypney.utils.eagerpy.ensure_numpy_float(p.default)
                for p in self._free_params()
            ]
        )
        bounds = [(p.min, p.max) for p in self._free_params()]

        if method in ("powell", "nelder-mead", "cobyla"):
            # Trying only gradient-free methods; autograd pointless
            autograd = False
        else:
            autograd = self.autograd

        if not autograd:
            jac = None

            def fun(params):
                nonlocal history
                if not self.in_bounds(params):
                    result = float("nan")
                else:
                    try:
                        result = self.sign * self.stat.compute(
                            params=self._param_sequence_to_dict(params)
                        )
                    except ValueError as e:
                        # pytorch actually crashes when you pass scale=0
                        result = float("nan")
                    # Giving float(inf) to optimizer can throw off convergence tests
                    # based on estimated function value. NaN seems safer.
                    if not np.isfinite(result):
                        result = float("nan")
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

            def fun(params):
                nonlocal history
                if not self.in_bounds(params):
                    result = float("nan")
                    grad = np.ones(len(params)) * float("nan")
                else:
                    param_tensor = self.stat.model._to_tensor(params)
                    # [:,None] ensures batch_shape = 1, so Model._scalar_method
                    # won't do the un-tensorifying .item() on the result... kludge?
                    param_tensor = param_tensor[:, None]
                    try:
                        result, grad = ep.value_and_grad(
                            lambda _ps: self.sign
                            * self.stat.model._to_tensor(
                                self.stat.compute(
                                    params=self._param_sequence_to_dict(_ps)
                                )
                            ),
                            param_tensor,
                        )
                    except ValueError:
                        result = float("nan")
                        grad = np.ones(len(params)) * float("nan")
                    else:
                        # JAX, and possibly other libraries, change the precision
                        # to 32-bit without asking. Scipy.optimize doesn't expect this
                        # and would throw a strange error if we don't cast back to 64:
                        # "failed to initialize intent(inout) array -- input not
                        #  fortran contiguous -- expected elsize=8 but got 4"
                        result = ep_util.np64(result).item()
                        grad = ep_util.np64(grad)
                    if not np.isfinite(result):
                        result *= float("nan")
                        grad *= float("nan")
                # grad.copy() since optimizer might modify it in-place
                # (e.g. LBFGS-B seems to replace NaN with the previous value)
                history.append([params, result, grad.copy()])
                return result, grad

        result = optimize.minimize(
            fun=fun,
            jac=jac,
            x0=guess,
            bounds=bounds,
            options=self.options,
            method=method,
        )

        if not result.success:
            msg = f"Optimizer failed: {result.message}"
            if on_failure == "raise":
                raise ValueError(msg)
            elif on_failure == "warn":
                warnings.warn(msg)
            elif on_failure == "silent":
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
