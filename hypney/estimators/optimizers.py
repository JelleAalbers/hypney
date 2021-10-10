from copy import copy
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
            except Exception as e:
                if final_method:
                    raise e
                elif self.on_failure in ('warn', 'raise'):
                    warnings.warn(f"Optimization with {method} failed: {e}")
                continue

    def _minimize(self, method, on_failure):
        self.history = []
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

        result = optimize.minimize(
            fun=self._objective
            if autograd
            else lambda x, autograd: self._objective(x, autograd)[0],
            jac=True if autograd else None,
            args=(autograd,),
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
            return result, self.history
        else:
            raise ValueError(f"Unknown return_result {self.return_result}")

    def _objective(self, params, autograd):
        nan_result = float("nan"), np.ones(len(params)) * float("nan")

        if not self.in_bounds(params):
            return nan_result
        try:
            result, grad = self._inner_objective(params, autograd)
        except ValueError as e:
            # e.g. pytorch actually crashes when you pass scale=0
            return nan_result
        # JAX, and possibly other libraries, change the precision
        # to 32-bit without asking. Scipy.optimize doesn't expect this
        # and would throw a strange error if we don't cast back to 64:
        # "failed to initialize intent(inout) array -- input not
        #  fortran contiguous -- expected elsize=8 but got 4"
        if isinstance(result, ep.Tensor):
            result = ep_util.np64(result).item()
        if isinstance(grad, ep.Tensor):
            grad = ep_util.np64(grad)
        elif grad is None:
            grad = nan_result[1]
        # Giving float(inf) to optimizer can throw off convergence tests
        # based on the estimated function value. NaN seems safer.
        if not np.isfinite(result):
            return nan_result

        # copy(grad) is needed since e.g. L-BFGS-B modifies it in-place
        self.history.append([params, result, copy(grad)])
        return result, grad

    def _inner_objective(self, params, autograd):
        if autograd:
            # [:,None] ensures batch_shape = 1, so Model._scalar_method
            # won't do the un-tensorifying .item() on the result... kludge?
            param_tensor = self.stat.model._to_tensor(params)[:, None]
            return ep.value_and_grad(
                lambda _ps: self.sign
                * self.stat.model._to_tensor(
                    self.stat.compute(params=self._param_sequence_to_dict(_ps))
                ),
                param_tensor,
            )
        else:
            result = self.sign * self.stat.compute(
                params=self._param_sequence_to_dict(params)
            )
            return result, None


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
