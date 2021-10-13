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
    return_kind = "point"

    def __init__(
        self,
        *args,
        options=None,
        method=("L-BFGS-B", "powell"),
        return_kind=None,
        on_failure="raise",
        autograd=None,
        **kwargs,
    ):
        self.on_failure = on_failure

        if options is None:
            options = dict()
        self.options = options

        if return_kind:
            self.return_kind = return_kind
        if isinstance(self.return_kind, str):
            self.return_kind = (self.return_kind,)

        if isinstance(method, str):
            method = (method,)
        self.methods = method

        super().__init__(*args, **kwargs)

        if autograd is None:
            autograd = self.stat.model._backend_name != "numpy"
        self.autograd = autograd

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
                elif self.on_failure in ("warn", "raise"):
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

        fun = (
            self.objective
            if autograd
            else lambda x, autograd: self.objective(x, autograd)[0]
        )
        result = optimize.minimize(
            fun=fun,
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

        possible_returns = dict(
            point=self._param_sequence_to_dict(result.x),
            value=result.fun * self.sign,
            optresult=result,
            history=self.history,
        )
        return [possible_returns[x] for x in self.return_kind]

    def objective(self, params, autograd):
        result, grad = self._objective(params, autograd)
        # copy(grad) is needed since e.g. L-BFGS-B modifies it in-place
        self.history.append([params, result, copy(grad)])
        return result, grad

    def _objective(self, params, autograd):
        if autograd:
            nan_result = float("nan"), np.ones(len(params)) * float("nan")
        else:
            nan_result = float("nan"), None

        # Bounds check
        for i, p in enumerate(self._free_params()):
            if not p.min <= params[i] <= p.max:
                return nan_result

        # Evaluate the statistic, possibly under autograd
        try:
            result, grad = self._innermost_objective(params, autograd)
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
            grad = ep_util.np64(grad).ravel()
        # Giving float(inf) to optimizer can throw off convergence tests
        # based on the estimated function value. NaN seems safer.
        if not np.isfinite(result):
            return nan_result
        return result, grad

    def _innermost_objective(self, params, autograd):
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
    return_kind = ("point", "value")


@export
class MaximumAndValue(MinimumAndValue):
    sign = -1


@export
class Maximum(Minimum):
    sign = -1
