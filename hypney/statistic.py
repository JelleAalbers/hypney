import inspect

import numpy as np

import hypney

export, __all__ = hypney.exporter()


class Statistic(hypney.Element):

    def __call__(self, data, params):
        raise NotImplementedError

    @classmethod
    def from_function(cls, f, model: hypney.Model = None, param_specs=None):
        self = cls()

        f_params = list(inspect.signature(f).parameters.values())
        assert f_params[0].name == "data", "First argument must be data"

        if f_params[1].name == "params":
            if model is not None:
                # Take param specs from model
                specs = model.param_specs
            else:
                # Take explicitly provided specs
                specs = param_specs
                assert isinstance(specs, tuple)
                assert all([isinstance(x, hypney.ParameterSpec) for x in specs])

            self.param_specs = specs
            self.__call__ = f

        else:
            # Construct parameter spec from the function signature
            # params_specs argument is now an optional as dict with options
            # TODO: Confusing! Make separate arg/kwarg!!
            if param_specs is None:
                spec = tuple()
            spec = tuple(
                [
                    hypney.ParameterSpec(
                        name=p.name,
                        default=p.default,
                        **param_specs.get(p.name, dict())
                    )
                    for p in f_params[1:]
                ]
            )
            assert not any([p.default == inspect.Parameter.empty for p in spec])

            self.__call__ = lambda data, params: f(data, **params)


class LogLikelihood(Statistic):
    def __init__(self, model: hypney.Model):
        self.model = model

    def __call__(self, data, params):
        return -self.model.rate(params) + np.sum(
            np.log(self.model.diff_rate(data, params))
        )
