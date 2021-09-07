from copy import copy

import numpy as np
from scipy import stats

import hypney

export, __all__ = hypney.exporter()


class Statistic:
    data_dependent = True
    param_dependent = True
    keep_data = True
    data = None

    def __init__(self, model: hypney.Model, data=None):
        self.model = model
        self._set_data(data)

        if not self.param_dependent:
            self.keep_data = False

    def _set_data(self, data=None):
        if data is None:
            return

        data = self.model.validate_data(data)
        self.data = data
        self.init_data()

        if not self.param_dependent:
            # Precompute result on this data. See __call__ for special case
            # where this is returned
            self._result = self.compute()
        if not self.keep_data:
            # Statistic relies only on stuff computed in init_data
            # (and possibly params); replace self.data with something other
            # than None that takes no memory
            self.data = True

    def init_data(self):
        # Child classes may wish to do some computations here
        pass

    def __call__(self, *args, data=None, params=None):

        if len(args) == 0:
            # data and params passed by keyword
            pass

        elif len(args) == 2:
            # data and params passed as positional arguments
            # No keywords can be passed:
            assert (
                params is None and data is None
            ), "Statistic takes at most 2 arguments"
            data, params = args

        elif len(args) == 1:
            # One positional argument... harder case.
            if params is None and data is None:
                # No keywords provided.
                # Data must have been set on init, and args[0] is params
                assert self.data is not None, "Must provide data"
                params = args[0]
            elif params is None:
                # Data provided by keyword; positional arg must be params
                params = args[0]
            elif data is None:
                # Params provided by keyword; positional arg must be data
                data = args[0]
            else:
                raise ValueError("Statistic takes at most 2 arguments")
        else:
            raise ValueError("Statistic takes at most 2 arguments")

        if data is not None:
            # Work on a copy of self with the new data set
            self = copy(self)
            self._set_data(data)

        elif not self.param_dependent:
            # Data has not changed, and doesn't depend on parameters:
            # use precomputed result
            return self._result

        params = self.model.validate_params(params)
        return self.compute(params)

    def compute(self, params):
        raise NotImplementedError


class LogLikelihood(Statistic):
    def compute(self, params):
        return -self.model.rate(params) + np.sum(
            np.log(self.model.diff_rate(self.data, params))
        )


class LogLikelihoodRatio(Statistic):
    def __init__(self, *args, max_estimator=None, **kwargs):
        super().__init__(*args, **kwargs)

        if max_estimator is None:
            max_estimator = hypney.Maximum
        self.ll = LogLikelihood(self.model)
        self.mle = max_estimator(self.ll)

    def init_data(self):
        self.bestfit = self.mle(self.data)
        self.ll_bestfit = self.ll(self.mle, self.data)

    def compute(self, params):
        return self.ll(params, self.data) - self.ll_bestfit


class Count(Statistic):
    param_dependent = False

    def compute(self):
        return self.n

    def pdf(self, params):
        return stats.poisson(mu=self.model.rate(params)).pmf


class Mean(Statistic):
    # TODO: specify which dimension to average over
    param_dependent = False

    def compute(self):
        return np.mean(self.data)


class Prior(Statistic):
    data_dependent = False

    def __call__(self, data, params):
        raise NotImplementedError


Constraint = Prior
