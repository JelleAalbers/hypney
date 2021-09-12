from copy import copy

import numpy as np
from scipy import stats

import hypney

export, __all__ = hypney.exporter()


class Statistic(hypney.Element):
    # Does statistic depends on data? If not, it depends only on parameters
    # (like a prior / constraint)
    data_dependent = True

    # Does statistic depends on parameters? If not, it depends only on the data
    # (like the count of events).
    # In the latter case,
    #   * compute takes only data as an argument (not params)
    #   * compute will be run on initialization (if data is available)
    #     and the result stored in _result, which __call__ returns.
    #   * The distribution will still be assumed to depend on parameters.
    param_dependent = True

    # Is data necessary to compute the statistic on different parameters?
    # If not, init_data should compute sufficient summaries.
    keep_data = True

    data = None

    def __init__(self, param_container: hypney.Element, data=None, distribution=None):
        self.param_specs = param_container.param_specs
        if not self.param_dependent:
            self.keep_data = False

        self._set_data(data)

        if hasattr(distribution, "build"):
            self.pdf, self.cdf = distribution.build(self, param_container)
        if isinstance(distribution, stats.rv_frozen):
            if isinstance(distribution.dist, stats.rv_continuous):
                self.pdf, self.cdf = distribution.pdf, distribution.cdf
            else:
                # Sorry statisticians, I'm just going to call pmf pdf...
                self.pdf, self.cdf = distribution.pmf, distribution.cdf
        else:
            raise ValueError("Invalid distribution")

    def validate_data(self, data):
        return self.param_container.validate_data(data)

    def _set_data(self, data=None):
        if data is None:
            return

        data = self.validate_data(data)
        self.data = data
        self.init_data()
        if not self.param_dependent:
            # Precompute result on the data.
            self._result = self.compute()

        if not self.keep_data:
            # Statistic relies only on stuff computed in init_data
            # (well, and params probably); replace self.data with something other
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

    def pdf(self, params):
        raise NotImplementedError

    def cdf(self, params):
        raise NotImplementedError


class StatisticFromModel(Statistic):
    @property
    def model(self):
        return self.param_container

    def rvs(self, size=1, params=None):
        """Return statistic evaluated on simulated data,
        generated from model with params"""
        results = np.zeros(size)
        for i in range(len(size)):
            sim_data = self.model.simulate(params=params)
            results[i] = self(data=sim_data, params=params)
        return results


class LogLikelihood(StatisticFromModel):
    def compute(self, params):
        return -self.model.rate(params) + np.sum(
            np.log(self.model.diff_rate(self.data, params))
        )


class LogLikelihoodRatio(StatisticFromModel):
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


class Count(StatisticFromModel):
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
