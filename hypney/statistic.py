from copy import copy

import numpy as np
from scipy import stats

import hypney
from hypney import NotChanged

export, __all__ = hypney.exporter()


@export
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

        if isinstance(distribution, stats.rv_frozen):
            if isinstance(distribution.dist, stats.rv_continuous):
                self.pdf, self.cdf = distribution.pdf, distribution.cdf
            else:
                # Sorry statisticians, I'm just going to call pmf pdf...
                self.pdf, self.cdf = distribution.pmf, distribution.cdf
        elif distribution is not None:
            raise ValueError("Invalid distribution")

    def validate_data(self, data):
        return self.param_container.validate_data(data)

    def _init_data(self):
        if not self.param_dependent:
            # Precompute result on the data.
            self._result = self._compute()

        if not self.keep_data:
            # Statistic relies only on stuff computed in init_data,
            # so we can throw away our reference to the data
            self.data = None
        super().init_data()

    def __call__(self, params: dict = None, data=NotChanged):
        if data is NotChanged:
            if not self.param_dependent:
                return self._result
            if self.data is None:
                raise ValueError("Must provide data")
        else:
            # Data was passed, work on a copy of self using the new data
            self = copy(self)
            self._set_data(data)

        params = self.param_container.validate_params(params)
        return self._compute(params)

    def _compute(self, params):
        raise NotImplementedError

    def pdf(self, params):
        raise NotImplementedError

    def cdf(self, params):
        raise NotImplementedError


@export
class StatisticFromModel(Statistic):
    @property
    def model(self) -> hypney.Model:
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
    def _compute(self, params):
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

    def _init_data(self):
        self.bestfit = self.mle(self.data)
        self.ll_bestfit = self.ll(self.mle, self.data)

    def _compute(self, params):
        return self.ll(params, self.data) - self.ll_bestfit


@export
class Count(StatisticFromModel):
    param_dependent = False

    def _compute(self):
        return self.n

    def pdf(self, params):
        return stats.poisson(mu=self.model.rate(params)).pmf


@export
class Mean(Statistic):
    # TODO: specify which dimension to average over
    param_dependent = False

    def _compute(self):
        return np.mean(self.data)


class Prior(Statistic):
    data_dependent = False

    def __call__(self, data, params):
        raise NotImplementedError


Constraint = Prior
