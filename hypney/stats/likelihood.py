import numpy as np

import hypney

export, __all__ = hypney.exporter()


@export
class LogLikelihood(hypney.Statistic):
    def _compute(self, params):
        return -self.model.rate(params) + np.sum(
            np.log(self.model.diff_rate(self.data, params))
        )


@export
class LikelihoodRatio(hypney.Statistic):
    def __init__(self, *args, max_estimator=None, **kwargs):
        super().__init__(*args, **kwargs)

        if max_estimator is None:
            max_estimator = hypney.Maximum
        self.max_estimator = max_estimator
        self.ll = LogLikelihood(self.model)
        self.mle = self.max_estimator(self.ll)

    def _init_data(self):
        self.bestfit = self.mle(self.data)
        self.ll_bestfit = self.ll(self.mle, self.data)

    def _compute(self, params):
        return -2 * (self.ll(params, self.data) - self.ll_bestfit)


@export
class PLR(LikelihoodRatio):
    # Dangerous! Goes down then up with rate poi, not nice / rectified / whatever

    def __init__(self, *args, poi=tuple(), **kwargs):
        if isinstance(poi, str):
            poi = (poi,)
        self.poi = tuple(poi)
        super().__init__(*args, **kwargs)

    def _filter_poi(self, params):
        return {pname: val for pname, val in params.items() if pname in self.poi}

    def _compute(self, params):
        conditional_bestfit = self.max_estimator(self.ll, fix=self._filter_poi(params))(
            self.data
        )
        return -2 * (self.ll(conditional_bestfit, self.data) - self.ll_bestfit)

    def _build_dist(self):
        return hypney.models.Chi2(df=len(self.poi))


class PLROrZero(PLR):
    def __init__(self, *args, zero_if="high", **kwargs):
        super().__init__(*args, **kwargs)
        assert zero_if in ("high", "low")
        self.zero_if = zero_if
        assert len(self.poi) == 1

    def _compute(self, params):
        result = super()._compute(params)
        saw_high = params[self.poi] > self.bestfit[self.poi]
        if saw_high == (self.zero_if == "high"):
            return 0
        return result

    def _build_dist():
        return hypney.models.DiracDelta(rate=0.5) + hypney.models.Chi2(df=1, rate=0.5)


@export
class SignedPLR(PLR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.poi) == 1

    def _compute(self, params):
        result = super()._compute(params)
        if params[self.poi] > self.bestfit[self.poi]:
            # High / Excess hypothesis (if poi ~ rate)
            return result
        else:
            # Low / Deficit hypothesis (if poi ~ rate)
            return -result

    def _build_dist():
        return hypney.models.NegativeData(
            hypney.models.Chi2(df=1, rate=0.5)
        ) + hypney.models.Chi2(df=1, rate=0.5)
