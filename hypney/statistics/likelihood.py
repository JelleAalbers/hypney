import eagerpy as ep

import hypney

export, __all__ = hypney.exporter()


@export
class LogLikelihood(hypney.Statistic):
    def _compute(self, params):
        result = (
            -self.model._rate(params)
            # keepdims=True since params has trailing (1,)... :-(
            + self.model._log_diff_rate(params=params).sum(axis=-1, keepdims=True)
        )
        return result


@export
class LikelihoodRatio(hypney.Statistic):
    def __init__(self, model, *args, max_est=None, **kwargs):
        if max_est is None:
            max_est = hypney.estimators.MaximumAndValue
        self.max_est = max_est

        super().__init__(model=model, *args, **kwargs)

    def _init_data(self):
        super()._init_data()
        self.ll = LogLikelihood(self.model)
        self.bestfit, self.ll_bestfit = self.max_est(self.ll)

    def _compute(self, params):
        return -2 * (self.ll._compute(params=params) - self.ll_bestfit)

    def _build_dist(self):
        return hypney.models.chi2(df=len(self.model.param_names)).freeze()


@export
class PLR(LikelihoodRatio):
    # Dangerous! Goes down then up with rate poi, not nice / rectified / whatever

    def __init__(self, *args, poi="rate", **kwargs):
        if isinstance(poi, str):
            poi = (poi,)
        self.poi = tuple(poi)
        super().__init__(*args, **kwargs)

    @property
    def only_poi(self):
        """Returns single parameter of interest,
        crashes if there is more than one"""
        assert len(self.poi) == 1
        return self.poi[0]

    def _filter_poi(self, params: dict):
        """Return only parameters of interest from params"""
        return {pname: val for pname, val in params.items() if pname in self.poi}

    def _compute(self, params):
        self.conditional_fit, self.ll_conditional_fit = self.max_est(
            self.ll, fix=self._filter_poi(params)
        )
        # Probably slower alternative:
        # conditional_ll = LogLikelihood(self.model(fix=self._filter_poi(params)))
        return -2 * (self.ll_conditional_fit - self.ll_bestfit)

    def _build_dist(self):
        return hypney.models.chi2(df=len(self.poi)).freeze()


@export
class PLROrZero(PLR):
    def __init__(self, *args, zero_if="high", **kwargs):
        super().__init__(*args, **kwargs)
        assert zero_if in ("high", "low")
        self.zero_if = zero_if

    def _compute(self, params):
        result = super()._compute(params)
        saw_high = params[self.only_poi] > self.bestfit[self.only_poi]
        if saw_high == (self.zero_if == "high"):
            return 0
        return result

    def _build_dist(self):
        return (
            hypney.models.DiracDelta(rate=0.5) + hypney.models.chi2(df=1, rate=0.5)
        ).freeze()


@export
class SignedPLR(PLR):
    def _compute(self, params):
        result = super()._compute(params)
        if self.bestfit[self.only_poi] > params[self.only_poi]:
            # Excess-like result: positive
            return result
        else:
            # Deficit-like result: negative
            return -result

    def _build_dist(self):
        half_chi2 = hypney.models.chi2(df=1, rate=0.5)
        dist = half_chi2 + half_chi2.scale(-1)
        return dist.freeze()
