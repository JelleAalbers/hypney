import eagerpy as ep

import hypney

export, __all__ = hypney.exporter()


@export
class LogLikelihood(hypney.Statistic):
    def _compute(self, params):
        result = (
            -self.model._rate(params)
            +
            # _log_diff_rate(params=params).sum() seems less stable...
            # Maybe this depends on whether pdf or diff_rate is fundamental for
            # a model?
            self.data.shape[-2] * self.model._log_rate(params)
            # keepdims=True since params has trailing (1,)... :-(
            + self.model._logpdf(params=params).sum(axis=-1, keepdims=True)
            # TODO: axis=-1 assumes sample_shape is 1d!!
        )
        # Somehow the scipy optimizer is fine with -inf,
        # but not with accurate but insanely large values??
        # TODO: madness. Investigate.
        if isinstance(self.data, ep.NumPyTensor):
            return ep.where(result < -1e5, result * float("inf"), result)
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

    def __init__(self, *args, poi=tuple(), **kwargs):
        if isinstance(poi, str):
            poi = (poi,)
        self.poi = tuple(poi)
        super().__init__(*args, **kwargs)

    def _filter_poi(self, params: dict):
        """Return only parameters of interest from params"""
        return {pname: val for pname, val in params.items() if pname in self.poi}

    def _compute(self, params):
        self.conditional_fit, self.ll_conditional_fit = self.max_est(
            self.ll, fix=self._filter_poi(params)
        )
        # Probably slower alternative:
        # conditional_ll = LogLikelihood(self.model(fix=self._filter_poi(params)))
        return self.ll_conditional_fit - self.ll_bestfit

    def _build_dist(self):
        return hypney.models.chi2(df=len(self.poi)).freeze()


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

    def _build_dist(self):
        return (
            hypney.models.DiracDelta(rate=0.5) + hypney.models.chi2(df=1, rate=0.5)
        ).freeze()


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

    def _build_dist(self):
        half_chi2 = hypney.models.chi2(df=1, rate=0.5)
        dist = half_chi2 + half_chi2.scale(-1)
        return dist.freeze()
