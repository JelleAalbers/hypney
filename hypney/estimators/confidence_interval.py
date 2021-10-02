import numpy as np
from scipy import optimize

import hypney

export, __all__ = hypney.exporter()


@export
class ConfidenceInterval(hypney.Estimator):
    def __init__(
        self,
        stat,
        poi=hypney.DEFAULT_RATE_PARAM.name,
        cl=0.9,
        sign=1,
        anchors=None,
        use_cdf=False,
        *args,
        **kwargs,
    ):
        super().__init__(stat, *args, **kwargs)
        self.poi = poi
        self.cl = cl
        self.sign = sign
        self.use_cdf = use_cdf
        self.poi_spec = stat.model.param_spec_for(poi)

        if not stat.dist:
            raise ValueError(
                "Statistic has no distribution, cannot set confidence intervals"
            )

        # Collect anchors
        user_gave_anchors = bool(anchors)
        if not anchors:
            # Get anchors from the distribution
            # (these will e.g. be present if the dist was generated from toys)
            anchors = stat.dist.param_spec_for(poi).anchors
        if not anchors:
            # No anchors in dist; try the model instead.
            anchors = self.poi_spec.anchors
        if not anchors:
            # If bounds on param are finite, use them as anchors
            bounds = np.array([self.poi_spec.min, self.poi_spec.max])
            if np.all(np.isfinite(bounds)):
                anchors = bounds
        if not anchors:
            raise ValueError("Provide anchors to initially evaluate poi on")
        anchors = np.asarray(hypney.utils.eagerpy.ensure_numpy(anchors))
        if not user_gave_anchors and hasattr(stat, "bestfit"):
            # Add bestfit of POI as an anchor
            anchors = np.concatenate(anchors, stat.bestfit[poi])
        self.anchors = np.sort(anchors)

        # Evaluate statistic at anchors
        # (statistic is vectorized over params)
        anchor_pars = {
            self.poi: hypney.utils.eagerpy.to_tensor(
                self.anchors, match_type=self.stat.data
            )
        }
        self.stat_at_anchors = self.stat.compute(params=anchor_pars)

    def _compute_side(self, side=+1):
        # +1 for upper limit on statistic that (on large scales)
        # takes higher-percentile values as the POI grows (like count)
        sign = self.sign * side

        if sign > 0:
            # Counterintuitive, but see Neyman belt construction diagram.
            crit_quantile = 1 - self.cl
        else:
            crit_quantile = self.cl

        # Find critical value (=corresponding to quantile crit_quantile) at anchors.
        if self.use_cdf:
            # Use CDF to transform statistic to a p-value
            stat_at_anchors = np.array(
                [
                    self.stat.dist.cdf(data=stat_val, params={self.poi: x})
                    for x, stat_val in zip(self.anchors, self.stat_at_anchors)
                ]
            )

            def ppf(params):
                return crit_quantile

            cdf = self.stat.dist.cdf
            crit_at_anchors = np.full_like(stat_at_anchors, crit_quantile)

        else:
            # Use ppf to find critical value of statistic, won't need cdf
            stat_at_anchors = self.stat_at_anchors

            def cdf(data, params):
                return data

            ppf = self.stat.dist(quantiles=crit_quantile).ppf
            crit_at_anchors = np.array(
                [ppf(params={self.poi: x}) for x in self.anchors]
            )

        crit_minus_stat = crit_at_anchors - stat_at_anchors
        isnan = np.isnan(crit_minus_stat)
        if np.any(isnan):
            raise ValueError(
                f"statistic or critical value NaN at {self.anchors[isnan]}"
            )

        # sign+1 => upper limit is above the highest anchor for which
        # crit - stat <= 0 (i.e. crit too low, so still in interval)
        still_in = np.where(sign * crit_minus_stat <= 0)[0]
        if not len(still_in):
            raise ValueError(
                f"None of the anchors {self.anchors} are inside the confidence interval"
            )

        if side > 0:
            ileft = still_in[-1]
            if ileft == len(self.anchors) - 1:
                # Highest possible value is still in interval.
                if self.anchors[-1] == self.poi_spec.max:
                    # Fine, since it's the maximum possible value
                    return self.anchors[-1]
                else:
                    raise ValueError(
                        f"Can't compute upper limit, highest anchor {self.anchors[-1]} still in interval"
                    )
            iright = ileft + 1

        else:
            iright = still_in[0]
            if iright == 0:
                # Lowest possible value is still in interval.
                if self.anchors[0] == self.poi_spec.min:
                    # Fine, since it's the minimum possible value
                    return self.anchors[0]
                else:
                    raise ValueError(
                        f"Can't compute lower limit, lowest anchor {self.anchors[0]} still in interval"
                    )
            ileft = iright - 1

        # Find zero of (crit - stat) - tiny_offset
        # The offset is needed if crit = stat for an extended length
        # e.g. for Count or other discrete-valued statistics.
        # TODO: can we use grad? optimize.root takes a jac arg...
        # Don't ask about the sign. All four side/sign combinations are tested...
        offset = self.sign * 1e-9 * (crit_minus_stat[ileft] - crit_minus_stat[iright])

        def objective(x):
            params = {self.poi: x}
            return (
                ppf(params=params)
                - cdf(data=self.stat.compute(params=params), params=params)
                + offset
            )

        return optimize.brentq(objective, self.anchors[ileft], self.anchors[iright])


@export
class UpperLimit(ConfidenceInterval):
    def _compute(self):
        return self._compute_side(+1)


@export
class LowerLimit(ConfidenceInterval):
    def _compute(self):
        return self._compute_side(-1)


@export
class CentralInterval(ConfidenceInterval):
    def _compute(self):
        self.cl = 1 - (1 - self.cl) / 2
        return self._compute_side(-1), self._compute_side(+1)
