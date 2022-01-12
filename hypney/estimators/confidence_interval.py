from functools import partial
import numpy as np
from scipy import optimize

import hypney

export, __all__ = hypney.exporter()


@export
class ConfidenceInterval:
    def __init__(
        self,
        stat,
        poi=hypney.DEFAULT_RATE_PARAM.name,
        cl=0.9,
        sign=1,
        anchors=None,
        use_cdf=False,
        ppf_fudge=0,
    ):
        self.stat = stat
        self.poi = poi
        self.cl = cl
        self.sign = sign
        self.use_cdf = use_cdf
        self.ppf_fudge = ppf_fudge
        self.poi_spec = self.stat.model.param_spec_for(poi)

        if not self.stat.dist:
            raise ValueError(
                "Statistic has no distribution, cannot set confidence intervals"
            )

        # Collect anchors
        user_gave_anchors = bool(anchors)
        # (conditions are wordy since np.array has no truth value
        if anchors is None or not len(anchors):
            # Get anchors from the (reparametrized) distribution
            # (these may e.g. be present if the dist was generated from toys)
            anchors = self.stat.dist.param_spec_for(poi).anchors
        if anchors is None or not len(anchors):
            # No anchors in dist; try the model instead.
            anchors = self.poi_spec.anchors
        if anchors is None or not len(anchors):
            # If bounds on param are finite, use them as anchors
            bounds = np.array([self.poi_spec.min, self.poi_spec.max])
            if np.all(np.isfinite(bounds)):
                anchors = bounds
        if anchors is None or not len(anchors):
            raise ValueError("Provide anchors to initially evaluate poi on")
        anchors = np.asarray(hypney.utils.eagerpy.ensure_numpy(anchors))
        if not user_gave_anchors and hasattr(self.stat, "bestfit"):
            # Add bestfit of POI as an anchor
            anchors = np.concatenate(anchors, self.stat.bestfit[poi])
        self.anchors = np.sort(anchors)

        # +1 for upper limit on statistic that (on large scales)
        # takes higher-percentile values as the POI grows (like count)
        self.combined_sign = self.sign * self.side

        if self.combined_sign > 0:
            # Counterintuitive, but see Neyman belt construction diagram.
            self.crit_quantile = 1 - self.cl
        else:
            self.crit_quantile = self.cl

        if self.use_cdf:
            # We will use the CDF to transform statistics to p-values
            # Can't do much here, have to wait for data.
            self._cdf = stat.dist.cdf
            self._ppf_pre_fudge = self._trivial_ppf
            self.crit_at_anchors = np.full(len(self.anchors), self.crit_quantile)

        else:
            # We will use the ppf to find critical value of statistic
            # Won't need the cdf, set it to identity.
            # Can compute critical value at anchors already here,
            # (so won't need to repeat it when testing several datasets)
            self._cdf = self._trivial_cdf
            self._ppf_pre_fudge = stat.dist(quantiles=self.crit_quantile).ppf
            # Find critical value (=corresponding to quantile crit_quantile) at anchors.
            self.crit_at_anchors = self._ppf(params={self.poi: self.anchors})

    def _ppf(self, *args, **kwargs):
        return self.ppf_fudge + self._ppf_pre_fudge(*args, **kwargs)

    def _trivial_cdf(self, data, params):
        return data

    def _trivial_ppf(self, params):
        return self.crit_quantile + self.ppf_fudge

    def __call__(self, data=hypney.NotChanged):
        stat = self.stat(data=data)

        # Evaluate statistic at anchors
        # (statistic is vectorized over params)
        anchor_pars = {self.poi: stat.model._to_tensor(self.anchors).raw}
        stat_at_anchors = stat.compute(params=anchor_pars)

        if self.use_cdf:
            # Use CDF to transform statistic to a p-value
            stat_at_anchors = np.array(
                [
                    stat.dist.cdf(data=stat_val, params={self.poi: x})
                    for x, stat_val in zip(self.anchors, stat_at_anchors)
                ]
            )

        crit_minus_stat = self.crit_at_anchors - stat_at_anchors
        isnan = np.isnan(crit_minus_stat)
        if np.any(isnan):
            raise ValueError(
                f"statistic or critical value NaN at {self.anchors[isnan]}"
            )

        # sign+1 => upper limit is above the highest anchor for which
        # crit - stat <= 0 (i.e. crit too low, so still in interval)
        still_in = np.where(self.combined_sign * crit_minus_stat <= 0)[0]
        if not len(still_in):
            raise ValueError(
                f"None of the anchors {self.anchors} are inside the confidence interval"
            )

        if self.side > 0:
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

        return optimize.brentq(
            partial(self._objective, stat=stat, offset=offset),
            self.anchors[ileft],
            self.anchors[iright],
        )

    def _objective(self, x, stat, offset):
        params = {self.poi: x}
        return (
            # One of ppf/cdf is trivial here, depending on self.use_cdf
            self._ppf(params=params)
            - self._cdf(data=stat.compute(params=params), params=params)
            + offset
        )


@export
class UpperLimit(ConfidenceInterval):
    side = +1


@export
class LowerLimit(ConfidenceInterval):
    side = -1


@export
class CentralInterval:
    def __init__(self, *args, cl=0.9, **kwargs):
        kwargs["cl"] = 1 - (1 - cl) / 2
        self._lower = LowerLimit(*args, **kwargs)
        self._upper = UpperLimit(*args, **kwargs)

    def __call__(self, data=hypney.NotChanged):
        return self._lower(data), self._upper(data)
