import numpy as np
from scipy import optimize

import hypney

export, __all__ = hypney.exporter()


@export
class UpperLimit(hypney.Estimator):
    def __init__(
        self,
        stat,
        poi=hypney.DEFAULT_RATE_PARAM.name,
        cl=0.9,
        anchors=None,
        *args,
        **kwargs,
    ):
        self.poi = poi
        self.cl = cl
        if not anchors:
            # Get anchors from the distribution
            # (e.g. will be present if generated from toys)
            if stat.dist and poi in stat.dist.param_names:
                anchors = stat.dist.param_spec_for(poi).anchors
        if not anchors:
            # Get anchors from the model
            anchors = stat.model.param_spec_for(poi).anchors
        if not anchors:
            raise ValueError("Provide anchors to initially evaluate poi on")
        self.anchors = np.asarray(hypney.utils.eagerpy.ensure_numpy(anchors))
        super().__init__(stat, *args, **kwargs)

    def _compute(self, stat: hypney.Statistic):
        # Note: upper limit boundaries are *low* percentiles
        # of the distribution! See Neyman belt construction diagram.
        crit_quantile = 1 - self.cl

        # Evaluate statistic at anchors
        # (statistic is vectorized over params)
        anchor_pars = {
            self.poi: hypney.utils.eagerpy.to_tensor(self.anchors, match_type=stat.data)
        }
        stat_at_anchors = stat.compute(params=anchor_pars)

        # Should it?
        # if np.diff(stat_at_anchors).min() < 0:
        #     raise ValueError("Statistic should be non-decreasing with POI")

        # Find critical values at anchors.
        # PPF is NOT vectorized, and most efficient when quantile is pre-loaded.
        # TODO: +1 for discrete stats??
        ppf = stat.dist(quantiles=crit_quantile).ppf_
        crit_at_anchors = np.array([ppf(params={self.poi: x}) for x in self.anchors])

        isnan = np.isnan(crit_at_anchors)
        if np.any(isnan):
            raise ValueError(f"ppf returned NaN at anchors {self.anchors[isnan]}")

        # Assume a count-like statistic; crit - stat generally grows with POI
        # => we want to find the highest value where crit <= stat
        crit_minus_stat = crit_at_anchors - stat_at_anchors
        below = np.where(crit_minus_stat <= 0)[0]
        if not len(below):
            raise ValueError(
                f"Upper limit is below the lowest anchor {self.anchors[0]}"
            )
        i_last = below[-1]
        if i_last == len(self.anchors) - 1:
            raise ValueError(
                f"Upper limit exceeds the highest anchor {self.anchors[-1]}"
            )

        # Find zero of (crit - stat) - tiny_offset
        # The offset is needed if crit = stat for an extended length
        # (e.g. for Count)
        # TODO: can we use grad? optimize.root takes a jac arg...
        offset = 1e-3 * (crit_minus_stat[i_last + 1] - crit_minus_stat[i_last])

        def objective(x):
            params = {self.poi: x}
            return ppf(params=params) - self.stat.compute(params=params) - offset

        return optimize.brentq(
            objective, self.anchors[i_last], self.anchors[i_last + 1]
        )
