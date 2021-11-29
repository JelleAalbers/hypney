import math

import numpy as np
import eagerpy as ep

import hypney
from .likelihood import SignedPLR

export, __all__ = hypney.exporter()


@export
class DeficitHawk(hypney.Statistic):

    cuts: tuple
    cut_stats: tuple
    statistic_class = SignedPLR

    def __init__(self, *args, signal_only=False, **kwargs):
        self.signal_only = signal_only

        super().__init__(*args, **kwargs)

        assert len(self.model.param_names) == 1, \
            "DeficitHawk supports only 1-parameter models for now"

    def _init_data(self):
        super()._init_data()
        self._init_cut_stats()

    def _init_cut_stats(self):
        """Initialize a statistic for each cut"""
        self.cut_stats = tuple([
            # Won't actually profile; DeficitHawk restricts to single parameter
            self.statistic_class(self.model.cut(cut, cut_data=True))
            for cut in self.cuts
        ])

    def _compute(self, params):
        return min(self.score_cuts(params))

    def score_cuts(self, params):
        return [stat._compute(params) for stat in self.cut_stats]

    # For extra info / debugging

    def best_cut(self, params=None):
        params = self.model.validate_params(params)
        best_i = self.model.backend.argmin(self.score_cuts(params))
        return self._cut_info(params, best_i)

    def _cut_info(self, params, cut_i):
        return dict(
            i=cut_i,
            cut=self.cuts[cut_i],
            stat=self.cut_stats[cut_i],
            model=self.cut_stats[cut_i].model,
        )


@export
class FixedRegionHawk(DeficitHawk):

    def __init__(self, *args, cuts, **kwargs):
        self.cuts = cuts
        super().__init__(*args, **kwargs)


@export
class AllRegionHawk(DeficitHawk):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.model.n_dim == 1, \
            "AllRegionHawk only support 1-dimensional models for now"

    def _init_cut_stats(self):
        # Build all intervals / cuts, 2-tuples of (left, right)
        # OK to use numpy here, outside autodiff anyway
        points = np.concatenate([
            [-float('inf')],
            np.sort(self.data[:,0].numpy()),
            [float('inf')]])
        n_points = len(points)
        indices = np.stack(np.indices((n_points, n_points)), axis=-1).reshape(-1, 2)
        indices = indices[indices[:,1] > indices[:,0]]
        cuts = points[indices]

        # TODO: this optimization could be used for fixed cuts too
        if self.signal_only:
            # Compute expected and observed events in each interval
            cdfs = self.model.cdf(points)
            i, j = indices[:, 0], indices[:, 1]
            acceptances = cdfs[j] - cdfs[i]
            ns = j - (i + 1)

            largest_cut_i = self._find_largest_cut(
                ns, acceptances, len(self.data))

            self.cuts, self.acceptances, self.ns = [
                x[largest_cut_i]
                for x in (cuts, acceptances, ns)]
            self.cuts = self.cuts.tolist()

        else:
            self.cuts = cuts.tolist()

        super()._init_cut_stats()


    @staticmethod
    @hypney.utils.numba.maybe_jit
    def _find_largest_cut(ns, acceptances, n_max):
        """Find cut with most events expected for each N_observed
        (= largest deficit)
        """
        # Note n+1 possible N values, [0, n]
        largest_acc = np.zeros(n_max + 1)
        largest_i = np.zeros(n_max + 1, dtype=np.int64)
        for i, (n, mu) in enumerate(zip(ns, acceptances)):
            if acceptances[i] > largest_acc[n]:
                largest_acc[n] = mu
                largest_i[n] = i
        return largest_i


@export
class PMaxOneCut(hypney.Statistic):
    """Return Poisson P of observing <= the observed N events"""

    def _compute(self, params):
        mu = self.model.rate(params)
        n = len(self.data)
        return hypney.models.poisson(rate=mu).cdf(n)


@export
class YellinPMax(AllRegionHawk):

    statistic_class = PMaxOneCut


@export
class OptItvOneCut(hypney.Statistic):

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        assert isinstance(self.model, hypney.models.CutModel)

    def _compute(self, params):
        mu = self.model.rate(params)
        n = len(self.data)
        x = self.model.cut_efficiency(params) * mu
        # Note minus:
        return -self.p_smaller_itv(n, mu, x)

    def p_smaller_x(self, n, mu, x):
        """Yellin's C_N(mu, x)
        Probability of finding a largest N-event-containing interval smaller
        than x
        """
        if n == 0:
            # Equation 2 from https://arxiv.org/pdf/physics/0203002.pdf
            m = int(math.ceil(mu/x))
            return sum([
                (k * x - mu)**k * math.exp(-k * x)/math.factorial(k)
                * (1 + k / (mu - k * x))
                for k in range(0, m + 1)])
        else:
            raise NotImplementedError


@export
class YellinOptItv(AllRegionHawk):

    statistic_class = OptItvOneCut