import math

import numpy as np
import eagerpy as ep
from scipy import stats
from scipy.interpolate.interpolate import RegularGridInterpolator

import hypney
from hypney.utils.numba import factorial
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

        assert (
            len(self.model.param_names) == 1
        ), "DeficitHawk supports only 1-parameter models for now"

    def _init_data(self):
        super()._init_data()
        self._init_cut_stats()

    def _init_cut_stats(self):
        """Initialize a statistic for each cut"""
        self.cut_stats = tuple(
            [
                # Won't actually run profile likelihood minimizer;
                # DeficitHawk restricts to single parameter
                self.statistic_class(
                    self.model.cut(cut, cut_data=True, cut_type="open")
                )
                for cut in self.cuts
            ]
        )

    def _compute(self, params):
        return min(self._score_cuts(params))

    def _score_cuts(self, params):
        return [stat._compute(params) for stat in self.cut_stats]

    # For extra info / debugging

    def best_cut(self, params=None, **kwargs):
        params = self.model.validate_params(params, **kwargs)
        best_i = self.model.backend.argmin(self._score_cuts(params))
        return self._cut_info(best_i)

    def _cut_info(self, cut_i):
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
        assert (
            self.model.n_dim == 1
        ), "AllRegionHawk only support 1-dimensional models for now"

    def _init_cut_stats(self):
        # Build all intervals / cuts, 2-tuples of (left, right)
        # OK to use numpy here, outside autodiff anyway
        points = np.concatenate(
            [[-float("inf")], np.sort(self.data[:, 0].numpy()), [float("inf")]]
        )
        n_points = len(points)
        indices = np.stack(np.indices((n_points, n_points)), axis=-1).reshape(-1, 2)
        indices = indices[indices[:, 1] > indices[:, 0]]
        cuts = points[indices]

        # TODO: this optimization could be used for fixed cuts too
        if self.signal_only:
            # Compute expected and observed events in each interval
            cdfs = self.model.cdf(points)
            i, j = indices[:, 0], indices[:, 1]
            acceptances = cdfs[j] - cdfs[i]
            ns = j - (i + 1)

            largest_cut_i = self._find_largest_cut(ns, acceptances, len(self.data))

            self.cuts, self.acceptances, self.ns = [
                x[largest_cut_i] for x in (cuts, acceptances, ns)
            ]
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
class YellinPOneCut(hypney.Statistic):
    """Return Poisson P of observing <= the observed N events

    Yellin's pmax method uses P(observing more events), i.e. 1 - this.
    (We want something that yields LOW value for deficits instead,
     since we take the minimum over cuts/regions.)
    """

    def _compute(self, params):
        mu = self.model._rate(params)
        n = len(self.data)
        return hypney.models.poisson(mu=mu).cdf(n)


@export
class YellinP(AllRegionHawk):

    statistic_class = YellinPOneCut


@export
class OptItvOneCut(hypney.Statistic):
    """Computes - C_n(x, mu) for one interval

    Here, C_n(x, mu) is the probability of finding a largest N-event-containing
    interval smaller than frac (i.e. with less fraction of expected signal)
    given the true rate mu.
    """

    # TODO For some reason we get a recursion error when overriding init
    # as we call super().__init___..
    def _init_data(self, *args, **kwargs):
        # We use cut_efficiency in the computations below
        assert isinstance(self.model, hypney.models.CutModel)
        self._ensure_cn_table_loaded()

    def _ensure_cn_table_loaded(self):
        # load the p_smaller_x table; store it with the class
        # so we don't load this for every interval!
        # TODO: where to store file in repo?
        # TODO: This won't paralellize well.. maybe we should just load on import?
        if hasattr(self, "_p_smaller_x_itp"):
            return

        import gzip
        import pickle
        import multihist
        from scipy.interpolate import RegularGridInterpolator

        with gzip.open(
            "/home/jaalbers/Documents/projects/robust_inference_2/cn_cdf.pkl.gz"
        ) as f:
            mh = pickle.load(f)

        # Pad with zero at frac = 0
        cdfs = np.pad(mh.histogram, [(0, 0), (0, 0), (1, 0)])

        points = mh.bin_centers()
        # We cumulated along the 'frac' dimension; values represent
        # P(frac <= right bin edge)
        points[2] = np.concatenate([[0], mh.bin_edges[2][1:]])

        # Full intervals (frac = 1) should not always score 1.
        # Linearly interpolate the last cdf bin instead:
        cdfs[:, :, -1] = (cdfs[:, :, -2] + (cdfs[:, :, -2] - cdfs[:, :, -3])).clip(0, 1)

        self.__class__._p_smaller_x_itp = RegularGridInterpolator(points, cdfs)
        self.__class__._itp_max_mu = mh.bin_centers("mu").max()

    def _compute(self, params):
        assert self.model._backend_name == "numpy"

        n = len(self.data)
        frac = self.model.cut_efficiency(params)

        # Get original rate after cuts... not very clean
        # TODO: this won't vectorize / autodiff!!
        mu = hypney.utils.eagerpy.ensure_float(self.model._orig_model._rate(params))

        # Minus, since deficit hawks take the minimum over cuts
        return -self.p_smaller_itv(mu, n, frac)

    def p_smaller_itv(self, mu, n, frac):
        """Probability of finding a largest N-event-containing interval
        smaller than frac (i.e. with less fraction of expected signal)
        """
        # if mu > self._itp_max_mu:
        #     # Above interpolator range: use maximum gap
        #     # TODO: this should throw some warning
        #     if n == 0:
        #         # Use exact formula for gaps
        #         return p_smaller_x_0(mu, frac)
        #     else:
        #         # Return giant meaningless value for other intervals
        #         return float('inf')
        return self.__class__._p_smaller_x_itp([mu, n, frac]).item()


@hypney.utils.numba.maybe_jit
def p_smaller_x_0(mu, frac):
    # Equation 2 from https://arxiv.org/pdf/physics/0203002.pdf
    # The factorial causes OverflowError for sufficiently small/unlikely fracs
    # TODO: maybe put 0 instead of raising exception?
    x = frac * mu
    m = int(np.floor(mu / x))
    ks = np.arange(0, m + 1)
    return (
        (ks * x - mu) ** ks * np.exp(-ks * x) / factorial(ks) * (1 + ks / (mu - ks * x))
    ).sum()


@export
class YellinOptItv(AllRegionHawk):

    statistic_class = OptItvOneCut
