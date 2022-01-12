import itertools
from math import prod
import typing as ty

import eagerpy as ep
import numpy as np
from scipy import stats

import hypney
from hypney.utils.numba import factorial
from .likelihood import SignedPLR

export, __all__ = hypney.exporter()


@export
class DeficitHawk(hypney.Statistic):

    cuts: np.ndarray

    def __init__(self, *args, signal_only=False, **kwargs):
        self.signal_only = signal_only
        super().__init__(*args, **kwargs)
        assert (
            len(self.model.param_names) == 1
        ), "DeficitHawk supports only 1-parameter models for now"

    def _init_data(self):
        super()._init_data()
        # In case cuts need to be inferred from data
        self._init_cuts()
        self._init_cut_stats()

    def _compute(self, params):
        # Check for vectorization, undo any (1,) batch shape nonsense
        batch_shape = self.model._batch_shape(params)
        assert prod(batch_shape) == 1, "DeficitHawk is not yet vectorized"
        params = {k: v.reshape((),) for k, v in params.items()}

        scores = self._score_cuts(params)
        result = ep.min(ep.stack(scores), axis=0)

        # Restore any desired (1,) batch shapes... may it do you fine
        return result.reshape(batch_shape)

    def best_cut(self, params=None, **kwargs):
        """Return dict with information about the best cut/region"""
        params = self.model.validate_params(params, **kwargs)
        best_i = self.model.backend.argmin(self._score_cuts(params))
        return self._cut_info(best_i)

    def _init_cuts(self):
        """Set self._cuts based on data (if needed)"""
        pass

    def _init_cut_stats(self):
        """Initialize data for each cut, e.g. compute summary statistics"""
        pass

    def _score_cuts(self, params):
        """Return statistic for each cut, given params"""
        raise NotImplementedError

    def _cut_info(self, cut_i):
        """Return debugging/general info about cut_i"""
        raise NotImplementedError


@export
class SimpleHawk(DeficitHawk):
    """Compute a simple function in each region
    """

    statistic_class = SignedPLR

    _observed: np.ndarray = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.signal_only and self._cached_acceptance is None:
            # Cut efficiencies are constant (wrt params and data)
            self._acceptance = self._get_cached_acceptance
            self._cached_acceptance = self._acceptance()

    def _init_cut_stats(self):
        """Compute _observed here if not already done earlier"""
        pass

    def _compute_scores(self):
        raise NotImplementedError()

    def _score_cuts(self, params):
        frac = self._acceptance(params)
        mu = frac * self.model.rate(params)
        return self._compute_scores(n=self._observed, mu=mu, frac=frac)

    def _cut_info(self, cut_i):
        return dict(i=cut_i, cut=self.cuts[cut_i], n_observed=self._observed[cut_i],)

    def _acceptance(self, params=None):
        """Compute fraction of expected signal passing each cut"""
        raise NotImplementedError

    def _get_cached_acceptance(self, params):
        return self._cached_acceptance


@export
class FullHawk(DeficitHawk):
    """Compute a Statistic in each region"""

    statistic_class: hypney.Statistic

    cut_stats: ty.List[hypney.models.CutModel]

    # FixedRegionFullHawk will fill this if signal_only=True
    _cached_acceptance = itertools.repeat(None)

    def _init_cut_stats(self):
        """Initialize a statistic for each cut"""
        self.cut_stats = tuple(
            [
                # Won't actually run profile likelihood minimizer;
                # DeficitHawk restricts to single parameter
                self.statistic_class(
                    self.model.cut(
                        tuple(cut),
                        cut_data=True,
                        fixed_cut_efficiency=cached_eff,
                        cut_type="open",
                    )
                )
                for cut, cached_eff in zip(self.cuts, self._cached_acceptance)
            ]
        )

    def _score_cuts(self, params):
        return [stat._compute(params) for stat in self.cut_stats]

    def _cut_info(self, cut_i):
        return dict(
            i=cut_i,
            cut=self.cuts[cut_i],
            stat=self.cut_stats[cut_i],
            model=self.cut_stats[cut_i].model,
        )


@export
class FixedRegionSimpleHawk(SimpleHawk):
    def __init__(self, *args, cuts, **kwargs):
        cuts = np.asarray(cuts)
        super().__init__(*args, cuts=cuts, **kwargs)
        # SimpleHawk.__init__ already caches acceptances if signal_only

    def _init_cut_stats(self):
        # O(n log n)
        data = np.sort(self.data)
        # Use two binary searches to count events in each cut
        # O(ncuts * log n)
        self._observed = (
            np.searchsorted(data, self.cuts[:, 1]),
            -np.searchsorted(data, self.cuts[:, 0]),
        )
        # Naive counting would be O(ncuts * n),
        # assuming ncuts > O(log n), the above is faster.

    def _acceptance(self, params):
        return self.model.cdf(data=self.cuts[:, 1], params=params) - self.model.cdf(
            data=self.cuts[:, 0], params=params
        )


@export
class FixedRegionFullHawk(FullHawk):
    def __init__(self, *args, cuts, **kwargs):
        self.cuts = np.asarray(cuts)
        super().__init__(*args, **kwargs)

        if self.signal_only and not isinstance(self._cached_effs, list):
            # Cut efficiencies are constant (wrt params and data).
            # FullHawk._init_cut_stats uses these efficiencies.
            # (unlike SimpleHawk, they are not directly used in statistic
            #  computations, but they are still worth caching)
            self._init_cut_stats()
            self._cached_acceptance = np.asarray(
                [stat.model.cut_efficiency() for stat in self.cut_stats]
            )


@export
class AllRegionHawk:
    def __init__(self, *args, n_max=float("inf"), side_constraint="none", **kwargs):
        self._n_max = n_max
        self._side_constraint = side_constraint
        super().__init__(*args, **kwargs)
        assert (
            self.model.n_dim == 1
        ), "AllRegionHawk only support 1-dimensional models for now"

    def _init_cuts(self):
        # Build all intervals / cuts, 2-tuples of (left, right)
        # OK to use numpy here, outside autodiff anyway
        self._points = np.concatenate(
            [[-float("inf")], np.sort(self.data[:, 0].numpy()), [float("inf")]]
        )
        self._indices = self._build_cut_indices()
        self.cuts = self._points[self._indices]

        i, j = self._indices[:, 0], self._indices[:, 1]
        self._observed = j - (i + 1)

        # TODO: this optimization could be used for fixed cuts too
        if self.signal_only:
            # Compute expected and observed events in each interval
            cdfs = self.model.cdf(data=self._points)
            accs = self._acceptance()

            # Select only the largest (most expected events) cuts
            # for each observed count <= n_max
            largest_cut_i = self._find_largest_cut(
                ns=self._observed,
                acceptances=accs,
                n_max=min(self._n_max, len(self.data)),
            )

            self.cuts, self._cached_acceptance, self._observed, self._indices = [
                x[largest_cut_i]
                for x in (self.cuts, accs, self._observed, self._indices)
            ]

    def _acceptance(self, params=None):
        cdfs = self.model.cdf(data=self._points, params=params)
        i, j = self._indices[:, 0], self._indices[:, 1]
        return cdfs[..., j] - cdfs[..., i]

    def _build_cut_indices(self, side_constraint=None):
        n_points = len(self._points)
        if side_constraint is None:
            side_constraint = self._side_constraint
        if side_constraint == "none":
            # (X, Y > X)
            indices = np.stack(np.indices((n_points, n_points)), axis=-1).reshape(-1, 2)
            return indices[indices[:, 1] > indices[:, 0]]
        elif side_constraint == "left":
            # (0, X)
            return np.stack(
                [np.zeros(n_points - 1, dtype=np.int), np.arange(1, n_points)], axis=-1
            )
        elif side_constraint == "right":
            # (X, n_points - 1)
            return np.stack(
                [
                    np.arange(0, n_points - 1),
                    np.full(n_points - 1, n_points - 1, dtype=np.int),
                ],
                axis=-1,
            )
        elif side_constraint == "both":
            # left and right-constrainted intervals
            return np.concatenate(
                [
                    self._build_cut_indices(n_points, "left"),
                    self._build_cut_indices(n_points, "right"),
                ]
            )
        raise ValueError(f"Unknown side constraint {side_constraint}")

    @staticmethod
    @hypney.utils.numba.maybe_jit
    def _find_largest_cut(ns, acceptances, n_max):
        """Find cut with most events expected for each N_observed
        (= largest deficit)
        """
        # Note n_max+1 possible N values, [0, n_max]
        largest_acc = np.zeros(n_max + 1)
        largest_i = np.zeros(n_max + 1, dtype=np.int64)
        for i, (n, mu) in enumerate(zip(ns, acceptances)):
            if n > n_max:
                continue
            if acceptances[i] > largest_acc[n]:
                largest_acc[n] = mu
                largest_i[n] = i
        return largest_i


@export
class AllRegionFullHawk(AllRegionHawk, FullHawk):
    pass


@export
class AllRegionSimpleHawk(AllRegionHawk, SimpleHawk):
    # AllRegionHawk provides _acceptance; _init_cuts computes _observed
    pass


@export
class PNOneCut(hypney.Statistic):
    """Return Poisson P of observing <= the observed N events

    Yellin's pmax method uses P(observing more events), i.e. 1 - this.
    (We want something that yields LOW value for deficits instead,
     since we take the minimum over cuts/regions.)
    """

    def _dist_params(self, params):
        # Distribution depends only on # expected events in the region
        return dict(mu=self.model._rate(params))

    def _compute(self, params):
        mu = self.model._rate(params)
        n = len(self.data)
        if self.model._backend_name == "numpy":
            # Shortcut, avoids some overhead... :-(
            q = stats.poisson.cdf(n, mu=mu.raw)
            return self.model._to_tensor(q)
        poisson_model = hypney.models.poisson(mu=mu)
        return poisson_model.cdf(n)


@export
class PNOneCutAugmented(PNOneCut):
    _augment = 0

    def _init_data(self):
        self._augment = float(np.random.rand())
        return super()._init_data()

    def _compute(self, params):
        mu = self.model._rate(params)
        n = len(self.data)
        if self.model._backend_name == "numpy":
            # Shortcut, avoids some overhead... :-(
            # TODO: test if still faster with augment
            q = stats.poisson.cdf(n, mu=mu.raw)
            q += self._augment * stats.poisson.pmf(n + 1, mu=mu.raw)
            return self.model._to_tensor(q)
        poisson_model = hypney.models.poisson(mu=mu)
        return poisson_model.cdf(n) + self._augment * poisson_model.pdf(n + 1)


@export
class PNFullHawk(AllRegionFullHawk):

    statistic_class = PNOneCut

    def _dist_params(self, params):
        # Distribution depends only on # expected events in the region
        return dict(mu=self.model._rate(params))


@export
class PNSimpleHawk(AllRegionSimpleHawk):
    def _dist_params(self, params):
        # Distribution depends only on # expected events
        return dict(mu=self.model._rate(params))

    def _compute_scores(self, n, mu, frac):
        return self.model._to_tensor(
            stats.poisson.cdf(n, mu=hypney.utils.eagerpy.ensure_numpy(mu))
        )


@export
class PNHawk(PNSimpleHawk):
    # Alias
    pass


@export
class YellinP(PNHawk):
    # Alias
    pass


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
        # TODO: This won't parallelize well.. maybe we should just load on import?
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

        # Get original rate after cuts... not very clean
        # TODO: this won't vectorize / autodiff!!
        mu = hypney.utils.eagerpy.ensure_float(self.model._orig_model._rate(params))
        n = len(self.data)
        frac = self.model.cut_efficiency(params)

        # Minus, since deficit hawks take the minimum over cuts
        return -self.p_smaller_itv(mu, n, frac)

    def p_smaller_itv(self, mu, n, frac):
        """Probability of finding a largest N-event-containing interval
        smaller than frac (i.e. with less fraction of expected signal)
        """
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
class YellinOptItv(AllRegionFullHawk):

    statistic_class = OptItvOneCut

    def _dist_params(self, params):
        # Distribution depends only on # expected events in the region
        return dict(mu=self.model._rate(params))
