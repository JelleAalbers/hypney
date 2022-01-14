import itertools
from math import prod
import typing as ty

import eagerpy as ep
import numba
import numpy as np
from scipy import stats

import hypney
from hypney.utils.numba import factorial
from .likelihood import SignedPLR

export, __all__ = hypney.exporter()


@export
class DeficitHawk(hypney.Statistic):

    cuts: np.ndarray
    _cached_acceptance = itertools.repeat(None)

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
        # (n_cuts, {batch_shape})
        scores = self._score_cuts(params)
        # assert scores.shape[1:] == self.model._batch_shape(params)
        return ep.min(scores, axis=0)

    def best_cut(self, params=None, **kwargs):
        """Return dict with information about the best cut/region"""
        params = self.model.validate_params(params, **kwargs)
        # TODO: this will probably crash.. batch shape nonsense
        best_i = self.model.backend.argmin(self._score_cuts(params), axis=0)
        return self._cut_info(best_i)

    def _init_cuts(self):
        """Set self._cuts based on data (if needed)"""
        pass

    def _init_cut_stats(self):
        """Initialize data for each cut, e.g. compute summary statistics"""
        pass

    def _score_cuts(self, params):
        """Return (n_cuts, {batch_shape} with statistic for each cut, given params"""
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

    def _init_cut_stats(self):
        """Compute _observed here if not already done earlier"""
        pass

    def _compute_scores(self):
        raise NotImplementedError()

    def _score_cuts(self, params):
        # (n_cuts)
        frac = self._acceptance(params)
        # ({batch_shape})
        rate = self.model._to_tensor(self.model.rate(params))
        # ({batch_shape}, n_cuts)
        mu = frac * rate[..., None]
        # ({batch_shape}, n_cuts)
        result = self._compute_scores(n=self._observed, mu=mu, frac=frac)
        # (n_cuts, {batch_shape})
        axis_order = tuple(np.roll(np.arange(len(result.shape)), 1).tolist())
        return result.transpose(axis_order)

    def _cut_info(self, cut_i):
        return dict(i=cut_i, cut=self.cuts[cut_i], n_observed=self._observed[cut_i],)

    def _acceptance(self, params=None):
        """Compute fraction of expected signal passing each cut"""
        raise NotImplementedError

    def _get_cached_acceptance(self, params=None):
        return self._cached_acceptance


@export
class FullHawk(DeficitHawk):
    """Compute a Statistic in each region"""

    statistic_class: hypney.Statistic

    cut_stats: ty.List[hypney.models.CutModel]

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
        # (n_cuts, {batch_shape})
        result = ep.stack([stat._compute(params) for stat in self.cut_stats])
        # assert result.shape[1:] == self.model._batch_shape(params)
        return result

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
        self.cuts = np.asarray(cuts)
        super().__init__(*args, **kwargs)
        if self.signal_only and self._acceptance != self._get_cached_acceptance:
            # Cut efficiencies are constant (wrt params and data)
            # and because cuts are fixed, we can compute them now
            # (even if we don't yet have data)
            self._cached_acceptance = self._acceptance()
            self._acceptance = self._get_cached_acceptance

    def _init_cut_stats(self):
        # O(n log n)
        data = np.sort(self.data[:, 0])
        # Use two binary searches to count events in each cut
        # O(ncuts * log n)
        # TODO: implement cut types / openness.
        self._observed = (
            np.searchsorted(data, self.cuts[:, 1])
            - np.searchsorted(data, self.cuts[:, 0]),
        )
        # Naive counting would be O(ncuts * n),
        # assuming ncuts > O(log n), the above is faster.

    def _acceptance(self, params=None):
        return self.model.cdf(data=self.cuts[:, 1], params=params) - self.model.cdf(
            data=self.cuts[:, 0], params=params
        )


@export
class FixedRegionFullHawk(FullHawk):
    def __init__(self, *args, cuts, **kwargs):
        self.cuts = np.asarray(cuts)
        super().__init__(*args, **kwargs)

    def _init_cut_stats(self):
        super()._init_cut_stats()

        if self.signal_only and not isinstance(self._cached_effs, list):
            # Cut efficiencies are constant (wrt params and data).
            # FullHawk._init_cut_stats uses these efficiencies.
            # (unlike SimpleHawk, they are not directly used in statistic
            #  computations, but they are still worth caching)
            self._cached_acceptance = np.asarray(
                [stat.model.cut_efficiency() for stat in self.cut_stats]
            )
            self._acceptance = self._get_cached_acceptance


@export
class AllRegionHawk:
    def __init__(
        self,
        *args,
        n_max=float("inf"),
        regions_type="all",
        central_point=None,
        **kwargs,
    ):
        self._n_max = n_max
        self._regions_type = regions_type
        # For regions_type = "central" / "central_symmetric"
        self._central_point = central_point

        super().__init__(*args, **kwargs)
        assert (
            self.model.n_dim == 1
        ), "AllRegionHawk only support 1-dimensional models for now"

    def _init_cuts(self):
        # Build all intervals / cuts, 2-tuples of (left, right)
        # OK to use numpy here, outside autodiff anyway
        x = self.data[:, 0].numpy()
        if self._regions_type == "central_symmetric":
            # Create 'mirror events' reflected in the central point
            x = np.concatenate([x, self._central_point - (x - self._central_point)])

        self._points = np.concatenate([[-float("inf")], np.sort(x), [float("inf")]])
        self._indices = self._build_cut_indices()
        self._observed = np.diff(self._indices, axis=1)[:, 0] - 1

        if self._regions_type == "central_symmetric":
            # Don't count artificial mirror events
            assert np.all(self._observed % 2) == 0
            self._observed = self._observed // 2

        # TODO: this optimization could be used for fixed cuts too
        if self.signal_only:
            # Compute expected and observed events in each interval
            accs = self._acceptance()

            # Select only the largest (most expected events) cuts
            # for each observed count <= n_max
            largest_cut_i = self._find_largest_cut(
                ns=self._observed,
                acceptances=accs,
                n_max=min(self._n_max, len(self.data)),
            )

            self._cached_acceptance, self._observed, self._indices = [
                x[largest_cut_i] for x in (accs, self._observed, self._indices)
            ]

        self.cuts = self._points[self._indices]

    def _acceptance(self, params=None):
        cdfs = self.model.cdf(data=self._points, params=params)
        # Shorter but less clear, seems equally fast
        # return np.diff(cdfs[..., self._indices], axis=-1)[...,0]
        i, j = self._indices[:, 0], self._indices[:, 1]
        return cdfs[..., j] - cdfs[..., i]

    def _build_cut_indices(self, regions_type=None):
        n_points = len(self._points)
        if regions_type is None:
            regions_type = self._regions_type

        if isinstance(regions_type, (tuple, list)):
            # Combine several types of regions
            # (Do not allow central_symmetric, which requires mirroring events)
            assert "central_symmetric" not in regions_type
            return np.concatenate([self._build_cut_indices(r) for r in regions_type])
        if regions_type == "all":
            # (X, Y > X)
            return all_cut_indices(n_points)
        elif regions_type == "left":
            # (0, X)
            return np.stack(
                [np.zeros(n_points - 1, dtype=np.int), np.arange(1, n_points)], axis=-1
            )
        elif regions_type == "right":
            # (X, n_points - 1)
            return np.stack(
                [
                    np.arange(0, n_points - 1),
                    np.full(n_points - 1, n_points - 1, dtype=np.int),
                ],
                axis=-1,
            )
        elif regions_type == "central":
            indices = self._build_cut_indices(regions_type="all")
            return indices[
                (self._points[indices[:, 0]] <= self._central_point)
                & (self._points[indices[:, 1]] >= self._central_point)
            ]
        elif regions_type == "central_symmetric":
            # Mirror events have been created around the central point
            assert n_points % 2 == 0
            # Index of closest event left of central point
            left_i = (n_points // 2) - 1
            right_i = left_i + 1
            n_shift = np.arange(left_i + 1, dtype=np.int)
            return np.stack([(left_i - n_shift), (right_i + n_shift)], axis=-1)

        raise ValueError(f"Unknown side constraint {regions_type}")

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
@numba.njit
def all_cut_indices(n):
    """Return indices of [l, r>l] cuts on n events"""
    # ~10x faster than stacking and cutting np.indices(n,n)
    result = np.empty((n * (n - 1) // 2, 2), np.int32)
    i = 0
    for l in range(n):
        for r in range(l + 1, n):
            result[i] = l, r
            i += 1
    return result


@export
class AllRegionFullHawk(AllRegionHawk, FullHawk):
    pass


@export
class AllRegionSimpleHawk(AllRegionHawk, SimpleHawk):
    # AllRegionHawk provides _acceptance; _init_cuts computes _observed
    pass


##
# P(observing fewer events)
##


@export
class PNHawk:
    def _dist_params(self, params):
        # Distribution depends only on # expected events
        return dict(mu=self.model._rate(params))

    def _compute_scores(self, n, mu, frac):
        return self.model._to_tensor(
            stats.poisson.cdf(n, mu=hypney.utils.eagerpy.ensure_numpy(mu))
        )


@export
class PNAllRegionHawk(PNHawk, AllRegionSimpleHawk):
    pass


@export
class PNFixedRegionHawk(PNHawk, FixedRegionSimpleHawk):
    pass


# @export
# class YellinP(PNAllRegionHawk):
#     # Alias
#     pass


# Alternate implementation via Full hawk. Should be much slower!


@export
class PNOneCut(hypney.Statistic):
    """Return Poisson P of observing <= the observed N events

    Yellin's pmax method uses P(observing more events), i.e. 1 - this.
    (We want something that yields LOW value for deficits instead,
     since we take the minimum over cuts/regions.)
    """

    def _compute(self, params):
        # ({batch_shape})
        mu = self.model._rate(params)
        n = len(self.data)
        if self.model._backend_name == "numpy":
            # Shortcut, avoids some overhead... :-(
            q = stats.poisson.cdf(n, mu=mu.raw)
            return self.model._to_tensor(q)
        poisson_model = hypney.models.poisson(mu=mu)
        return poisson_model.cdf(n)


@export
class PNFixedRegionHawkSlow(FixedRegionFullHawk):

    statistic_class = PNOneCut

    def _dist_params(self, params):
        # Distribution depends only on # expected events in the region
        return dict(mu=self.model._rate(params))


@export
class PNAllRegionHawkSlow(AllRegionFullHawk):

    statistic_class = PNOneCut

    def _dist_params(self, params):
        # Distribution depends only on # expected events in the region
        return dict(mu=self.model._rate(params))


##
# Optimum interval
##


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


@export
def uniform_sn_cuts(n):
    """Return cuts terminating at the n-quantiles of a uniform(0,1)
    signal distribution"""
    fracs = np.linspace(0, 1, n + 1)
    cuts = np.array(list(itertools.product(fracs, fracs)))
    return cuts[cuts[:, 0] < cuts[:, 1]]
