import itertools
import typing as ty
import warnings

import eagerpy as ep
import numpy as np

import hypney
from .likelihood import SignedPLR

export, __all__ = hypney.exporter()


@export
class DeficitHawk(hypney.Statistic):

    cuts: np.ndarray
    _signal_only_default = False
    _cached_acceptance = itertools.repeat(None)

    def __init__(self, *args, signal_only=None, **kwargs):
        self.signal_only = self._signal_only_default
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
        assert scores.shape[1:] == self.model._batch_shape(params)
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

    Defaults to signed likelihood ratio assuming no background
    """

    _signal_only_default = True

    _observed: np.ndarray = None

    def _init_cut_stats(self):
        """Compute _observed here if not already done earlier"""
        pass

    def _compute_scores(self, n, mu, frac):
        # mu here is expected events in the interval (not total/pre-cuts)
        # TODO: eagerpy-ify
        mu = hypney.utils.eagerpy.ensure_raw(mu)
        n = hypney.utils.eagerpy.ensure_raw(n)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(divide="ignore"):
                result = -2 * np.sign(n - mu) * ((n - mu) + n * np.log(mu / n))
        # For n = 0, return -2 mu
        result = np.where(n == 0, -2 * mu, result)
        return self.model._to_tensor(result)

    def _score_cuts(self, params):
        # (n_cuts)
        frac = self._acceptance(params)
        # ({batch_shape})
        rate = self.model._to_tensor(self.model.rate(params))
        # ({batch_shape}, n_cuts)
        mu = frac * rate[..., None]
        # n is still (n_cuts,); OK for tail-first broadcasting)
        # ({batch_shape}, n_cuts)
        result = self._compute_scores(n=self._observed, mu=mu, frac=frac)
        assert result.shape == mu.shape
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

    # Won't actually do a _profile _likelihood,
    # DeficitHawk restricts to single parameter,
    # so conditional fit optimization is trivial (no optimizer called)
    statistic_class = SignedPLR

    cut_stats: ty.List[hypney.models.CutModel]

    def _init_cut_stats(self):
        """Initialize a statistic for each cut"""
        self.cut_stats = tuple(
            [
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
        # TODO: why is _to_tensor needed here? Shouldn't _compute already
        # return eagerpy tensors?
        result = ep.stack(
            [self.model._to_tensor(stat._compute(params)) for stat in self.cut_stats]
        )
        assert result.shape[1:] == self.model._batch_shape(params)
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
        left, right = self.cuts[:, 0], self.cuts[:, 1]
        # TODO: figure out precise threshold when each algorithm is optimal
        if len(self.cuts) > 1 + np.log(1 + len(self.data)):
            # Many cuts. First sort data [O(n log n)], then use
            # two binary searches to count events [O(ncuts * log n))]
            data = np.sort(self.data[:, 0].raw)
            self._observed = np.searchsorted(data, right) - np.searchsorted(data, left)
        else:
            # Many events. Just loop over data for each cut: O(ncuts * n)
            data = self.data[:, 0].raw
            self._observed = np.asarray(
                [((l < data) * (data < r)).sum() for l, r in zip(left, right)]
            )

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
    # TODO: how to raise proper error (instead of cryptic init failure)
    # if someone tries to init this class directly? Make abc base class?

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
@hypney.utils.numba.maybe_jit
def all_cut_indices(n):
    """Return indices of [l, r>l] cuts ending at n points"""
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


@export
def uniform_sn_cuts(n):
    """Return cuts terminating at the n-quantiles of a uniform(0,1)
    signal distribution"""
    fracs = np.linspace(0, 1, n + 1)
    cuts = np.array(list(itertools.product(fracs, fracs)))
    return cuts[cuts[:, 0] < cuts[:, 1]]
