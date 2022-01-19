import gzip
import pickle

import numpy as np
import multihist  # pickle contains a multihist
from scipy.interpolate import RegularGridInterpolator

import hypney
from hypney.utils.numba import factorial
from .deficit_hawks import AllRegionFullHawk, AllRegionSimpleHawk

export, __all__ = hypney.exporter()


# TODO: where to store file in repo?
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

p_smaller_x_itp = RegularGridInterpolator(points, cdfs)
itp_max_mu = mh.bin_centers("mu").max()


def p_smaller_itv(n, mu, frac):
    """Probability of finding a largest N-event-containing interval
    smaller than frac (i.e. with less fraction of expected signal)
    """
    # TODO: doesn't vectorize I guess...
    return p_smaller_x_itp([mu, n, frac]).item()


class YellinOptItv(hypney.Statistic):
    def _dist_params(self, params):
        # Distribution depends only on # expected events
        return dict(mu=self.model._rate(params))

    def _compute_scores(self, n, mu, frac):
        return -p_smaller_itv(n, hypney.utils.eagerpy.ensure_numpy(mu), frac)


##
# Old slow implementation. Not sure it works actually..
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

    def _compute(self, params):
        assert self.model._backend_name == "numpy"

        # Get original rate after cuts... not very clean
        # TODO: this won't vectorize / autodiff!!
        mu = hypney.utils.eagerpy.ensure_float(self.model._orig_model._rate(params))
        n = len(self.data)
        frac = self.model.cut_efficiency(params)

        # Minus, since deficit hawks take the minimum over cuts
        return -p_smaller_itv(mu, n, frac)


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
class YellinOptItvSlow(AllRegionFullHawk):

    statistic_class = OptItvOneCut

    def _dist_params(self, params):
        # Distribution depends only on # expected events in the region
        return dict(mu=self.model._rate(params))
