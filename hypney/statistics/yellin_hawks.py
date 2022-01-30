"""Deficit hawks using Yellin's CN statistic

i.e. the original optimum interval method.
"""
import gzip
import pickle
from pathlib import Path

import numpy as np
import multihist  # pickle contains a multihist
from scipy.interpolate import RegularGridInterpolator

import hypney
from hypney.utils.numba import factorial
from .deficit_hawks import AllRegionFullHawk, AllRegionSimpleHawk

export, __all__ = hypney.exporter()


# TODO: where to store file in repo?
cn_cdf_file = "/home/jaalbers/Documents/projects/robust_inference_2/cn_cdf.pkl.gz"

if Path(cn_cdf_file).exists():
    with gzip.open(cn_cdf_file) as f:
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

    p_smaller_x_itp = RegularGridInterpolator(points, cdfs, bounds_error=False)
    itp_max_mu = mh.bin_centers("mu").max()

else:
    p_smaller_x_itp = None


@export
def p_smaller_itv(n, mu, frac):
    """Probability of finding a largest N-event-containing interval
    smaller than frac (i.e. with less fraction of expected signal)

    Args:
     - n: observed events
     - mu: *total* expected events (not expected event in interval!)
     - frac: fraction of events expected in interval
    """
    if p_smaller_x_itp is None:
        raise FileNotFoundError(
            f"CN distribution file {cn_cdf_file} not found, cannot compute p_smaller_itv"
        )
    # I'm assuming mu has the largest shape. Ravel may be inefficient but
    # I think RegularGridInterpolator won't work without it
    # TODO: eagerpy-ify this.
    mu = hypney.utils.eagerpy.ensure_numpy(mu)
    was_float = isinstance(mu, (int, float))
    mu = np.asarray(mu)
    n = np.asarray(0 * mu + n)
    frac = np.asarray(0 * mu + frac)

    points = np.stack([mu.ravel(), n.ravel(), frac.ravel()]).T
    result = p_smaller_x_itp(points)
    result = result.reshape(mu.shape)
    if was_float:
        return result.item()
    return result


@export
class YellinCNHawk(AllRegionSimpleHawk):
    def _dist_params(self, params):
        # Distribution depends only on # expected events
        return dict(mu=self.model._rate(params))

    def _compute_scores(self, n, mu, frac):
        return -p_smaller_itv(n=n, mu=mu / frac, frac=frac)


##
# Alternate implementation as a full hawk
# use only for testing; it's just slower than YellinCNHawk!
##


@export
class YellinCN(hypney.Statistic):
    """Computes - C_n(x, mu) for one interval

    Here, C_n(x, mu) is the probability of finding a largest N-event-containing
    interval smaller than frac (i.e. with less fraction of expected signal)
    given the true rate mu.
    """

    def _compute(self, params):
        assert self.model._backend_name == "numpy"

        mu = self.model.rate(params)
        n = len(self.data)

        assert isinstance(self.model, hypney.models.CutModel)
        frac = self.model.cut_efficiency(params)

        # Minus, since deficit hawks take the minimum over cuts
        result = -p_smaller_itv(n=n, mu=mu / frac, frac=frac)
        return result


@export
class YellinCNFullHawk(AllRegionFullHawk):

    statistic_class = YellinCN

    def _dist_params(self, params):
        # Distribution depends only on # expected events in the region
        return dict(mu=self.model._rate(params))


# Not really needed but useful for testing


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
