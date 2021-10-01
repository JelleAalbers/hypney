import warnings

import numpy as np
from scipy import stats

import hypney
from .univariate import UnivariateDistribution

export, __all__ = hypney.exporter()


@export
class OneDHistogram(UnivariateDistribution):
    def __init__(self, histogram, bin_edges=None, *args, **kwargs):
        if bin_edges is None:
            # We probably got some kind of histogram container
            if isinstance(histogram, tuple) and len(histogram) == 2:
                histogram, bin_edges = histogram
            elif hasattr(histogram, "to_numpy"):
                # boost_histogram / hist
                histogram, bin_edges = histogram.to_numpy()
            elif hasattr(histogram, "bin_edges"):
                # multihist
                histogram, bin_edges = histogram.histogram, histogram.bin_edges
            else:
                raise ValueError("Pass histogram and bin edges arrays")

        self.scipy_dist = stats.rv_histogram((histogram, bin_edges),)
        super().__init__(*args, **kwargs)


@export
def from_histogram(histogram, bin_edges=None, *args, **kwargs):
    return OneDHistogram(histogram, bin_edges=bin_edges, *args, **kwargs)


@export
def from_samples(samples, bin_edges=None, bin_count_multiplier=1):
    assert len(samples)
    is_fin = np.isfinite(samples)
    if not np.all(is_fin):
        warnings.warn(
            f"Throwing away {(~is_fin).sum()} non-finite samples out of {len(samples)}"
        )
        samples = samples[is_fin]

    if bin_edges is None:
        # Use the Freedman-Diaconis rule to guess a bin width
        iqr = stats.iqr(samples)
        max_offset = 0
        if iqr == 0:
            # Can happen for discrete statistics, e.g. count with mean 0.01
            iqr = 1.34 * samples.std()
        if iqr == 0:
            # All values are the same; return a delta function
            # rather than a histogram
            return hypney.models.DiracDelta(loc=samples[0])
        width_fd = 2 * iqr / len(samples) ** (1 / 3)

        # The final bin's right edge is inclusive (see np.histogram)
        # So don't fret about masses at the end.
        # But if all values are the same, we need to offset max
        # to obtain a finite bin width.
        mi, ma = samples.min(), samples.max() + max_offset
        bin_edges = np.linspace(
            mi, ma, int(bin_count_multiplier * (ma - mi) / width_fd)
        )

    histogram, bin_edges = np.histogram(samples, bins=bin_edges)
    return from_histogram(histogram, bin_edges)
