import warnings

import numpy as np
from scipy import stats

import hypney
from .univariate import UnivariateDistribution

export, __all__ = hypney.exporter()


class rv_histogram(stats.rv_histogram):
    # With fix https://github.com/scipy/scipy/pull/15381

    def __init__(self, histogram, *args, **kwargs):
        self._histogram = histogram
        if len(histogram) != 2:
            raise ValueError("Expected length 2 for parameter histogram")
        counts_per_bin = np.asarray(histogram[0])
        counts_total = counts_per_bin.sum()
        self._hbins = np.asarray(histogram[1])
        if len(counts_per_bin) + 1 != len(self._hbins):
            raise ValueError(
                "Number of elements in histogram content "
                "and histogram boundaries do not match, "
                "expected n and n+1."
            )
        self._hbin_widths = self._hbins[1:] - self._hbins[:-1]
        self._hpdf = counts_per_bin / (counts_total * self._hbin_widths)
        self._hcdf = np.cumsum(counts_per_bin) / counts_total
        self._hpdf = np.hstack([0.0, self._hpdf, 0.0])
        self._hcdf = np.hstack([0.0, self._hcdf])
        # Set support
        kwargs["a"] = self.a = self._hbins[0]
        kwargs["b"] = self.b = self._hbins[-1]
        stats.rv_continuous.__init__(self, *args, **kwargs)


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

        self.scipy_dist = rv_histogram((histogram, bin_edges),)
        super().__init__(*args, **kwargs)


@export
def from_histogram(histogram, bin_edges=None, *args, **kwargs):
    return OneDHistogram(histogram, bin_edges=bin_edges, *args, **kwargs)


@export
def from_samples(samples, bin_edges=None, bin_count_multiplier=1, mass_bins=False):
    samples = np.asarray(samples)
    assert len(samples)
    is_fin = np.isfinite(samples)
    if not np.all(is_fin):
        warnings.warn(
            f"Throwing away {(~is_fin).sum()} non-finite samples out of {len(samples)}"
        )
        samples = samples[is_fin]

    if bin_edges is None:
        bin_edges = guess_bin_edges(
            samples, bin_count_multiplier=bin_count_multiplier, mass_bins=mass_bins
        )
    if not len(bin_edges):
        # All samples are the same, return a delta function distribution
        return hypney.models.DiracDelta(loc=samples[0])

    histogram, bin_edges = np.histogram(samples, bins=bin_edges)
    if mass_bins:
        # Fixed the location of the mass bins
        # causes slight numerical errors
        bin_edges = np.nextafter(bin_edges, float("-inf"))

    return from_histogram(histogram, bin_edges)


@export
def guess_bin_edges(samples, bin_count_multiplier=1, mass_bins=False):
    samples = np.asarray(samples)

    if mass_bins:
        # Search for and temporarily remove probability masses
        from collections import Counter

        c = Counter(samples)
        masses = np.array([mass for mass, count in c.items() if count > 5])
        samples = samples[~np.in1d(samples, masses)]

    # Use the Freedman-Diaconis rule to guess a bin width
    if len(samples):
        iqr = stats.iqr(samples)
        if iqr == 0:
            # Can happen for discrete statistics, e.g. count with mean 0.01
            iqr = 1.34 * samples.std()
    else:
        iqr = 0

    if iqr == 0:
        # All samples are the same!
        result = []
    else:
        width_fd = 2 * iqr / len(samples) ** (1 / 3)

        mi, ma = samples.min(), samples.max()
        result = np.linspace(mi, ma, int(bin_count_multiplier * (ma - mi) / width_fd))

    if mass_bins:
        # Add ultratight bins for the probability masses
        # Should work for cdf/ppf, though pdf etc. will be horrible.
        result = np.sort(
            np.concatenate([result, masses, np.nextafter(masses, masses + 1)])
        )

    return result
