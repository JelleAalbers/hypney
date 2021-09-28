import hypney

from .univariate import UnivariateDistribution

from scipy import stats


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
