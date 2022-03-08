from functools import partial
import warnings

import numpy as np
from scipy import stats

import hypney

export, __all__ = hypney.exporter()


@export
def progress_iter(progress=False, **kwargs):
    if progress is True:
        try:
            from tqdm import tqdm

            return partial(tqdm, **kwargs)
        except ImportError:
            warnings.warn("Progress bar requested but tqdm did not import")
            return lambda x: x
    elif progress:
        # Custom progress iterator, e.g. tqdm.notebook
        return partial(progress, **kwargs)
    else:
        return lambda x: x


@export
def flexible_clip(x, xmin=-float("inf"), xmax=float("inf")):
    """clip that works on scalars, eagerpy and raw tensors"""
    if isinstance(x, (float, int)):
        return min(max(x, xmin), xmax)
    return x.clip(xmin, xmax)


@export
def poisson_ul(n, mu_bg=0, cl=0.9):
    """Upper limit on mu_signal, from observing n events
    where mu_bg background events were expected

    NB: can be negative if mu_bg large enough.
    It's your responsibility to clip to 0...
    """
    # Adapted from https://stackoverflow.com/a/14832525
    return stats.chi2.ppf(cl, 2 * n + 2) / 2 - mu_bg


@export
def poisson_ll(n, cl=0.9):
    # Adapted from https://stackoverflow.com/a/14832525
    # Checked through
    #     n = np.arange(0, 100)
    #     stats.poisson(poisson_ul(n)).cdf(n)
    # (and similarly for upper limit)
    n = np.asarray(n)
    return np.where(n == 0, np.zeros_like(n), stats.chi2.ppf(1 - cl, 2 * n + 2) / 2)
