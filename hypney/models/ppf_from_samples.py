from collections import Counter

import eagerpy as ep
import numpy as np

import hypney

export, __all__ = hypney.exporter()


def compress_ppf(samples, compress="log", max_quantiles=250):
    """Return (quantiles, ppf_values) for samples

    Arguments:
     - samples: 1d numpy array
     - compress: 'log' (default), 'linear', or something else
         log: quantiles are log spaced between 1/len(samples) and 0.5,
             symmetrically flipped on 0.5 to 1
         linear: quantiles are linearly spaced
             between 1/len(samples) and 1
     - max_quantiles: Maximum number of quantiles to produce

    Adapted from Tom Abel; original at
    https://github.com/yipihey/SEDist/blob/main/SEdist.py
    """
    samples = np.asarray(samples)
    assert len(samples.shape) == 1

    lena = np.size(samples)
    max_quantiles = np.min([max_quantiles, lena])
    if lena < 2:
        raise ValueError(f"Samples must be at least of length 2, {max_quantiles}.")
    x = np.sort(samples)

    # y: indices of values whose quantiles we should keep
    if (lena > max_quantiles) and compress in ("linear", "log"):
        if compress == "linear":
            y = np.linspace(0, lena - 1, max_quantiles, dtype=int)
        else:
            assert compress == "log"
            N2 = max_quantiles // 2
            lh = np.logspace(0, np.log10(lena / 2 - 1), max_quantiles // 2, dtype=int)[
                :-1
            ]
            y = np.unique(np.hstack([lh, [lena // 2], lena - lh + 1]))
            y = y - 1
    else:
        # Few samples, just keep all of them
        y = np.arange(lena, dtype=int)

    # Use exact bounding indices for significant probability masses
    add_to_y = []
    for mass, count in Counter(x[y]).most_common():
        if count < 2:
            break
        # Found a mass (saw 2 or more equal quantiles)
        # Find exact (inclusive) indices of the mass
        left = np.searchsorted(x, mass, side="left")
        right = np.searchsorted(x, mass, side="right") - 1
        # Remove indices/quantiles currently pointing into the mass;
        y = y[x[y] != mass]
        # and include new indices at the edges of the mass
        add_to_y += [left, right]
    y = np.sort(np.concatenate([y, add_to_y])).astype(int)

    x = x[y]
    return (y + 1) / lena, x


@export
class PPFInterpolation(hypney.Model):
    """Linearly interpolate a ppf from its values at several quantiles

    Currently only supports ppf, and simple rate scaling.
    """

    param_specs = (hypney.DEFAULT_RATE_PARAM,)
    observables = (hypney.DEFAULT_OBSERVABLE,)

    def __init__(self, base_quantiles, base_ppf, *args, **kwargs):
        self.base_quantiles = base_quantiles
        self.base_ppf = base_ppf
        super().__init__(*args, **kwargs)

    def _ppf(self, params: dict) -> ep.TensorType:
        assert self._backend_name == "numpy"
        return ep.astensor(
            np.interp(self.quantiles.raw, self.base_quantiles, self.base_ppf)
        )


@export
def ppf_from_samples(samples, max_quantiles=250):
    if len(samples.shape) == 2:
        # Output of hypney's univariate simulations
        assert samples.shape[1] == 1
        samples = samples.ravel()
    quantiles, ppf = compress_ppf(samples, max_quantiles)
    return PPFInterpolation(quantiles, ppf)
