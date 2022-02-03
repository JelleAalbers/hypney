from functools import partial
import warnings

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
