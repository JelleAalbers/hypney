from functools import partial
import warnings


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
