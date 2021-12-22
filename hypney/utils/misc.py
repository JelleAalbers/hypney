from functools import partial
import warnings


def progress_iter(progress=False):
    if progress is True:
        try:
            from tqdm import tqdm

            return tqdm
        except ImportError:
            warnings.warn("Progress bar requested but tqdm did not import")
            return lambda x: x
    elif progress:
        # Custom progress iterator, e.g. tqdm.notebook
        return progress
    else:
        return lambda x: x
