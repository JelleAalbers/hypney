def exporter(*, also_export=tuple(), export_self=False):
    """Export utility modified from https://stackoverflow.com/a/41895194
    Returns export decorator, __all__ list
    """
    all_ = list(also_export)
    if export_self:
        all_.append("exporter")

    def decorator(obj):
        all_.append(obj.__name__)
        return obj

    return decorator, all_


from . import utils

from .basics import *

from .model import *
from . import models

from .statistic import *
from . import statistics

from . import estimators
