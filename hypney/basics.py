import typing as ty

import numpy as np
import hypney as hp


export, __all__ = hp.exporter(
    also_export=[
        "DEFAULT_RATE_PARAM",
        "DEFAULT_LOC_PARAM",
        "DEFAULT_SCALE_PARAM",
        "DEFAULT_OBSERVABLE",
        "RATE_LOC_PARAMS",
        "RATE_LOC_SCALE_PARAMS",
        "DEFAULT_CUT_TYPE",
        "DEFAULT_RATE_GRID",
    ]
)


@export
class NotChanged:
    """Default argument used where None would be ambiguous or unclear

    (for example, would data=None set data to None, or keep data unchanged?)
    """

    pass


@export
class Parameter(ty.NamedTuple):
    """Description of a parameter: name, default, and limits"""

    name: str
    default: float = 0.0
    min: float = -float("inf")
    max: float = float("inf")
    share: bool = False  # Should param be shared when building mixtures?
    anchors: tuple = tuple()  # Values at which model is most accurate


DEFAULT_RATE_PARAM = Parameter(name="rate", min=0, max=float("inf"), default=1.0)
DEFAULT_LOC_PARAM = Parameter(name="loc", min=-float("inf"))
DEFAULT_SCALE_PARAM = Parameter(name="scale", min=0, max=float("inf"), default=1.0)

RATE_LOC_PARAMS = (DEFAULT_RATE_PARAM, DEFAULT_LOC_PARAM)
RATE_LOC_SCALE_PARAMS = RATE_LOC_PARAMS + (DEFAULT_SCALE_PARAM,)

# open, halfopen, or closed
DEFAULT_CUT_TYPE = "halfopen"


@export
class Observable(ty.NamedTuple):
    """Description of a observable space: name and limits"""

    name: str
    min: float = -float("inf")
    max: float = float("inf")
    # Whether only integer values are allowed
    integer: bool = False


DEFAULT_OBSERVABLE = Observable(name="x", min=-float("inf"), max=float("inf"))


##
# Create a sensible anchor/interpolation grid for the rate parameter
# < 100.
##

# Start with 0.1 - 2, with 0.1 steps
_q = np.arange(0.1, 2.1, 0.1).tolist()
# Advance by 5%, or 0.25 * sigma, whichever is lower.
# Until 150, i.e. +5 sigma if true signal is 100.
# This way, we should get reasonable results for signals < 100 events
# even if there is some unknown background
while _q[-1] < 150:
    _q.append(min(_q[-1] + 0.25 * _q[-1] ** 0.5, _q[-1] * 1.05))
# Round to one decimal, and at most three significant figures,
# so results don't appear unreasonably precise
# _q = [float('%.3g' % x) for x in _q]
DEFAULT_RATE_GRID = np.unique(np.round([float("%.3g" % x) for x in _q], decimals=1))
# Prevent accidental clobbering later
DEFAULT_RATE_GRID.flags.writeable = False
del _q
