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


@export
def make_rate_grid(max_mu=1200):
    # Start with 0.1 - 2, with 0.1 steps
    _q = np.arange(0.1, 2.1, 0.1).tolist()
    # Advance by 5% each step until 1200, i.e. +~6 sigma if true signal is 1000.
    while _q[-1] < max_mu:
        _q.append(_q[-1] * 1.05)
    # Round to one decimal, and at most three significant figures,
    # so results don't appear unreasonably precise
    return np.unique(np.round([float("%.3g" % x) for x in _q], decimals=1))


DEFAULT_RATE_GRID = make_rate_grid()
# Prevent accidental clobbering later
DEFAULT_RATE_GRID.flags.writeable = False
