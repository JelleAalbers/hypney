import collections
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
    ]
)


@export
class NotChanged:
    """Default argument used where None would be ambiguous or unclear

    (for example, would data=None set data to None, or keep data unchanged?)
    """

    pass


@export
class ParameterSpec(ty.NamedTuple):
    """Description of a parameter: name, default, and limits"""

    name: str
    default: float = 0.0
    min: float = -float("inf")
    max: float = float("inf")
    share: bool = False  # Should param be shared when building mixtures?
    anchors: tuple = tuple()  # Values at which model is most accurate


DEFAULT_RATE_PARAM = ParameterSpec(name="rate", min=0.0, max=float("inf"), default=1.0)
DEFAULT_LOC_PARAM = ParameterSpec(name="loc", min=-float("inf"))
DEFAULT_SCALE_PARAM = ParameterSpec(name="scale", min=1e-6, max=float("inf"), default=1)

RATE_LOC_PARAMS = (DEFAULT_RATE_PARAM, DEFAULT_LOC_PARAM)

RATE_LOC_SCALE_PARAMS = RATE_LOC_PARAMS + (DEFAULT_SCALE_PARAM,)


@export
class Observable(ty.NamedTuple):
    """Description of a observable space: name and limits"""

    name: str
    min: float = -float("inf")
    max: float = float("inf")
    # Whether only integer values are allowed
    integer: bool = False


DEFAULT_OBSERVABLE = Observable(name="x", min=-float("inf"), max=float("inf"))


@export
class NoCut:
    """Instruction to not cut data"""

    pass
