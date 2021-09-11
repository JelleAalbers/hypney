import collections
import typing as ty

import numpy as np

import hypney as hp


export, __all__ = hp.exporter(also_export=["DEFAULT_RATE_PARAM"])


@export
class ParameterSpec(ty.NamedTuple):
    """Description of a parameter: name, default, and limits"""

    name: str
    default: float = 0.0
    min: float = -float("inf")
    max: float = float("inf")
    share: bool = False  # Should param be shared when building mixtures?
    anchors: tuple = tuple()  # Values at which model is most accurate


DEFAULT_RATE_PARAM = ParameterSpec(name="rate", min=0.0, max=float("inf"), default=10)


@export
class Element:
    name: str = ""
    param_specs: ty.Tuple[ParameterSpec] = (DEFAULT_RATE_PARAM,)
    data: np.ndarray = None

    @property
    def param_names(self):
        return tuple([p.name for p in self.param_specs])

    @property
    def defaults(self):
        return {p.name: p.default for p in self.param_specs}

    def _set_data(self, data=None):
        if data is None:
            return

        data = self.validate_data(data)
        self.data = data
        self.init_data()

    def _set_defaults(self, new_defaults: dict):
        new_defaults = self.validate_params(new_defaults)
        self.param_specs = tuple(
            [p._replace(default=new_defaults[p.name]) for p in self.param_specs]
        )

    def init_data(self):
        pass

    def validate_params(self, params: dict) -> dict:
        if params is None:
            params = dict()
        if not isinstance(params, dict):
            raise ValueError(f"Params must be a dict, got {type(params)}")

        # Set defaults for missing params
        for p in self.param_specs:
            params.setdefault(p.name, p.default)

        # Flag spurious parameters
        spurious = set(params.keys()) - set(self.param_names)
        if spurious:
            raise ValueError(f"Unknown parameters {spurious} passed")

        return params


@export
def combine_param_specs(elements, names=None, share_all=False):
    """Return param spec, mapping for new element made of elements.
    Mapping is name -> (old name, new name)

    Clashing unshared parameter names are renamed elementname_paramname
    For shared params, defaults and bounds are taken from
    the earliest model in the combination
    """
    if names is None:
        names = [e.name if e.name else str(i) for i, e in enumerate(elements)]
    all_names = sum([list(m.param_names) for m in elements], [])
    name_count = collections.Counter(all_names)
    unique = [pn for pn, count in name_count.items() if count == 1]
    specs = []
    pmap = dict()
    seen = []
    for m, name in zip(elements, names):
        pmap[name] = []
        for p in m.param_specs:
            if p.name in unique or p.share or share_all:
                pmap[name].append((p.name, p.name))
                if p.name not in seen:
                    specs.append(p)
                    seen.append(p.name)
            else:
                new_name = name + "_" + p.name
                pmap[name].append((p.name, new_name))
                specs.append(
                    ParameterSpec(
                        name=new_name, min=p.min, max=p.max, default=p.default
                    )
                )
    return tuple(specs), pmap
