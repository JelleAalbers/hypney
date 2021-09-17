import typing as ty

import hypney

import eagerpy as ep
import numpy as np

export, __all__ = hypney.exporter()


@export
def sequence_to_tensor(x: ty.Sequence, *, match_type: ep.TensorType):
    if isinstance(x, type(match_type)):
        return x
    return match_type.zeros(len(x)) + x


@export
def ep_average_axis0(x, weights=None):
    x = ep.astensor(x)
    stacked = ep.stack(x, axis=0)
    weights = sequence_to_tensor(weights, match_type=x)
    weights = weights.reshape(tuple([len(stacked)] + [1] * (len(stacked.shape) - 1)))
    return ep.sum(stacked * weights, axis=0)


@export
def ep_split(x, *args, **kwargs):
    return [ep.astensor(x) for x in tensorlib(x).split(x.raw, *args, **kwargs)]


@export
def tensorlib(x: ep.TensorType):
    if isinstance(x, ep.NumPyTensor):
        return np
    elif isinstance(x, ep.PyTorchTensor):
        return ep.torch
    elif isinstance(x, ep.TensorFlowTensor):
        return ep.tensorflow
    elif isinstance(x, ep.JAXTensor):
        return ep.jax
    raise ValueError(f"Unknown tensor type {type(x)}")
