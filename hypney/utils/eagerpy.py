import math
import functools
import typing as ty

import hypney

import eagerpy as ep
import numpy as np
from scipy import special

export, __all__ = hypney.exporter()


@export
def to_tensor(x: ty.Sequence, *, match_type: ep.TensorType):
    if isinstance(x, type(match_type)):
        return x
    if not isinstance(match_type, ep.Tensor):
        match_type = ep.astensor(match_type)
    if isinstance(x, ep.Tensor):
        raise ValueError("Passed tensor of incompatible type")

    # Check, why is this not included in eagerpy?
    # Maybe it is and I haven't found it?
    # e.g. ep.JAXTensor([1,2,3]) gives something pathological
    # (list wrapped in a JAXTensor, should be wrapped jax array)
    if isinstance(match_type, ep.NumPyTensor):
        return ep.numpy.asarray(x)
    # ep.jax etc. will automatically wrap return values as eagerpy tensors
    if isinstance(match_type, ep.JAXTensor):
        return ep.jax.numpy.asarray(x, dtype=match_type.dtype)
    if isinstance(match_type, ep.PyTorchTensor):
        return ep.torch.as_tensor(x, dtype=match_type.dtype)
    if isinstance(match_type, ep.TensorFlowTensor):
        return ep.tensorflow.convert_to_tensor(x, dtype=match_type.dtype)
    raise ValueError(f"match_type of unknown type {type(match_type)}")


@export
def ensure_raw(x):
    try:
        return x.raw
    except AttributeError:
        return x


@export
def ensure_numpy(x):
    try:
        return x.numpy()
    except AttributeError:
        return x


@export
def ensure_numpy_float(x):
    if isinstance(x, ep.Tensor):
        x = x.numpy()
    if isinstance(x, np.ndarray):
        assert x.shape == 0
        return x.item()
    return x


@export
def sin(x):
    return ep.astensor(tensorlib(x).sin(x.raw))


@export
def cos(x):
    return ep.astensor(tensorlib(x).cos(x.raw))


@export
def np64(x):
    return x.numpy().astype(np.float64)


@export
def average_axis0(x, weights=None):
    x = ep.astensor(x)
    stacked = ep.stack(x, axis=0)
    weights = to_tensor(weights, match_type=x)
    weights = weights.reshape(tuple([len(stacked)] + [1] * (len(stacked.shape) - 1)))
    return ep.sum(stacked * weights, axis=0)


@export
def split(x, *args, **kwargs):
    return [ep.astensor(x) for x in tensorlib(x).split(x.raw, *args, **kwargs)]


@export
def tensorlib(x: ep.TensorType):
    if isinstance(x, ep.NumPyTensor) or x is None:
        return ep.numpy
    elif isinstance(x, ep.PyTorchTensor):
        return ep.torch
    elif isinstance(x, ep.TensorFlowTensor):
        return ep.tensorflow
    elif isinstance(x, ep.JAXTensor):
        return ep.jax
    raise ValueError(f"Unknown tensor type {type(x)}")


@export
def bucketize(x, p):
    # TODO check these are really equivalent
    if isinstance(x, ep.NumPyTensor):
        return ep.numpy.searchsorted(p, x)
    elif isinstance(x, ep.TensorFlowTensor):
        return ep.tf.bucketize(x, p)
    elif isinstance(x, ep.PyTorchTensor):
        return ep.torch.bucketize(x, p)
    elif isinstance(x, ep.JAXTensor):
        return ep.jax.numpy.searchsorted(p, x)
    raise TypeError(f"Unknown tensor type {x}")


@export
def logsumexp(tensor, axis=0):
    if isinstance(tensor, ep.NumPyTensor):
        result = special.logsumexp(tensor.raw, axis=axis)
        return ep.astensor(np.asarray(result))
    # TODO: tf/torch/jax only support two-argument function logaddexp.
    # Do something horrible for now:
    return ep.log(ep.sum(ep.exp(tensor), axis=axis))
