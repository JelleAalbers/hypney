import math
import functools
import typing as ty

import hypney

import eagerpy as ep
import numpy as np
from scipy import special

export, __all__ = hypney.exporter()


@export
def astensor(x: ty.Sequence, tensorlib=None, match_type=None):
    """Convert x to an eagerpy tensor specified by tensorlib or match_type.

    Args:
     - tensorlib: name of module or ep.module to use
     - match_type: other tensor whose type to match.
    """
    if match_type is None:
        if tensorlib is None:
            raise ValueError("pass tensorlib or match_type")
        # Create a dummy tensor of the given tensorlib
        if isinstance(tensorlib, str):
            tensorlib = getattr(ep, tensorlib)
        match_type = tensorlib.zeros(0)
    if isinstance(x, type(match_type)):
        return x
    if not isinstance(match_type, ep.Tensor):
        match_type = ep.astensor(match_type)
    if isinstance(x, ep.Tensor):
        # Eagerpy tensor with different backend
        x = x.numpy()

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
    """Return raw tensor from x, unless it already is a raw tensor"""
    try:
        return x.raw
    except AttributeError:
        return x


@export
def ensure_numpy(x):
    """Return numpy array from x, unless it already is a numpy array"""
    try:
        return x.numpy()
    except AttributeError:
        return x


@export
def ensure_float(x):
    """Return a simple float from a 0-dimensional array, tensor, or float"""
    if isinstance(x, ep.Tensor):
        x = x.numpy()
    if isinstance(x, np.ndarray):
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
def average_axis0(x, weights):
    weights = astensor(weights, match_type=x)
    # Add as many ones to shape as we need
    while len(weights.shape) < len(x.shape):
        weights = weights[..., None]
    return ep.sum(x * weights, axis=0) / ep.sum(weights, axis=0)


@export
def split(x, *args, **kwargs):
    return [ep.astensor(x) for x in tensorlib(x).split(x.raw, *args, **kwargs)]


@export
def broadcast_to(x, shape):
    return tensorlib(x).broadcast_to(x.raw, shape)


@export
def tensorlib(x: ep.TensorType):
    if isinstance(x, str):
        return getattr(ep, x)
    if isinstance(x, ep.modules.ModuleWrapper):
        return x

    if not isinstance(x, ep.Tensor):
        x = ep.astensor(x)

    if isinstance(x, ep.NumPyTensor) or x is None:
        return ep.numpy
    elif isinstance(x, ep.PyTorchTensor):
        return ep.torch
    elif isinstance(x, ep.TensorFlowTensor):
        return ep.tensorflow
    elif isinstance(x, ep.JAXTensor):
        return ep.jax.numpy
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
