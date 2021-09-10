"""Interpolate functions outputing arbitrary-shaped arrays in arbitrary dimensions

adapted from https://github.com/JelleAalbers/blueice/blob/master/blueice/pdf_morphers.py
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator

import hypney

export, __all__ = hypney.exporter()


class Morpher:
    def __init__(self, parameters):
        """Initialize the interpolator, telling it which shape_parameters we're going to use

        anchors_per_parameter: sequence of anchor points per parameter
        """
        raise NotImplementedError

    def get_anchor_points(self):
        """Returns list of tuples of anchor coordinates
        """
        raise NotImplementedError

    def make_interpolator(self, f, inputs_at_anchors=None, extra_dims=tuple()):
        """Return interpolator of f between anchor points.

        The interpolator is vectorized, so t will add one dimension for scalar inputs.

        Args:
         - f: Function taking one argument, and returning an extra_dims shaped array.
         - inputs_at_anchors: dict {anchor: f_input} mapping anchor values
            inputs to function arguments. If None, anchor values themselves are used.
         - extra_dims: tuple of integers, shape of return value of f
            Leave to the default if f returns a scalar

        """
        raise NotImplementedError


@export
class GridInterpolator(Morpher):
    @hypney.inherit_docstring_from(Morpher)
    def __init__(self, anchors_per_parameter):
        # Compute the regular grid of anchor models at the specified anchor points
        self.anchor_z_arrays = [
            np.array(list(sorted(anchors)))
            for anchors in anchors_per_parameter]
        self.anchor_z_grid = arrays_to_grid(self.anchor_z_arrays)

    @hypney.inherit_docstring_from(Morpher)
    def get_anchor_points(self):
        return [zs for _, zs in self._anchor_grid_iterator()]

    @hypney.inherit_docstring_from(Morpher)
    def make_interpolator(self, f, inputs_at_anchors=None, extra_dims=tuple()):
        if inputs_at_anchors is None:
            inputs_at_anchors = {z: z for z in self.get_anchor_points()}

        # Allocate an array which will hold the scores at each anchor model
        anchor_scores = np.zeros(list(self.anchor_z_grid.shape)[:-1] + list(extra_dims))


        # Iterate over the anchor grid points
        for anchor_grid_index, _zs in self._anchor_grid_iterator():
            # Compute f at this point, and store it in anchor_scores
            anchor_scores[tuple(anchor_grid_index + [slice(None)] * len(extra_dims))] = \
                f(inputs_at_anchors[tuple(_zs)])

        return RegularGridInterpolator(self.anchor_z_arrays, anchor_scores)

    def _anchor_grid_iterator(self):
        """Iterates over the anchor grid, yielding index, z-values"""
        fake_grid = np.zeros(list(self.anchor_z_grid.shape)[:-1])
        it = np.nditer(fake_grid, flags=["multi_index"])
        while not it.finished:
            anchor_grid_index = list(it.multi_index)
            yield (
                anchor_grid_index,
                tuple(self.anchor_z_grid[tuple(anchor_grid_index + [slice(None)])]),
            )
            it.iternext()


def arrays_to_grid(arrs):
    """Convert a list of n 1-dim arrays to an n+1-dim. array, where last dimension denotes coordinate values at point.
    """
    return np.stack(np.meshgrid(*arrs, indexing="ij"), axis=-1)
