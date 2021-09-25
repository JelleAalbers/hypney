import itertools
import eagerpy as ep

import hypney

export, __all__ = hypney.exporter()


@export
class RegularGridInterpolator:
    """Eagerpy RegularGridInterpolator

    Modified from Shane Barratt's very nice torch interpolation code at
    https://github.com/sbarratt/torch_interpolations/blob/master/torch_interpolations/multilinear.py
    """

    def __init__(self, points, values=None):
        self.points = points

        assert isinstance(self.points, tuple) or isinstance(self.points, list)

        self.n_dimensions = len(self.points)

        if values is not None:
            self.values = values
            assert isinstance(self.values, ep.Tensor)
            # Values may be higher-dimensional than the points, but not less
            self.grid_shape = list(self.values.shape)
            assert len(self.grid_shape) >= self.n_dimensions

        for i, p in enumerate(self.points):
            assert isinstance(p, ep.Tensor)
            assert p.shape[0] == self.values.shape[i]

    def __call__(self, points_to_interp):

        if len(points_to_interp.shape) == 1:
            assert self.n_dimensions == 1
            points_to_interp = points_to_interp[:, None]

        # points_to_interp should now be (n_points, n_dims),
        # matching scipy RegularGridInterpolator.__call__.

        points_to_interp = points_to_interp.T

        # points_to_interp should now be: (n_dims, n_points),
        # matching Barratt's transposed convention.

        assert self.points is not None

        assert points_to_interp.shape[0] == self.n_dimensions
        K = points_to_interp[0].shape[0]
        for x in points_to_interp:
            assert x.shape[0] == K

        idxs = []
        dists = []
        overalls = []
        for p, x in zip(self.points, points_to_interp):
            idx_right = hypney.utils.eagerpy.bucketize(x, p)
            idx_right = ep.where(idx_right >= p.shape[0], p.shape[0] - 1, idx_right)
            idx_left = (idx_right - 1).clip(0, p.shape[0] - 1)

            dist_left = x - p[idx_left]
            dist_right = p[idx_right] - x

            dist_left = dist_left.clip(0, None)
            dist_right = dist_right.clip(0, None)

            both_zero = (dist_left == 0) * (dist_right == 0)
            dist_left = ep.where(both_zero, 1, dist_left)
            dist_right = ep.where(both_zero, 1, dist_right)

            idxs.append((idx_left, idx_right))
            dists.append((dist_left, dist_right))
            overalls.append(dist_left + dist_right)

        numerator = 0.0
        for indexer in itertools.product([0, 1], repeat=self.n_dimensions):
            as_s = [idx[onoff] for onoff, idx in zip(indexer, idxs)]
            bs_s = [dist[1 - onoff] for onoff, dist in zip(indexer, dists)]

            values = self.get_values(as_s)
            weights = ep.prod(ep.stack(bs_s), axis=0)

            # If values are arrays, we'll need to expand_dims the weights
            new_dims = list(range(len(weights.shape), len(values.shape)))
            weights = weights.expand_dims(axis=new_dims)

            numerator += values * weights
        denominator = ep.prod(ep.stack(overalls), axis=0)
        return numerator / denominator.expand_dims(axis=new_dims)

    def get_values(self, list_of_indices):
        assert self.values is not None
        return self.values[tuple([x.raw for x in list_of_indices])]


@export
class InterpolatorBuilder:
    def __init__(self, anchors_per_parameter):
        """Initialize the interpolator, telling it which parameters we're going to use

        anchors_per_parameter: sequence of anchor points per parameter
        """
        # Compute the regular grid of anchor models at the specified anchor points
        self.anchor_tuples = [
            tuple(list(sorted(anchors))) for anchors in anchors_per_parameter
        ]

    def make_interpolator(self, f, tensorlib=ep.numpy):
        """Return interpolator of f between anchor points.

        The interpolator is vectorized, so t will add one dimension for scalar inputs.

        Args:
         - f: Function taking one argument, and returning an extra_dims shaped array.

        """
        # Compute f at each anchor point
        results = [f(tuple(z)) for z in itertools.product(*self.anchor_tuples)]

        # Convert to flat tensor
        try:
            extra_dims = results[0].shape
            results = ep.stack(results)
        except AttributeError:
            extra_dims = tuple()
            # Can't stack with eagerpy, results[0] is a scalar
            results = tensorlib.stack(results)

        # Reshape to (n_dim0, n_dim1, ..., *extra_dims) tensor
        grid_dimensions = (
            tuple([len(anchors) for anchors in self.anchor_tuples]) + extra_dims
        )
        results = results.reshape(grid_dimensions)

        return RegularGridInterpolator(
            [
                hypney.utils.eagerpy.sequence_to_tensor(
                    x, match_type=tensorlib.zeros(1)
                )
                for x in self.anchor_tuples
            ],
            results,
        )
