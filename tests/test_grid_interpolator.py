import pytest
import numpy as np

import hypney


def test_grid_interpolator():
    with pytest.raises(ValueError):
        hypney.GridInterpolator(anchors_per_parameter=dict())

    itp = hypney.GridInterpolator([(-1, 0, 1)])
    anchor_points = itp.get_anchor_points()
    assert isinstance(anchor_points, list)
    assert isinstance(anchor_points[0], tuple)

    def scalar_f(z):
        return z[0]

    z = np.array([1, 0, -1, 0, 1, 1, -1])

    scalar_itp = itp.make_interpolator(scalar_f)
    np.testing.assert_array_equal(scalar_itp(z), z)

    scalar_itp_2 = itp.make_interpolator(
        scalar_f, extra_dims=[], inputs_at_anchors={z: (z[0] ** 2,) for z in anchor_points}
    )
    np.testing.assert_array_equal(scalar_itp_2(z), z ** 2)

    def matrix_f(z):
        return np.ones((2, 2)) * z[0]

    matrix_itp = itp.make_interpolator(
        matrix_f, extra_dims=[2, 2], inputs_at_anchors={z: (z[0] ** 2,) for z in anchor_points}
    )
    # np.testing.assert_array_equal(matrix_itp([0, 0, 0]), np.ones((2, 2)))
    np.testing.assert_array_equal(
        matrix_itp(z), (z ** 2)[:, None, None] * np.ones((1, 2, 2))
    )
