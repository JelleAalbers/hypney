import eagerpy as ep
import numpy as np
from scipy.interpolate import RegularGridInterpolator

import hypney

tl = ep.numpy


def test_regular_grid_interpolator():
    """Adapted from
    https://github.com/sbarratt/torch_interpolations/blob/master/tests/test_grid_interpolator.py
    """
    points = [tl.arange(-0.5, 2.5, 0.1) * 1.0, tl.arange(-0.5, 2.5, 0.2) * 1.0]
    values = (
        hypney.utils.eagerpy.sin(points[0])[:, None]
        + 2 * hypney.utils.eagerpy.cos(points[1])[None, :]
        + hypney.utils.eagerpy.sin(5 * points[0][:, None] @ points[1][None, :])
    )

    X, Y = ep.meshgrid(tl.arange(-0.5, 2, 0.1), tl.arange(-0.5, 2, 0.1))
    points_to_interp = ep.stack([X.flatten(), Y.flatten()]).T

    gi = hypney.utils.interpolation.RegularGridInterpolator(points, values)
    fx = gi(points_to_interp)

    rgi = RegularGridInterpolator(
        [p.numpy() for p in points], [x.numpy() for x in values], bounds_error=False
    )
    rfx = rgi(points_to_interp.numpy())

    np.testing.assert_allclose(rfx, fx.numpy(), atol=1e-6)


# TODO: port derivative test to eagerpy
# note that points_to_interp has to be transposed
#
# def test_regular_grid_interpolator_derivative():
#     points = [torch.arange(-.5, 2.5, .5) * 1., torch.arange(-.5, 2.5, .5) * 1.]
#     values = torch.sin(points[0])[:, None] + 2 * torch.cos(points[1])[None, :] + torch.sin(5 * points[0][:, None] @ points[1][None, :])
#     values.requires_grad_(True)
#
#     X, Y = np.meshgrid(np.arange(-.5, 2, .19), np.arange(-.5, 2, .19))
#     points_to_interp = [torch.from_numpy(
#         X.flatten()).float(), torch.from_numpy(Y.flatten()).float()]
#
#     def f(values):
#         return torch_interpolations.RegularGridInterpolator(
#             points, values)(points_to_interp)
#
#     torch.autograd.gradcheck(f, (values,), eps=1e-5, atol=1e-1, rtol=1e-1)


def test_interpolator_builder():

    itp = hypney.utils.interpolation.InterpolatorBuilder([(-1, 0, 1)])

    def scalar_f(z):
        return z[0]

    z = ep.astensor(np.array([1, 0, -1, 0, 1, 1, -1]))

    scalar_itp = itp.make_interpolator(scalar_f)
    np.testing.assert_array_equal(scalar_itp(z).numpy(), z.numpy())

    def matrix_f(z):
        return ep.astensor(np.ones((2, 2)) * z[0])

    matrix_itp = itp.make_interpolator(matrix_f)
    np.testing.assert_array_equal(
        matrix_itp(z).numpy(), z[:, None, None].numpy() * np.ones((1, 2, 2))
    )

    # What happened here? Does the test not make sense or did the API change?
    # np.testing.assert_array_equal(
    #     matrix_itp(ep.numpy.array([0, 0, 0])).numpy(),
    #     np.ones((2, 2)))
