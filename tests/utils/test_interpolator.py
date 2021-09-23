import eagerpy as ep
import numpy as np
from scipy.interpolate import RegularGridInterpolator

import hypney as hp

tl = ep.numpy


def test_regular_grid_interpolator():
    """Adapted from
    https://github.com/sbarratt/torch_interpolations/blob/master/tests/test_grid_interpolator.py
    """
    points = [tl.arange(-.5, 2.5, .1) * 1., tl.arange(-.5, 2.5, .2) * 1.]
    values = (
        hp.utils.eagerpy.sin(points[0])[:, None]
        + 2 * hp.utils.eagerpy.cos(points[1])[None, :]
        + hp.utils.eagerpy.sin(5 * points[0][:, None] @ points[1][None, :]))

    X, Y = ep.meshgrid(tl.arange(-.5, 2, .1), tl.arange(-.5, 2, .1))
    points_to_interp = ep.stack([X.flatten(), Y.flatten()]).T

    gi = hp.utils.RegularGridInterpolator(points, values)
    fx = gi(points_to_interp)

    rgi = RegularGridInterpolator(
        [p.numpy() for p in points],
        [x.numpy() for x in values],
        bounds_error=False)
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