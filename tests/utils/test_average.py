import numpy as np
import eagerpy as ep

from hypney.utils.eagerpy import average_axis0

def test_average():
    x = np.arange(100).reshape((50,2))
    weights = np.arange(50)
    y = np.average(x, weights=weights, axis=0)
    y_ep = average_axis0(
        ep.astensor(x),
        weights=ep.astensor(weights)).numpy()
    np.testing.assert_almost_equal(y, y_ep)
