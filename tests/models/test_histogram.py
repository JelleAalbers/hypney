import numpy as np
from scipy import stats

import hypney


def test_from_histogram():
    hist, edges = np.array([1, 2, 1]), np.array([0, 1, 2, 3])
    m = hypney.models.from_histogram(hist, edges)
    data = m.simulate()
    np.testing.assert_equal(
        m.pdf(data), stats.rv_histogram((hist, edges),).pdf(data[:, 0]),
    )
