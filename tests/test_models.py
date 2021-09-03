import hypney

import numpy as np

def test_uniform():
    m = hypney.Uniform()

    assert m.expected_count() == 0.

    data = m.simulate()
    assert data.shape == (0, 1)
    data = m.simulate_n(5)
    assert data.shape == (5, 1)

    assert m.pdf(0.) == m.pdf([0.]) == m.pdf(np.array([0])) == m.pdf(np.array([[0]]))
    assert m.pdf(0.) == 1.

    assert m.diff_rate(0.) == 0.

    m = hypney.Uniform(expected_events=100)
    assert m.expected_count() == 100.
    assert m.simulate().shape[0] > 0
