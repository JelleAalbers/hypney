import hypney

import numpy as np
from scipy import stats


def test_cut():
    m_cut = hypney.models.norm().cut(0, None)
    assert m_cut.cut == ((0, float("inf")),)
    assert m_cut.cut_efficiency() == 0.5
    assert m_cut.cut_efficiency(loc=1) == stats.norm(loc=1).sf(0)

    m_half = hypney.models.halfnorm()
    x = np.linspace(-5, 5, 100)
    np.testing.assert_almost_equal(m_cut.pdf(x), m_half.pdf(x))
    np.testing.assert_almost_equal(m_cut.diff_rate(x), m_half.diff_rate(x) / 2)

    np.testing.assert_almost_equal(m_cut.rate(), 0.5)
