import numpy as np
from scipy import stats

import hypney


def test_negative_data():

    m = hypney.models.uniform()
    m_flip = m.shift_and_scale(scale=-1)

    assert m_flip.pdf(data=-0.3) == m.pdf(data=0.3)
    np.testing.assert_almost_equal(m_flip.logpdf(data=-0.3), m.logpdf(data=0.3))
    assert m_flip.diff_rate(data=-0.3) == m.pdf(data=0.3)
    assert m_flip.rate() == m.rate()
    assert m_flip.cdf(data=-0.3) == 1 - m.cdf(data=0.3)


def test_normalize_data():
    m = hypney.models.norm(loc=42, scale=12)
    m_std = m.normalized_data()
    assert m_std.pdf(1.3) == stats.norm.pdf(1.3)
    np.testing.assert_almost_equal(m_std.logpdf(1.3), stats.norm.logpdf(1.3))
    assert m_std.cdf(1.3) == stats.norm.cdf(1.3)
    assert m_std.mean() == 0.0
    assert m_std.std() == 1.0
