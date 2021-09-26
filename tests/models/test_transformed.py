import numpy as np
from scipy import stats

import hypney


def test_filter_params():
    m = hypney.models.uniform()
    data = m.rvs(size=100)

    m2 = m.fix_except("rate")
    assert len(m2.param_specs) == 1
    np.testing.assert_array_equal(m2.pdf(data=data), m.pdf(data=data))

    fix = dict(loc=3, scale=2)
    m2 = m.fix(fix)
    np.testing.assert_array_equal(m2.pdf(data=data), m.pdf(params=fix, data=data))


def test_negative_data():

    m = hypney.models.uniform()
    m_flip = m.transformed_data(scale=-1)

    assert m_flip.pdf(data=-0.3) == m.pdf(data=0.3)
    assert m_flip.diff_rate(data=-0.3) == m.pdf(data=0.3)
    assert m_flip.rate() == m.rate()
    assert m_flip.cdf(data=-0.3) == 1 - m.cdf(data=0.3)


def test_normalize_data():
    m = hypney.models.norm(loc=42, scale=12)
    m_std = m.normalized_data()
    assert m_std.pdf(1.3) == stats.norm.pdf(1.3)
    assert m_std.cdf(1.3) == stats.norm.cdf(1.3)
    assert m_std.mean() == 0.0
    assert m_std.std() == 1.0
