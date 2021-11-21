import hypney

import numpy as np
from scipy import stats


def test_cut():
    m_base = hypney.models.norm()
    m_cut = m_base.cut(0, None)
    assert isinstance(m_cut.simulate(), np.ndarray)
    assert m_cut._cut == ((0, float("inf")),)
    assert m_cut.cut_efficiency() == 0.5
    assert m_cut.cut_efficiency(loc=1) == stats.norm(loc=1).sf(0)
    assert m_cut.rvs(100).min() >= 0

    m_half = hypney.models.halfnorm()
    x = np.linspace(-5, 5, 100)

    np.testing.assert_almost_equal(m_cut.rate(), 0.5)
    np.testing.assert_almost_equal(m_cut.diff_rate(x), m_half.diff_rate(x) / 2)

    for params in dict(a=0, b=float("inf")), dict(a=-1, b=1):
        m_cut = m_base.cut(params["a"], params["b"])
        m_trunc = hypney.models.truncnorm(**params)
        np.testing.assert_almost_equal(m_cut.pdf(x), m_trunc.pdf(x, params))
        np.testing.assert_almost_equal(m_cut.logpdf(x), m_trunc.logpdf(x, params))
        np.testing.assert_almost_equal(m_cut.cdf(x), m_trunc.cdf(x, params))

        q = np.linspace(0, 1, 100)
        np.testing.assert_almost_equal(m_cut.ppf(q), m_trunc.ppf(q, params))
