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
    x = np.linspace(-5, 5, 10)

    np.testing.assert_almost_equal(m_cut.rate(), 0.5)
    np.testing.assert_almost_equal(m_cut.diff_rate(x), m_half.diff_rate(x) / 2)

    # Extra vectorization tests... last one catches low, high = corner_cdf unpacking bug
    np.testing.assert_almost_equal(m_cut.rate(rate=2), 1)
    np.testing.assert_almost_equal(m_cut.rate(rate=[2]), np.array([1]))
    np.testing.assert_almost_equal(m_cut.cdf(0), 0)
    np.testing.assert_almost_equal(m_cut.cdf([0, float("inf")]), np.array([0, 1]))
    np.testing.assert_almost_equal(
        m_cut.cdf([0, float("inf")], rate=[2, 4]), np.array([[0, 1], [0, 1]])
    )

    for params in dict(a=0, b=float("inf")), dict(a=-1, b=1):
        m_cut = m_base.cut(params["a"], params["b"])
        m_trunc = hypney.models.truncnorm(**params)
        np.testing.assert_almost_equal(m_cut.pdf(x), m_trunc.pdf(x, params))
        np.testing.assert_almost_equal(m_cut.logpdf(x), m_trunc.logpdf(x, params))
        np.testing.assert_almost_equal(m_cut.cdf(x), m_trunc.cdf(x, params))

        q = np.linspace(0, 1, 100)
        np.testing.assert_almost_equal(m_cut.ppf(q), m_trunc.ppf(q, params))

    # Test cutting a combined model (caught a bug with cut_efficiency > 1 once)
    m = (
        hypney.models.uniform().fix_except("rate") + hypney.models.uniform().freeze()
    ).cut(None, None)
    assert m.rate(rate=1) == m.rate(rate=[1])[0]
